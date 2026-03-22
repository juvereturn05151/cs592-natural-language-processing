"""
File Name:    knowledge_graph.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""


import json
import csv
import re
from pathlib import Path
from collections import defaultdict

import src.Project2.Project2Globals as Globals
import src.Project2.Data_Extraction.data_extractor as DataExtractor
from src.Project2.Play_Configs.play_configs import PLAY_CONFIGS

try:
    from gqlalchemy import Memgraph
    MEMGRAPH_AVAILABLE = True
except ImportError:
    MEMGRAPH_AVAILABLE = False
    print("WARNING: gqlalchemy not installed. Run: pip install gqlalchemy")
    print("Running in DRY-RUN mode.\n")

MEMGRAPH_HOST = "127.0.0.1"
MEMGRAPH_PORT = 7687

KNOWN_FAMILIES = {"Montague", "Capulet", "Macbeth"}
REL_FIELDS = [
    "play", "from_node", "from_label",
    "rel_type", "to_node", "to_label", "extraction_source"
]


def execute(mg, query: str, params):
    params = params or {}
    if MEMGRAPH_AVAILABLE and mg:
        mg.execute(query, params)
    else:
        display = query.strip()
        for k, v in params.items():
            display = display.replace(f"${k}", f'"{v}"')
        print(f"  CYPHER: {display}")


def connect_memgraph():
    if not MEMGRAPH_AVAILABLE:
        return None
    try:
        mg = Memgraph(host=MEMGRAPH_HOST, port=MEMGRAPH_PORT)
        mg.execute("RETURN 1")
        print("Connected to Memgraph successfully.")
        return mg
    except Exception as e:
        print(f"Could not connect to Memgraph ({e})")
        print("Running in DRY-RUN mode.\n")
        return None


def clear_old_data(mg):
    execute(mg, "MATCH (n) DETACH DELETE n")


def create_node(mg, label: str, name: str):
    execute(mg, f"MERGE (n:{label} {{name: $name}})", {"name": name})


def sanitise_rel_type(rel_type: str) -> str:
    rel_type = rel_type.split(",")[0]
    rel_type = re.sub(r"[\s\-]+", "_", rel_type)
    rel_type = re.sub(r"[^A-Z0-9_]", "", rel_type.upper())
    return re.sub(r"_+", "_", rel_type).strip("_")


def create_relationship(mg, from_name, from_label, rel_type, to_name, to_label):
    rel_type = sanitise_rel_type(rel_type)
    execute(
        mg,
        f"""
        MATCH (a:{from_label} {{name: $from_name}})
        MATCH (b:{to_label} {{name: $to_name}})
        MERGE (a)-[:{rel_type}]->(b)
        """,
        {"from_name": from_name, "to_name": to_name},
    )


def create_play_node(mg, play_title: str):
    create_node(mg, "PLAY", play_title)


def link_character_to_play(mg, char_name: str, play_title: str):
    create_relationship(mg, char_name, "PERSON", "APPEARS_IN", play_title, "PLAY")


def load_rules(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_entities_csv(path: Path) -> dict:
    grouped = defaultdict(set)
    if not path.exists():
        print(f"  WARNING: CSV not found: {path}")
        return grouped

    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip()
            name = row["entity_text"].strip()

            if (
                not label or not name or len(name) < 3
                or not any(c.isalpha() for c in name)
                or name[0] in ",-—.;:!?'\"()/\\"
            ):
                continue

            if label == "PERSON":
                name = name.upper()

            grouped[label].add(name)

    return grouped


def load_cast_relationships(path: Path) -> list:
    if not path.exists():
        print(f"  WARNING: Cast relationships CSV not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_relationships_csv(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{
        "play": r.get("play", ""),
        "from_node": r.get("source", ""),
        "from_label": r.get("source_label", ""),
        "rel_type": r.get("rel_type", ""),
        "to_node": r.get("target", ""),
        "to_label": r.get("target_label", ""),
        "extraction_source": r.get("extraction_source", ""),
    } for r in records]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  → Saved {len(rows):,} relationships : {path.name}")


def add_record(records, play, source, source_label, rel_type, target, target_label, extraction_source):
    records.append({
        "play": play,
        "source": source,
        "source_label": source_label,
        "rel_type": rel_type,
        "target": target,
        "target_label": target_label,
        "extraction_source": extraction_source,
    })


def insert_edge(mg, records, play_title, fn, fl, rel, tn, tl, source_tag, verbose_prefix=""):
    create_node(mg, fl, fn)
    create_node(mg, tl, tn)
    create_relationship(mg, fn, fl, rel, tn, tl)
    add_record(records, play_title, fn, fl, rel, tn, tl, source_tag)
    if verbose_prefix is not None:
        print(f"    {verbose_prefix}{fn:20s} -[{rel}]-> {tn}")
    return 1


def insert_edges(mg, records, play_title, edges, source_tag, verbose_prefix=""):
    count = 0
    for fn, fl, rel, tn, tl in edges:
        count += insert_edge(
            mg, records, play_title, fn, fl, rel, tn, tl, source_tag, verbose_prefix
        )
    return count


def extract_family_nodes(entity_groups: dict, known_families: set[str]) -> list[str]:
    persons = {n.upper() for n in entity_groups.get("PERSON", set())}
    families = []

    for family in known_families or KNOWN_FAMILIES:
        upper = family.upper()
        if upper in persons:
            related = [p for p in persons if upper in p and p != upper]
            if related:
                families.append(upper)
        else:
            families.append(upper)

    return families


def extract_knows_from_dialogue(scenes: list, entity_groups: dict) -> list:
    speaker_re = re.compile(r"([A-Z][A-Z\s]{2,})\.\s+")
    known_persons = {n.upper() for n in entity_groups.get("PERSON", set())}
    triples, seen = [], set()

    for scene_text in scenes:
        speakers = sorted({
            m.group(1).strip().upper()
            for m in speaker_re.finditer(scene_text)
            if m.group(1).strip().upper() in known_persons
        })

        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                a, b = speakers[i], speakers[j]
                if (a, "KNOWS", b) not in seen:
                    seen.add((a, "KNOWS", b))
                    seen.add((b, "KNOWS", a))
                    triples.append((a, "PERSON", "KNOWS", b, "PERSON"))

    print(f"    KNOWS from dialogue: {len(triples)} pairs")
    return triples


def extract_relationships(scenes: list, entity_groups: dict, nlp) -> list:
    entity_lookup = {
        name.lower(): (name, label)
        for label, names in entity_groups.items()
        for name in names
    }

    verb_map = {
        "serve": "SERVES", "follow": "FOLLOWS", "obey": "OBEYS",
        "attend": "SERVES", "swear": "LOYAL_TO", "pledge": "LOYAL_TO",
        "betray": "BETRAYS", "deceive": "DECEIVES", "trick": "DECEIVES",
        "conspire": "CONSPIRES_WITH", "command": "COMMANDS", "banish": "BANISHES",
        "fight": "FIGHTS", "oppose": "FIGHTS", "challenge": "FIGHTS",
        "defeat": "DEFEATS", "flee": "FLEES_FROM", "know": "KNOWS",
        "meet": "MEETS", "trust": "TRUSTS", "fear": "FEARS", "hate": "HATES",
        "help": "HELPS", "seek": "SEEKS", "warn": "WARNS", "curse": "CURSES",
        "send": "SENDS", "accuse": "ACCUSES", "suspect": "SUSPECTS",
        "protect": "PROTECTS", "forgive": "FORGIVES", "greet": "MEETS",
        "visit": "MEETS",
    }

    triples, seen = [], set()
    speaker_re = re.compile(r"([A-Z][A-Z\s]{2,})\.\s+")

    def add_triple(subj, rel, obj):
        if not subj or not obj or subj[0] == obj[0]:
            return
        key = (subj[0], rel, obj[0])
        if key not in seen and subj[1] == "PERSON" and obj[1] == "PERSON":
            seen.add(key)
            triples.append((subj[0], subj[1], rel, obj[0], obj[1]))

    def resolve_entity(token):
        chunk = " ".join(
            t.text for t in token.subtree if t.dep_ in ("compound", "nn") or t == token
        ).strip()
        for candidate in (chunk, token.text):
            if candidate.lower() in entity_lookup:
                return entity_lookup[candidate.lower()]
        for t in token.subtree:
            if t.text.lower() in entity_lookup:
                return entity_lookup[t.text.lower()]
        return None

    def get_utterances(scene_text):
        parts = speaker_re.split(scene_text)
        return [
            (parts[i].strip(), parts[i + 1].strip())
            for i in range(1, len(parts) - 1, 2)
            if parts[i].strip() and parts[i + 1].strip()
        ]

    for scene_text in scenes:
        for _, utt_text in get_utterances(scene_text):
            doc = nlp(utt_text)
            for sent in doc.sents:
                sent_entities = [
                    entity_lookup[ent.text.lower()]
                    for ent in sent.ents
                    if ent.text.lower() in entity_lookup
                ]

                for token in sent:
                    if token.pos_ != "VERB":
                        continue
                    rel = verb_map.get(token.lemma_.lower())
                    if not rel:
                        continue

                    subj = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
                    obj = next((c for c in token.children if c.dep_ in ("dobj", "pobj", "attr", "oprd")), None)
                    if not subj or not obj:
                        continue

                    subj_match = resolve_entity(subj)
                    obj_match = resolve_entity(obj)

                    if not subj_match and subj.pos_ == "PRON" and sent_entities:
                        subj_match = sent_entities[0]

                    add_triple(subj_match, rel, obj_match)

        utterances = get_utterances(scene_text)
        for i in range(len(utterances) - 1):
            a, b = utterances[i][0], utterances[i + 1][0]
            if a == b:
                continue
            match_a = entity_lookup.get(a.lower())
            match_b = entity_lookup.get(b.lower())
            if match_a and match_b:
                add_triple(match_a, "KNOWS", match_b)
                add_triple(match_b, "KNOWS", match_a)

    print(f"    Total auto relationships: {len(triples)}")
    return triples


def populate_play(mg, play_file: Path, rules: dict, nlp, cast_relationships: list):
    node_labels = set(rules["nodes"].keys())
    config = PLAY_CONFIGS.get(play_file.name, {})

    csv_path = Globals.OUTPUT_DIR / f"02_{play_file.stem}_finetuned_ner.csv"
    entity_groups = load_entities_csv(csv_path)

    play_title = play_file.stem.replace("Shakespeare_", "").replace("_", " ")
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            first_row = next(csv.DictReader(f), None)
            if first_row:
                play_title = first_row.get("play", play_title).strip()

    print(f"\n  Play: {play_title}")

    create_play_node(mg, play_title)

    node_count = 0
    for label in node_labels:
        for name in sorted(entity_groups.get(label, set())):
            create_node(mg, label, name)
            node_count += 1

    all_persons = entity_groups.get("PERSON", set())
    for person in sorted(all_persons):
        link_character_to_play(mg, person, play_title)
    print(f"  APPEARS_IN edges     : {len(all_persons)}")

    all_rel_records = []
    rel_count = 0

    known_families = config.get("known_families", set())
    for family in extract_family_nodes(entity_groups, known_families):
        create_node(mg, "FAMILY", family)
        node_count += 1
        for person in sorted(all_persons):
            if family in person.upper():
                rel_count += insert_edge(
                    mg, all_rel_records, play_title,
                    person, "PERSON", "MEMBER_OF", family, "FAMILY",
                    "family_nodes", verbose_prefix=""
                )

    play_cast_rels = [
        r for r in cast_relationships
        if r["play"].strip().upper() == play_title.upper()
    ]
    print(f"\n  Source 1 — Cast list relationships:")
    cast_edges = []
    person_names_upper = {n.upper() for n in entity_groups.get("PERSON", set())}
    for r in play_cast_rels:
        source = r["source"].strip().upper()
        target = r["target"].strip()
        rel = r["rel_type"].strip()
        target_label = "PERSON" if target.upper() in person_names_upper else "GPE"
        cast_edges.append((source, "PERSON", rel, target, target_label))
    rel_count += insert_edges(mg, all_rel_records, play_title, cast_edges, "cast_list")
    print(f"  Cast relationships   : {len(cast_edges)}")

    print(f"\n  Source 2 — play_configs relationships:")
    config_edges = config.get("relationships", [])
    rel_count += insert_edges(mg, all_rel_records, play_title, config_edges, "play_configs")
    print(f"  Config relationships : {len(config_edges)}")

    print(f"\n  Source 3 — Dependency parsing + KNOWS from dialogue:")
    root = DataExtractor.load_play(play_file)[0]
    scene_tuples = DataExtractor.extract_scenes(root)
    scenes = [text for _, _, _, text in scene_tuples]

    auto_edges = extract_relationships(scenes, entity_groups, nlp)
    knows_edges = extract_knows_from_dialogue(scenes, entity_groups)

    rel_count += insert_edges(
        mg, all_rel_records, play_title, auto_edges, "auto_extracted", verbose_prefix="[AUTO] "
    )
    rel_count += insert_edges(
        mg, all_rel_records, play_title, knows_edges, "knows_dialogue"
    )

    print(f"  Auto + KNOWS rels    : {len(auto_edges) + len(knows_edges)}")

    save_relationships_csv(
        all_rel_records,
        Globals.OUTPUT_DIR / f"04_{play_file.stem}_relationships.csv"
    )

    print(f"\n  Total relationships  : {rel_count}")
    return node_count, rel_count, all_rel_records


def run(mg, play_files: list, rules: dict, nlp):
    cast_rel_path = Globals.OUTPUT_DIR / "01_ALL_cast_relationships.csv"
    cast_relationships = load_cast_relationships(cast_rel_path)
    print(f"  Loaded {len(cast_relationships)} cast relationships from CSV")

    total_nodes = total_rels = 0
    all_records = []

    for play_file in play_files:
        print(f"\n{'=' * 60}")
        print(f"  Processing: {play_file.name}")
        print(f"{'=' * 60}")

        nodes, rels, records = populate_play(mg, play_file, rules, nlp, cast_relationships)
        total_nodes += nodes
        total_rels += rels
        all_records.extend(records)

    save_relationships_csv(all_records, Globals.OUTPUT_DIR / "04_ALL_relationships.csv")

    print(f"\n{'=' * 60}")
    print("  COMPLETE")
    print(f"  Total nodes : {total_nodes}")
    print(f"  Total rels  : {total_rels}")
    print("\n  Open Memgraph Lab at http://localhost:3000")