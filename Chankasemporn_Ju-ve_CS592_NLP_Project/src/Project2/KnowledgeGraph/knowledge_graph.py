"""
CS 374 – Assignment 2
Notebook 4: Building the Knowledge Graph with Memgraph + gqlalchemy
Reads kg_rules.json and the fine-tuned entity CSV to:
  1. Populate PERSON, GPE, LOCATION, TITLE nodes
  2. Insert hand-coded + auto-extracted relationships
  3. Demonstrates batch Cypher transactions via gqlalchemy
"""

import json
import csv
import spacy
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
import re

import src.Project2.Project2Globals as Globals
import src.Project2.Data_Extraction.data_extractor as DataExtractor
from src.Project2.Play_Configs.play_configs import PLAY_CONFIGS

try:
    from gqlalchemy import Memgraph
    MEMGRAPH_AVAILABLE = True
except ImportError:
    MEMGRAPH_AVAILABLE = False
    print("WARNING: gqlalchemy not installed. Run: pip install gqlalchemy")
    print("Running in DRY-RUN mode (queries will be printed, not executed).\n")

MEMGRAPH_HOST = "127.0.0.1"
MEMGRAPH_PORT = 7687

def clear_old_data(mg):
    mg.execute("MATCH (n) DETACH DELETE n")

# ─────────────────────────────────────────────
# 1. LOAD HELPERS
# ─────────────────────────────────────────────

def load_rules(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_entities_csv(path: Path) -> dict:
    """
    Load a fine-tuned entity CSV and group unique entity names by label.
    - PERSON names are uppercased for consistency (fixes Ursula vs URSULA)
    - GPE, LOCATION, TITLE keep their original casing
    - Filters out noise: short strings, punctuation-only, stage directions
    Returns dict: { label -> set(names) }
    """
    NOISE_WORDS = {
        "alarums", "alarum", "hautboys", "hautboy", "sennet", "flourish",
        "exeunt", "exit", "enter", "hence", "thence", "whence",
        "within", "without", "above", "below", "aside", "all",
        "first", "second", "third", "fourth", "fifth",
        "servant", "soldier", "messenger", "attendant",
        "officer", "captain", "porter", "doctor", "chorus",
        "prologue", "epilogue", "boy", "duff",
    }

    grouped = defaultdict(set)
    if not path.exists():
        print(f"  WARNING: CSV not found: {path}")
        return grouped

    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip()
            name  = row["entity_text"].strip()

            if not label or not name:
                continue
            # Filter out very short strings
            if len(name) < 3:
                continue
            # Filter out strings with no alphabetic characters
            if not any(c.isalpha() for c in name):
                continue
            # Filter out known noise words
            if name.lower() in NOISE_WORDS:
                continue
            # Filter out strings starting with punctuation or special chars
            if name[0] in ",-—.;:!?'\"()/\\":
                continue

            # Normalise PERSON names to uppercase to avoid duplicates
            if label == "PERSON":
                name = name.upper()

            grouped[label].add(name)

    return grouped


# ─────────────────────────────────────────────
# 2. CYPHER EXECUTION HELPER
# ─────────────────────────────────────────────

def execute(mg, query: str, params: dict = None):
    """Execute a Cypher query, or print it in dry-run mode."""
    params = params or {}
    if MEMGRAPH_AVAILABLE and mg:
        mg.execute(query, params)
    else:
        # Dry-run: substitute params for readable output
        display = query.strip()
        for k, v in params.items():
            display = display.replace(f"${k}", f'"{v}"')
        print(f"  CYPHER: {display}")


# ─────────────────────────────────────────────
# 3. NODE + RELATIONSHIP CREATION
# ─────────────────────────────────────────────

def create_node(mg, label: str, name: str):
    execute(mg,
            f"MERGE (n:{label} {{name: $name}})",
            {"name": name}
            )


def sanitise_rel_type(rel_type: str) -> str:
    """
    Cypher relationship types must be alphanumeric + underscores only.
    - Strip anything after a comma (e.g. "A_YOUNG_NOBLEMAN,_KINSMAN" → "A_YOUNG_NOBLEMAN")
    - Replace spaces and hyphens with underscores
    - Remove all other non-alphanumeric characters
    """
    rel_type = rel_type.split(",")[0]                      # drop after comma
    rel_type = re.sub(r'[\s\-]+', '_', rel_type)           # spaces/hyphens → _
    rel_type = re.sub(r'[^A-Z0-9_]', '', rel_type.upper()) # keep only valid chars
    rel_type = re.sub(r'_+', '_', rel_type).strip('_')     # collapse multiple _
    return rel_type


def create_relationship(mg, from_name: str, from_label: str,
                        rel_type: str, to_name: str, to_label: str):
    rel_type = sanitise_rel_type(rel_type)
    execute(mg, f"""
        MATCH (a:{from_label} {{name: $from_name}})
        MATCH (b:{to_label}   {{name: $to_name}})
        MERGE (a)-[:{rel_type}]->(b)
    """, {"from_name": from_name, "to_name": to_name})


def create_play_node(mg, play_title: str):
    """Create a PLAY node for each Shakespeare play."""
    execute(mg,
            "MERGE (p:PLAY {name: $name})",
            {"name": play_title}
            )


def link_character_to_play(mg, char_name: str, play_title: str):
    """Link a PERSON node to the PLAY it appears in."""
    execute(mg, """
        MATCH (c:PERSON {name: $char_name})
        MATCH (p:PLAY   {name: $play_title})
        MERGE (c)-[:APPEARS_IN]->(p)
    """, {"char_name": char_name, "play_title": play_title})


# ─────────────────────────────────────────────
# 4. AUTO RELATIONSHIP EXTRACTION
# ─────────────────────────────────────────────

def clean_xml(raw: str) -> str:
    return (raw
        .replace("&#8217;", "'").replace("&#8216;", "'")
        .replace("&#8220;", '"').replace("&#8221;", '"')
        .replace("&#8212;", "—").replace("&#8211;", "–"))

def load_play_scenes(play_file: Path) -> list:
    """Parse the play XML and return all scene texts."""
    raw  = play_file.read_text(encoding="utf-8")
    root = ET.fromstring(clean_xml(raw))
    scenes = []
    for act in root.findall(".//Act"):
        scene_elems = act.findall("Scene")
        if scene_elems:
            for scene in scene_elems:
                text = re.sub(r'\s+', ' ', " ".join(scene.itertext()).strip())
                if text:
                    scenes.append(text)
        else:
            text = re.sub(r'\s+', ' ', " ".join(act.itertext()).strip())
            if text:
                scenes.append(text)
    return scenes

def extract_from_finetuned_rels(play_file, entity_groups: dict) -> list:
    """
    Read REL-labeled entities from the fine-tuned NER CSV and match them
    against PERSON entities in the same scene to infer new relationships.
    """
    from collections import defaultdict as _dd
    csv_path = Globals.OUTPUT_DIR / f"02_{play_file.stem}_finetuned_ner.csv"
    if not csv_path.exists():
        return []

    scene_entities = _dd(lambda: {"REL": [], "PERSON": []})
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip()
            name  = row["entity_text"].strip()
            key   = (row["act"], row["scene"])
            if label in ("REL", "PERSON") and name:
                scene_entities[key][label].append(name)

    REL_NOISE = {
        "SERVANTS", "ATTENDANTS", "LORDS", "OFFICERS",
        "SOLDIERS", "MURDERERS", "ALL", "OTHERS"
    }
    REL_WORD_MAP = {
        "SON": "SON_OF", "DAUGHTER": "DAUGHTER_OF",
        "BROTHER": "BROTHER_OF", "SISTER": "SISTER_OF",
        "NEPHEW": "NEPHEW_OF", "NIECE": "NIECE_OF",
        "FATHER": "FATHER_OF", "MOTHER": "MOTHER_OF",
        "WIFE": "WIFE_OF", "HUSBAND": "HUSBAND_OF",
        "SERVANT": "SERVANT_OF", "FOLLOWER": "FOLLOWER_OF",
        "KINSMAN": "KINSMAN_OF", "FRIEND": "FRIEND_OF",
        "ATTENDANT": "ATTENDS", "PAGE": "PAGE_OF",
    }

    triples      = []
    seen         = set()
    known_persons = {n.upper() for n in entity_groups.get("PERSON", set())}

    for (act, scene), ents in scene_entities.items():
        rel_entities    = ents["REL"]
        person_entities = [p for p in ents["PERSON"] if p.upper() in known_persons]

        for rel_word in rel_entities:
            if rel_word.upper() in REL_NOISE:
                continue
            rel_type = REL_WORD_MAP.get(rel_word.upper())
            if not rel_type or len(person_entities) < 2:
                continue
            subj = person_entities[0].upper()
            obj  = person_entities[1].upper()
            key  = (subj, rel_type, obj)
            if key not in seen:
                seen.add(key)
                triples.append((subj, "PERSON", rel_type, obj, "PERSON"))

    print(f"    REL entities from fine-tuned CSV: {len(triples)} relationships")
    return triples


def extract_relationships(scenes: list, entity_groups: dict, nlp) -> list:
    """
    Relationship extraction using dependency parsing only.
    Co-occurrence window removed — produced too many false positives.
    KILLS, LOVES, MARRIED_TO removed from VERB_MAP — handled by play_configs.
    """
    entity_lookup = {}
    for label, names in entity_groups.items():
        for name in names:
            entity_lookup[name.lower()] = (name, label)

    NOISE_WORDS = {
        "alarums", "alarum", "hautboys", "hautboy", "sennet", "flourish",
        "exeunt", "exit", "enter", "hence", "thence", "whence",
        "within", "without", "above", "below", "aside",
        "first", "second", "third", "fourth", "all",
        "boy", "duff", "servant", "soldier", "messenger",
        "attendant", "officer", "captain", "porter", "doctor",
    }
    entity_lookup = {
        k: v for k, v in entity_lookup.items()
        if k not in NOISE_WORDS and len(k) > 2
    }

    VERB_MAP = {
        "serve":     "SERVES",   "follow":    "FOLLOWS",
        "obey":      "OBEYS",    "attend":    "SERVES",
        "swear":     "LOYAL_TO", "pledge":    "LOYAL_TO",
        "betray":    "BETRAYS",  "deceive":   "DECEIVES",
        "trick":     "DECEIVES", "conspire":  "CONSPIRES_WITH",
        "command":   "COMMANDS", "banish":    "BANISHES",
        "fight":     "FIGHTS",   "oppose":    "FIGHTS",
        "challenge": "FIGHTS",   "defeat":    "DEFEATS",
        "flee":      "FLEES_FROM",
        "know":      "KNOWS",    "meet":      "MEETS",
        "trust":     "TRUSTS",   "fear":      "FEARS",
        "hate":      "HATES",    "help":      "HELPS",
        "seek":      "SEEKS",    "warn":      "WARNS",
        "curse":     "CURSES",   "send":      "SENDS",
        "accuse":    "ACCUSES",  "suspect":   "SUSPECTS",
        "protect":   "PROTECTS", "forgive":   "FORGIVES",
        "greet":     "MEETS",    "visit":     "MEETS",
    }

    RELATIONSHIP_CONSTRAINTS = {
        rel: ("PERSON", "PERSON") for rel in VERB_MAP.values()
    }

    triples = []
    seen    = set()

    def add_triple(subj_match, rel_type, obj_match):
        if not subj_match or not obj_match:
            return
        if subj_match[0] == obj_match[0]:
            return
        constraint = RELATIONSHIP_CONSTRAINTS.get(rel_type)
        if constraint:
            if subj_match[1] != constraint[0] or obj_match[1] != constraint[1]:
                return
        key = (subj_match[0], rel_type, obj_match[0])
        if key not in seen:
            seen.add(key)
            triples.append((subj_match[0], subj_match[1],
                            rel_type, obj_match[0], obj_match[1]))

    def resolve_entity(token):
        chunk = " ".join(
            t.text for t in token.subtree
            if t.dep_ in ("compound", "nn") or t == token
        ).strip()
        for candidate in [chunk, token.text]:
            match = entity_lookup.get(candidate.lower())
            if match:
                return match
        for t in token.subtree:
            match = entity_lookup.get(t.text.lower())
            if match:
                return match
        return None

    SPEAKER_RE = re.compile(r'([A-Z][A-Z\s]{2,})\.\s+')

    def get_utterances(scene_text):
        parts = SPEAKER_RE.split(scene_text)
        utterances = []
        i = 1
        while i < len(parts) - 1:
            speaker = parts[i].strip()
            text    = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if speaker and text:
                utterances.append((speaker, text))
            i += 2
        return utterances

    for scene_text in scenes:
        for speaker, utt_text in get_utterances(scene_text):
            doc = nlp(utt_text)
            for sent in doc.sents:
                sent_entities = []
                for ent in sent.ents:
                    m = entity_lookup.get(ent.text.lower())
                    if m:
                        sent_entities.append(m)

                for token in sent:
                    if token.pos_ != "VERB":
                        continue
                    rel_type = VERB_MAP.get(token.lemma_.lower())
                    if not rel_type:
                        continue

                    subj = None
                    obj  = None
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass") and not subj:
                            subj = child
                        if child.dep_ in ("dobj", "pobj", "attr", "oprd") and not obj:
                            obj = child

                    if not subj or not obj:
                        continue

                    subj_match = resolve_entity(subj)
                    obj_match  = resolve_entity(obj)

                    if not subj_match and subj.pos_ == "PRON" and sent_entities:
                        subj_match = sent_entities[0]

                    add_triple(subj_match, rel_type, obj_match)

    print(f"    Dependency parsing: {len(triples)} relationships extracted")
    return triples


# ─────────────────────────────────────────────
# SAVE RELATIONSHIPS CSV
# ─────────────────────────────────────────────

REL_FIELDS = [
    "play", "from_node", "from_label",
    "rel_type", "to_node", "to_label", "extraction_source"
]

def save_relationships_csv(records: list, path: Path):
    """Save relationship records to CSV for debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        rows.append({
            "play":              r.get("play", ""),
            "from_node":         r.get("source", ""),
            "from_label":        r.get("source_label", ""),
            "rel_type":          r.get("rel_type", ""),
            "to_node":           r.get("target", ""),
            "to_label":          r.get("target_label", ""),
            "extraction_source": r.get("extraction_source", ""),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Saved {len(rows):,} relationships : {path.name}")

# ─────────────────────────────────────────────
# LOAD CAST RELATIONSHIPS CSV
# ─────────────────────────────────────────────

def load_cast_relationships(path: Path) -> list:
    """
    Load 01_ALL_cast_relationships.csv.
    Returns list of dicts: {play, source, rel_type, target, description}
    """
    if not path.exists():
        print(f"  WARNING: Cast relationships CSV not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ─────────────────────────────────────────────
# 5. PER-PLAY POPULATION
# ─────────────────────────────────────────────

def populate_play(mg, play_file: Path, rules: dict, nlp,
                  cast_relationships: list):
    """
    Build the knowledge graph for one play using three relationship sources
    in priority order:

    Source 1 — Cast relationships CSV (01_ALL_cast_relationships.csv)
      Automatically extracted from the cast list descriptions.
      e.g. FLEANCE -[SON_TO]-> Banquo
      Most accurate — comes directly from Shakespeare's own cast list.

    Source 2 — play_configs.py relationships
      Hand-curated edges covering kills, marriages, loyalty, titles etc.
      e.g. MACBETH -[KILLS]-> DUNCAN
      High accuracy — manually verified against the play.

    Source 3 — Auto-extracted from dialogue text
      Dependency parsing + co-occurrence window on scene text.
      Broadest coverage but noisiest — used as a fallback to catch
      relationships not covered by sources 1 and 2.
    """
    node_labels = set(rules["nodes"].keys())

    # Load entities from fine-tuned CSV
    csv_path      = Globals.OUTPUT_DIR / f"02_{play_file.stem}_finetuned_ner.csv"
    entity_groups = load_entities_csv(csv_path)

    # Get play title
    play_title = play_file.stem.replace("Shakespeare_", "").replace("_", " ")
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                play_title = row.get("play", play_title).strip()
                break

    print(f"\n  Play: {play_title}")

    # ── Create PLAY node ──────────────────────────────────────────
    create_play_node(mg, play_title)

    # ── Create entity nodes from fine-tuned CSV ───────────────────
    node_count = 0
    for label in node_labels:
        for name in sorted(entity_groups.get(label, set())):
            create_node(mg, label, name)
            node_count += 1
    print(f"  Nodes created        : {node_count}")

    # ── Link PERSON nodes → PLAY via APPEARS_IN ───────────────────
    all_persons = entity_groups.get("PERSON", set())
    for person in sorted(all_persons):
        link_character_to_play(mg, person, play_title)
    print(f"  APPEARS_IN edges     : {len(all_persons)}")

    rel_count = 0
    all_rel_records = []  # collect all relationships for CSV export

    # ── SOURCE 1: Cast relationships CSV ─────────────────────────
    play_cast_rels = [
        r for r in cast_relationships
        if r["play"].strip().upper() == play_title.upper()
    ]
    print(f"\n  Source 1 — Cast list relationships:")
    for r in play_cast_rels:
        source   = r["source"].strip().upper()
        target   = r["target"].strip()
        rel_type = r["rel_type"].strip()

        target_upper = target.upper()
        target_label = "PERSON" if target_upper in {
            n.upper() for n in entity_groups.get("PERSON", set())
        } else "GPE"

        create_node(mg, "PERSON", source)
        create_node(mg, target_label, target)
        create_relationship(mg, source, "PERSON", rel_type, target, target_label)
        rel_count += 1
        all_rel_records.append({
            "play":              play_title,
            "source":            source,
            "source_label":      "PERSON",
            "rel_type":          rel_type,
            "target":            target,
            "target_label":      target_label,
            "extraction_source": "cast_list",
        })
        print(f"    {source:20s} -[{rel_type}]-> {target}")

    print(f"  Cast relationships   : {len(play_cast_rels)}")

    # ── SOURCE 2: play_configs.py hand-curated relationships ──────
    config = PLAY_CONFIGS.get(play_file.name, {})
    config_rels = config.get("relationships", [])
    print(f"\n  Source 2 — play_configs relationships:")
    for (fn, fl, rel, tn, tl) in config_rels:
        create_node(mg, fl, fn)
        create_node(mg, tl, tn)
        create_relationship(mg, fn, fl, rel, tn, tl)
        rel_count += 1
        all_rel_records.append({
            "play":              play_title,
            "source":            fn,
            "source_label":      fl,
            "rel_type":          rel,
            "target":            tn,
            "target_label":      tl,
            "extraction_source": "play_configs",
        })
        print(f"    {fn:20s} -[{rel}]-> {tn}")
    print(f"  Config relationships : {len(config_rels)}")

    # ── SOURCE 3: Dependency parsing ─────────────────────────────
    print(f"\n  Source 3 — Dependency parsing:")
    root         = DataExtractor.load_play(play_file)[0]
    scene_tuples = DataExtractor.extract_scenes(root)
    scenes       = [text for _, _, _, text in scene_tuples]
    auto_rels    = extract_relationships(scenes, entity_groups, nlp)

    auto_inserted = 0
    for (fn, fl, rel, tn, tl) in auto_rels:
        create_node(mg, fl, fn)
        create_node(mg, tl, tn)
        create_relationship(mg, fn, fl, rel, tn, tl)
        rel_count += 1
        auto_inserted += 1
        print(f"    [AUTO] {fn:20s} -[{rel}]-> {tn}")
        all_rel_records.append({
            "play":              play_title,
            "source":            fn,
            "source_label":      fl,
            "rel_type":          rel,
            "target":            tn,
            "target_label":      tl,
            "extraction_source": "auto_extracted",
        })
    print(f"  Auto-extracted rels  : {auto_inserted} inserted")

    # ── Save per-play relationships CSV ───────────────────────────
    save_relationships_csv(
        all_rel_records,
        Globals.OUTPUT_DIR / f"04_{play_file.stem}_relationships.csv"
    )

    print(f"\n  Total relationships  : {rel_count}")
    return node_count, rel_count, all_rel_records


# ─────────────────────────────────────────────
# 6. MEMGRAPH CONNECTION
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# RUN  (called from Project2Runner)
# ─────────────────────────────────────────────

def run(mg, play_files: list, rules: dict, nlp):
    """
    Full knowledge graph population pipeline.

    Loads cast relationships CSV once, then processes each play using
    all three relationship sources in priority order.
    """
    cast_rel_path      = Globals.OUTPUT_DIR / "01_ALL_cast_relationships.csv"
    cast_relationships = load_cast_relationships(cast_rel_path)
    print(f"  Loaded {len(cast_relationships)} cast relationships from CSV")

    total_nodes    = 0
    total_rels     = 0
    all_records    = []

    for play_file in play_files:
        print(f"\n{'='*60}")
        print(f"  Processing: {play_file.name}")
        print(f"{'='*60}")

        nodes, rels, records = populate_play(
            mg, play_file, rules, nlp, cast_relationships
        )
        total_nodes += nodes
        total_rels  += rels
        all_records.extend(records)

    # Save combined CSV across all plays
    save_relationships_csv(
        all_records,
        Globals.OUTPUT_DIR / "04_ALL_relationships.csv"
    )

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"  Total nodes : {total_nodes}")
    print(f"  Total rels  : {total_rels}")
    print(f"\n  Open Memgraph Lab at http://localhost:3000")