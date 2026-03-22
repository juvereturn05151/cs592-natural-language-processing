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
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
import re
from src.Project2.Data_Extraction.data_extractor import find_repo_root

try:
    from gqlalchemy import Memgraph
    MEMGRAPH_AVAILABLE = True
except ImportError:
    MEMGRAPH_AVAILABLE = False
    print("WARNING: gqlalchemy not installed. Run: pip install gqlalchemy")
    print("Running in DRY-RUN mode (queries will be printed, not executed).\n")

# ─────────────────────────────────────────────
# PATHS + CONFIG
# ─────────────────────────────────────────────

REPO_ROOT, _ = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = DATA_DIR / "output"
RULES_PATH = DATA_DIR / "kg_rules.json"

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


def create_relationship(mg, from_name: str, from_label: str,
                        rel_type: str, to_name: str, to_label: str):
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

def extract_relationships(scenes: list, entity_groups: dict, nlp) -> list:
    """
    Two-method relationship extraction designed for Shakespeare's archaic language.

    Method 1 — Dependency parsing (precise but misses a lot in archaic text)
      Finds subject→verb→object triples using spaCy's dependency tree.

    Method 2 — Co-occurrence window (broader, catches what the parser misses)
      If two named entities appear in the same utterance AND a keyword verb
      appears anywhere between them in the raw text, record the relationship.
      This works well for Shakespeare because it doesn't rely on correct parsing.

    Results from both methods are merged and deduplicated.
    """

    # Build lookup: entity name (lowercase) → (original_name, label)
    entity_lookup = {}
    for label, names in entity_groups.items():
        for name in names:
            entity_lookup[name.lower()] = (name, label)

    # ── Filter out noise: stage directions and non-character words ──
    # These get picked up by NER but are not real characters
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

    # Sorted longest-first so "LADY MACBETH" matches before "MACBETH"
    sorted_entities = sorted(entity_lookup.keys(), key=len, reverse=True)

    VERB_MAP = {
        # Killing — split into high-confidence and ambiguous
        "kill":      "KILLS",   "slay":      "KILLS",
        "murder":    "KILLS",   "slaughter": "KILLS",
        "stab":      "KILLS",   "poison":    "KILLS",
        "hang":      "KILLS",   "execute":   "KILLS",
        # Ambiguous — only kept for dependency parsing (not co-occurrence)
        "smite":     "KILLS",   "strike":    "KILLS",
        "die":       "KILLS",   "fall":      "KILLS",
        # Love / marriage
        "love":      "LOVES",   "adore":     "LOVES",
        "fancy":     "LOVES",   "woo":       "LOVES",
        "marry":     "MARRIED_TO", "wed":    "MARRIED_TO",
        "betroth":   "MARRIED_TO",
        # Loyalty / service
        "serve":     "SERVES",  "follow":    "FOLLOWS",
        "obey":      "OBEYS",   "attend":    "SERVES",
        "swear":     "LOYAL_TO","pledge":    "LOYAL_TO",
        # Betrayal / deception
        "betray":    "BETRAYS", "deceive":   "DECEIVES",
        "trick":     "DECEIVES","lie":       "DECEIVES",
        "conspire":  "CONSPIRES_WITH",
        # Power
        "rule":      "RULES",   "command":   "COMMANDS",
        "govern":    "RULES",   "crown":     "RULES",
        "banish":    "BANISHES","exile":     "BANISHES",
        # Conflict
        "fight":     "FIGHTS",  "oppose":    "FIGHTS",
        "challenge": "FIGHTS",  "defeat":    "DEFEATS",
        "flee":      "FLEES_FROM",
        # Family
        "bear":      "PARENT_OF", "beget":   "PARENT_OF",
        # Social
        "know":      "KNOWS",   "meet":      "MEETS",
        "trust":     "TRUSTS",  "fear":      "FEARS",
        "hate":      "HATES",   "help":      "HELPS",
        "seek":      "SEEKS",   "warn":      "WARNS",
        "curse":     "CURSES",  "send":      "SENDS",
        "call":      "CALLS",   "speak":     "SPEAKS_TO",
        "tell":      "SPEAKS_TO","bid":      "COMMANDS",
        "greet":     "MEETS",   "thank":     "SPEAKS_TO",
        "accuse":    "ACCUSES", "forgive":   "FORGIVES",
        "protect":   "PROTECTS","rescue":    "PROTECTS",
        "visit":     "MEETS",   "embrace":   "LOVES",
        "suspect":   "SUSPECTS","question":  "SPEAKS_TO",
    }

    # Also build a raw keyword list for co-occurrence
    # Maps any word form → relationship type
    # NOTE: ambiguous verbs (fall, die, strike, smite) are intentionally
    # excluded here — they cause too many false positives in co-occurrence
    COOCCURRENCE_BLACKLIST = {"fall", "die", "strike", "smite"}
    KEYWORD_MAP = {}
    for verb, rel in VERB_MAP.items():
        if verb not in COOCCURRENCE_BLACKLIST:
            KEYWORD_MAP[verb] = rel
    # Add archaic/inflected forms not caught by lemmatiser
    KEYWORD_MAP.update({
        "slain":     "KILLS",   "slew":       "KILLS",
        "killed":    "KILLS",   "murdered":   "KILLS",
        "stabbed":   "KILLS",   "loves":      "LOVES",
        "loved":     "LOVES",   "married":    "MARRIED_TO",
        "serves":    "SERVES",  "served":     "SERVES",
        "betrayed":  "BETRAYS", "betrays":    "BETRAYS",
        "deceived":  "DECEIVES","rules":      "RULES",
        "ruled":     "RULES",   "fights":     "FIGHTS",
        "fought":    "FIGHTS",  "fears":      "FEARS",
        "feared":    "FEARS",   "hates":      "HATES",
        "hated":     "HATES",   "trusts":     "TRUSTS",
        "trusted":   "TRUSTS",  "banished":   "BANISHES",
        "fled":      "FLEES_FROM","knows":    "KNOWS",
        "knew":      "KNOWS",   "met":        "MEETS",
        "sent":      "SENDS",   "warned":     "WARNS",
        "cursed":    "CURSES",  "accused":    "ACCUSES",
        "suspects":  "SUSPECTS","protected":  "PROTECTS",
        "crowned":   "RULES",   "woos":       "LOVES",
        "wooed":     "LOVES",   "conspired":  "CONSPIRES_WITH",
        "slaughtered": "KILLS", "poisoned":   "KILLS",
    })

    seen    = set()
    triples = []

    # Rules: certain relationship types only make sense between specific label pairs
    RELATIONSHIP_CONSTRAINTS = {
        "KILLS":        ("PERSON", "PERSON"),
        "LOVES":        ("PERSON", "PERSON"),
        "MARRIED_TO":   ("PERSON", "PERSON"),
        "BETRAYS":      ("PERSON", "PERSON"),
        "SERVES":       ("PERSON", "PERSON"),
        "FOLLOWS":      ("PERSON", "PERSON"),
        "OBEYS":        ("PERSON", "PERSON"),
        "LOYAL_TO":     ("PERSON", "PERSON"),
        "FEARS":        ("PERSON", "PERSON"),
        "HATES":        ("PERSON", "PERSON"),
        "TRUSTS":       ("PERSON", "PERSON"),
        "FIGHTS":       ("PERSON", "PERSON"),
        "DEFEATS":      ("PERSON", "PERSON"),
        "CONSPIRES_WITH":("PERSON","PERSON"),
        "DECEIVES":     ("PERSON", "PERSON"),
        "ACCUSES":      ("PERSON", "PERSON"),
        "SUSPECTS":     ("PERSON", "PERSON"),
        "COMMANDS":     ("PERSON", "PERSON"),
        "HELPS":        ("PERSON", "PERSON"),
        "WARNS":        ("PERSON", "PERSON"),
        "PROTECTS":     ("PERSON", "PERSON"),
        "MEETS":        ("PERSON", "PERSON"),
        "KNOWS":        ("PERSON", "PERSON"),
        "SENDS":        ("PERSON", "PERSON"),
        "CALLS":        ("PERSON", "PERSON"),
        "SPEAKS_TO":    ("PERSON", "PERSON"),
        "FORGIVES":     ("PERSON", "PERSON"),
        "CURSES":       ("PERSON", "PERSON"),
        "PARENT_OF":    ("PERSON", "PERSON"),
        "RULES":        ("PERSON", "GPE"),
        "BANISHES":     ("PERSON", "PERSON"),
        "FLEES_FROM":   ("PERSON", "PERSON"),
        "SEEKS":        ("PERSON", "PERSON"),
    }

    def add_triple(subj_match, rel_type, obj_match):
        if not subj_match or not obj_match:
            return
        if subj_match[0] == obj_match[0]:
            return
        # Enforce label constraints — e.g. KILLS must be PERSON → PERSON
        constraint = RELATIONSHIP_CONSTRAINTS.get(rel_type)
        if constraint:
            if subj_match[1] != constraint[0] or obj_match[1] != constraint[1]:
                return
        key = (subj_match[0], rel_type, obj_match[0])
        if key not in seen:
            seen.add(key)
            triples.append((
                subj_match[0], subj_match[1],
                rel_type,
                obj_match[0],  obj_match[1],
            ))

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

    # ── Split scenes into individual utterances ──────────────────
    SPEAKER_RE = re.compile(r'([A-Z][A-Z\s]{2,})\.\s+')

    def get_utterances(scene_text):
        parts      = SPEAKER_RE.split(scene_text)
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
        utterances = get_utterances(scene_text)

        for speaker, utt_text in utterances:
            doc = nlp(utt_text)

            # ── METHOD 1: Dependency parsing ─────────────────────
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

            # ── METHOD 2: Co-occurrence window ───────────────────
            # Find all entity mentions in this utterance (in order)
            utt_lower = utt_text.lower()
            found_entities = []
            for ent_name in sorted_entities:
                start = 0
                while True:
                    idx = utt_lower.find(ent_name, start)
                    if idx == -1:
                        break
                    found_entities.append((idx, entity_lookup[ent_name]))
                    start = idx + len(ent_name)

            # Sort by position in text
            found_entities.sort(key=lambda x: x[0])

            # Also add the speaker as the first entity if known
            speaker_match = entity_lookup.get(speaker.lower())
            if speaker_match:
                found_entities.insert(0, (-1, speaker_match))

            # For each pair of entities within 150 chars of each other,
            # check if a keyword verb appears between them
            for i in range(len(found_entities)):
                for j in range(i + 1, len(found_entities)):
                    pos_a, ent_a = found_entities[i]
                    pos_b, ent_b = found_entities[j]

                    # Only look within a 60-character window
                    if pos_a >= 0 and pos_b - pos_a > 60:
                        break

                    # Get the text between the two entities
                    if pos_a < 0:
                        between = utt_lower[:pos_b]
                    else:
                        between = utt_lower[pos_a:pos_b]

                    # Check if any keyword appears in the between-text
                    for keyword, rel_type in KEYWORD_MAP.items():
                        if keyword in between.split():
                            add_triple(ent_a, rel_type, ent_b)
                            break   # one relationship per pair per utterance

    print(f"    Extracted {len(triples)} relationships "
          f"({len([t for t in triples if t[2] in ['KILLS','LOVES','MARRIED_TO']])} "
          f"high-confidence)")
    return triples

# ─────────────────────────────────────────────
# 5. PER-PLAY POPULATION
# ─────────────────────────────────────────────

def populate_play(mg, play_file: Path, rules: dict, nlp):
    """
    For one play:
      1. Create a PLAY node
      2. Create all entity nodes from the fine-tuned CSV
      3. Link all PERSON nodes to the PLAY via APPEARS_IN
      4. Auto-extract relationships via dependency parsing
    """
    node_labels = set(rules["nodes"].keys())

    # Load entities from fine-tuned CSV
    csv_path      = OUTPUT_DIR / f"02_{play_file.stem}_finetuned_ner.csv"
    entity_groups = load_entities_csv(csv_path)

    # Get play title
    play_title = play_file.stem.replace("Shakespeare_", "").replace("_", " ")
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                play_title = row.get("play", play_title).strip()
                break

    print(f"\n  Play: {play_title}")

    # Create PLAY node
    create_play_node(mg, play_title)

    # Create entity nodes
    node_count = 0
    for label in node_labels:
        for name in sorted(entity_groups.get(label, set())):
            create_node(mg, label, name)
            node_count += 1
    print(f"  Nodes created/merged : {node_count}")

    # Link PERSON nodes → PLAY
    all_persons = entity_groups.get("PERSON", set())
    for person in sorted(all_persons):
        link_character_to_play(mg, person, play_title)
    print(f"  APPEARS_IN edges     : {len(all_persons)}")

    # Auto-extract and insert relationships
    scenes        = load_play_scenes(play_file)
    relationships = extract_relationships(scenes, entity_groups, nlp)

    for (fn, fl, rel, tn, tl) in relationships:
        create_node(mg, fl, fn)
        create_node(mg, tl, tn)
        create_relationship(mg, fn, fl, rel, tn, tl)
    print(f"  Auto-extracted rels  : {len(relationships)}")

    return node_count, len(relationships)


# ─────────────────────────────────────────────
# 5. MEMGRAPH CONNECTION
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
# MAIN — loops over all Shakespeare plays
# ─────────────────────────────────────────────