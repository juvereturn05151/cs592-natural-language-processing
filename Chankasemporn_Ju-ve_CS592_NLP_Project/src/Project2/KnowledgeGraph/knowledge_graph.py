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
from src.Project2.NER_Extraction.data_extractor import find_repo_root

# gqlalchemy connects to Memgraph via the Bolt protocol
try:
    from gqlalchemy import Memgraph
    MEMGRAPH_AVAILABLE = True
except ImportError:
    MEMGRAPH_AVAILABLE = False
    print("WARNING: gqlalchemy not installed. Run: pip install gqlalchemy")
    print("The script will still build query strings and print them.\n")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

REPO_ROOT, _  = find_repo_root()
DATA_DIR      = REPO_ROOT / "data"
RULES_PATH    = DATA_DIR / "kg_rules.json"
ENTITIES_CSV  = DATA_DIR / "output" / "02_finetuned_ner_entities.csv"

MEMGRAPH_HOST = "127.0.0.1"
MEMGRAPH_PORT = 7687

# ─────────────────────────────────────────────
# 1. LOAD RULES + ENTITIES
# ─────────────────────────────────────────────

def load_rules(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_entities_csv(path):
    """Load fine-tuned entities and group by label."""
    grouped = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            name  = row["entity_text"].strip()
            if name:
                grouped[label].add(name)
    return grouped

# ─────────────────────────────────────────────
# 2. HAND-CODED RELATIONSHIPS
#    (extracted by reading the play)
# ─────────────────────────────────────────────

# Format: (from_name, from_label, relationship, to_name, to_label)
RELATIONSHIPS = [
    # ── Family / Marriage ──────────────────────
    ("MACBETH",      "PERSON", "MARRIED_TO",     "LADY MACBETH",  "PERSON"),
    ("MACDUFF",      "PERSON", "MARRIED_TO",     "LADY MACDUFF",  "PERSON"),
    ("FLEANCE",      "PERSON", "SON_OF",         "BANQUO",        "PERSON"),
    ("MALCOLM",      "PERSON", "SON_OF",         "DUNCAN",        "PERSON"),
    ("DONALBAIN",    "PERSON", "SON_OF",         "DUNCAN",        "PERSON"),
    ("BOY",          "PERSON", "SON_OF",         "MACDUFF",       "PERSON"),
    ("YOUNG SIWARD", "PERSON", "SON_OF",         "SIWARD",        "PERSON"),

    # ── Kills ──────────────────────────────────
    ("MACBETH",      "PERSON", "KILLS",          "DUNCAN",        "PERSON"),
    ("MACBETH",      "PERSON", "KILLS",          "BANQUO",        "PERSON"),  # orders it
    ("MACBETH",      "PERSON", "KILLS",          "YOUNG SIWARD",  "PERSON"),
    ("MACDUFF",      "PERSON", "KILLS",          "MACBETH",       "PERSON"),
    ("MACBETH",      "PERSON", "KILLS_OFFSTAGE", "LADY MACDUFF",  "PERSON"),
    ("MACBETH",      "PERSON", "KILLS_OFFSTAGE", "BOY",           "PERSON"),
    ("LADY MACBETH", "PERSON", "KILLS_OFFSTAGE", "DUNCAN",        "PERSON"),  # instigates

    # ── Loyalty / Betrayal ─────────────────────
    ("MACBETH",      "PERSON", "LOYAL_TO",       "DUNCAN",        "PERSON"),
    ("MACBETH",      "PERSON", "BETRAYS",        "DUNCAN",        "PERSON"),
    ("BANQUO",       "PERSON", "LOYAL_TO",       "DUNCAN",        "PERSON"),
    ("MACDUFF",      "PERSON", "LOYAL_TO",       "MALCOLM",       "PERSON"),
    ("ROSS",         "PERSON", "LOYAL_TO",       "DUNCAN",        "PERSON"),
    ("LENNOX",       "PERSON", "LOYAL_TO",       "DUNCAN",        "PERSON"),

    # ── Rules ──────────────────────────────────
    ("DUNCAN",       "PERSON", "RULES",          "Scotland",      "GPE"),
    ("MACBETH",      "PERSON", "RULES",          "Scotland",      "GPE"),
    ("MALCOLM",      "PERSON", "RULES",          "Scotland",      "GPE"),
    ("SIWARD",       "PERSON", "RULES",          "Northumberland","GPE"),

    # ── Titles ─────────────────────────────────
    ("MACBETH",      "PERSON", "HOLDS_TITLE",    "Thane of Glamis",   "TITLE"),
    ("MACBETH",      "PERSON", "HOLDS_TITLE",    "Thane of Cawdor",   "TITLE"),
    ("MACBETH",      "PERSON", "HOLDS_TITLE",    "King of Scotland",  "TITLE"),
    ("DUNCAN",       "PERSON", "HOLDS_TITLE",    "King of Scotland",  "TITLE"),
    ("MACDUFF",      "PERSON", "HOLDS_TITLE",    "Thane of Fife",     "TITLE"),
    ("SIWARD",       "PERSON", "HOLDS_TITLE",    "Earl of Northumberland", "TITLE"),

    # ── Location containment ───────────────────
    ("Inverness",    "GPE",    "LOCATED_IN",     "Scotland",      "GPE"),
    ("Forres",       "GPE",    "LOCATED_IN",     "Scotland",      "GPE"),
    ("Fife",         "GPE",    "LOCATED_IN",     "Scotland",      "GPE"),
    ("Dunsinane",    "GPE",    "LOCATED_IN",     "Scotland",      "GPE"),
    ("Birnam Wood",  "GPE",    "LOCATED_IN",     "Scotland",      "GPE"),
]

# ─────────────────────────────────────────────
# 3. CYPHER HELPERS
# ─────────────────────────────────────────────

def execute(mg, query, params=None):
    """Run a Cypher query, print it if Memgraph is unavailable."""
    if MEMGRAPH_AVAILABLE and mg:
        mg.execute(query, params or {})
    else:
        display = query
        if params:
            for k, v in params.items():
                display = display.replace(f"${k}", f'"{v}"')
        print(f"  CYPHER: {display.strip()}")

def create_node(mg, label, name):
    query = f"MERGE (n:{label} {{name: $name}})"
    execute(mg, query, {"name": name})

def create_relationship(mg, from_name, from_label, rel_type, to_name, to_label):
    query = f"""
    MATCH (a:{from_label} {{name: $from_name}})
    MATCH (b:{to_label}   {{name: $to_name}})
    MERGE (a)-[:{rel_type}]->(b)
    """
    execute(mg, query, {"from_name": from_name, "to_name": to_name})

# ─────────────────────────────────────────────
# 4. POPULATE THE GRAPH
# ─────────────────────────────────────────────

def populate_graph(mg, rules, entity_groups):
    """
    Step 1: Create all nodes from extracted entities.
    Step 2: Create all relationships.
    """
    node_labels = set(rules["nodes"].keys())

    print("\n── Creating nodes ──")
    node_count = 0
    for label in node_labels:
        names = entity_groups.get(label, set())
        for name in sorted(names):
            create_node(mg, label, name)
            node_count += 1
        print(f"  [{label}] {len(names)} nodes created")

    # Also create nodes that appear in relationships but may not be in CSV
    for (fn, fl, rel, tn, tl) in RELATIONSHIPS:
        create_node(mg, fl, fn)
        create_node(mg, tl, tn)

    print(f"\n  Total nodes: {node_count}")

    print("\n── Creating relationships ──")
    for i, (fn, fl, rel, tn, tl) in enumerate(RELATIONSHIPS):
        create_relationship(mg, fn, fl, rel, tn, tl)
        if (i + 1) % 10 == 0 or i + 1 == len(RELATIONSHIPS):
            print(f"  {i+1}/{len(RELATIONSHIPS)} relationships inserted")

    print(f"\n  Total relationships: {len(RELATIONSHIPS)}")

# ─────────────────────────────────────────────
# 5. CONNECT AND RUN
# ─────────────────────────────────────────────

def connect_memgraph():
    if not MEMGRAPH_AVAILABLE:
        return None
    try:
        mg = Memgraph(host=MEMGRAPH_HOST, port=MEMGRAPH_PORT)
        mg.execute("RETURN 1")  # ping
        print("Connected to Memgraph successfully.")
        return mg
    except Exception as e:
        print(f"Could not connect to Memgraph: {e}")
        print("Running in DRY RUN mode (printing queries only).\n")
        return None

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Loading rules and entities ===")
    rules = load_rules(RULES_PATH)

    # Try to load the fine-tuned entity CSV; fall back to empty if not found
    if Path(ENTITIES_CSV).exists():
        entity_groups = load_entities_csv(ENTITIES_CSV)
        print(f"Loaded entity groups: { {k: len(v) for k, v in entity_groups.items()} }")
    else:
        print(f"Entity CSV not found at {ENTITIES_CSV}")
        print("Using only hand-coded relationship nodes.\n")
        entity_groups = defaultdict(set)

    print("\n=== Connecting to Memgraph ===")
    mg = connect_memgraph()

    print("\n=== Populating knowledge graph ===")
    populate_graph(mg, rules, entity_groups)

    print("\n=== Done ===")
    print("Open Memgraph Lab at http://localhost:3000 to visualize.")
    print("Run the 5 Cypher queries from 05_cypher_queries.cypher")
