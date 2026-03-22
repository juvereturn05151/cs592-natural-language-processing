"""
CS 374 – Assignment 2
Notebook 1: Default NER Extraction using spaCy en_core_web_md
Runs Named-Entity Recognition on Shakespeare's Macbeth using the
default spaCy medium model and outputs a CSV of all found entities.
"""

import spacy
import xml.etree.ElementTree as ET
import re
import csv
from pathlib import Path
from collections import defaultdict
from src.Project2.NER_Extraction.data_extractor import find_repo_root

# ─────────────────────────────────────────────
# 1. RESOLVE PATHS VIA REPO ROOT
# ─────────────────────────────────────────────

REPO_ROOT, _ = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = DATA_DIR / "output"


# ─────────────────────────────────────────────
# 1. XML PARSING HELPERS
# ─────────────────────────────────────────────

def clean_xml(raw: str) -> str:
    """Replace common HTML entities so ET can parse cleanly."""
    return (raw
            .replace("&#8217;", "'").replace("&#8216;", "'")
            .replace("&#8220;", '"').replace("&#8221;", '"')
            .replace("&#8212;", "—").replace("&#8211;", "–"))


def load_play(filepath: Path):
    """Read and parse one play XML file. Returns (xml_root, cleaned_string)."""
    raw = filepath.read_text(encoding="utf-8")
    raw = clean_xml(raw)
    root = ET.fromstring(raw)
    return root, raw


def get_title(root) -> str:
    """Extract play title from <Title> tag."""
    elem = root.find(".//Title")
    return elem.text.strip() if elem is not None and elem.text else "Unknown"


def extract_cast(root) -> list:
    """Return cleaned character names from <Cast> section."""
    characters = []
    for char in root.findall(".//Character"):
        raw = char.attrib.get("name", "")
        name = raw.split(",")[0].strip()
        # Skip purely generic role descriptions
        if name and not re.match(
                r'^(A |An |The )(Soldier|Porter|Doctor|Man|Boy|Captain|Servant|'
                r'Sexton|Officer|Apothecary|Messenger|Attendant|Musician)', name
        ):
            characters.append(name)
    return characters


def extract_scenes(root) -> list:
    """
    Return list of (act_id, scene_id, location, text) for every scene.
    Handles Prologue-style acts (e.g. Romeo & Juliet Act 0) that contain
    text directly inside <Act> with no <Scene> children.
    """
    scenes = []
    for act in root.findall(".//Act"):
        act_id = act.attrib.get("id", "?")
        scene_elems = act.findall("Scene")

        if scene_elems:
            for scene in scene_elems:
                scene_id = scene.attrib.get("id", "?")
                location = scene.attrib.get("location", "")
                text = re.sub(r'\s+', ' ', " ".join(scene.itertext()).strip())
                if text:
                    scenes.append((act_id, scene_id, location, text))
        else:
            # Prologue: text is directly inside <Act>
            text = re.sub(r'\s+', ' ', " ".join(act.itertext()).strip())
            if text:
                scenes.append((act_id, "0", "Prologue", text))

    return scenes


# ─────────────────────────────────────────────
# 2. DEFAULT spaCy NER
# ─────────────────────────────────────────────

def run_default_ner(scenes: list, nlp, play_title: str) -> tuple:
    """
    Run en_core_web_md on all scenes of one play.
    Returns (entity_records_list, entity_summary_dict).
    """
    all_entities = []
    entity_summary = defaultdict(set)

    for act_id, scene_id, location, text in scenes:
        doc = nlp(text)
        for ent in doc.ents:
            record = {
                "play": play_title,
                "act": act_id,
                "scene": scene_id,
                "location": location,
                "entity_text": ent.text.strip(),
                "spacy_label": ent.label_,
                "label_description": spacy.explain(ent.label_) or ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            }
            all_entities.append(record)
            entity_summary[ent.label_].add(ent.text.strip())

    return all_entities, entity_summary


# ─────────────────────────────────────────────
# 3. LABEL ANALYSIS
# ─────────────────────────────────────────────

def analyse_labels(entity_summary: dict, official_characters: list, play_title: str) -> list:
    """
    Print label summary, flag mislabeled and missing characters.
    Returns list of (entity_text, wrong_label, correct_label).
    """
    char_set = {c.upper() for c in official_characters}

    print(f"\n  Label summary:")
    for label, ents in sorted(entity_summary.items()):
        preview = sorted(ents)[:6]
        more = f"  (+{len(ents) - 6} more)" if len(ents) > 6 else ""
        print(f"    [{label:12s}] {len(ents):3d} unique  e.g. {preview}{more}")

    print(f"\n  Mislabeled characters (found but wrong label):")
    mislabeled = []
    for label, ents in entity_summary.items():
        if label != "PERSON":
            for e in ents:
                if e.upper() in char_set or any(c in e.upper() for c in char_set):
                    print(f"    '{e}'  [{label}] → should be PERSON")
                    mislabeled.append((e, label, "PERSON"))
    if not mislabeled:
        print("    (none detected)")

    print(f"\n  Missing characters (in cast but not found by spaCy):")
    all_found = {e.upper() for ents in entity_summary.values() for e in ents}
    missing = [c for c in official_characters if c.upper() not in all_found]
    for c in missing:
        print(f"    '{c}'")
    if not missing:
        print("    (none)")

    return mislabeled


# ─────────────────────────────────────────────
# 4. CSV SAVE
# ─────────────────────────────────────────────

ENTITY_FIELDS = [
    "play", "act", "scene", "location",
    "entity_text", "spacy_label", "label_description",
    "start_char", "end_char"
]


def save_csv(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENTITY_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"  → Saved {len(records):,} records  :  {path.name}")
