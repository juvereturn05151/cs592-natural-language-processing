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
from collections import defaultdict
from src.Project2.NER_Extraction.data_extractor import find_repo_root

# ─────────────────────────────────────────────
# 1. RESOLVE PATHS VIA REPO ROOT
# ─────────────────────────────────────────────

REPO_ROOT, _ = find_repo_root()
DATA_DIR   = REPO_ROOT / "data"
FILE_PATH  = DATA_DIR / "train" / "Shakespeare_Macbeth.txt"
OUTPUT_CSV = DATA_DIR / "output" / "01_default_ner_entities.csv"

def load_and_clean(path):
    """Read the XML file and clean HTML entities for proper parsing."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Replace common HTML entities
    raw = raw.replace("&#8217;", "'")
    raw = raw.replace("&#8216;", "'")
    raw = raw.replace("&#8220;", '"')
    raw = raw.replace("&#8221;", '"')
    raw = raw.replace("&#8212;", "—")
    raw = raw.replace("&#8211;", "–")
    return raw

def extract_dialogue(xml_content):
    """
    Parse the XML and return all dialogue text, grouped by scene.
    Returns: list of dicts with keys: act, scene, location, text
    """
    tree = ET.fromstring(xml_content)
    scenes = []

    for act_elem in tree.findall(".//Act"):
        act_id = act_elem.attrib.get("id", "?")
        for scene_elem in act_elem.findall(".//Scene"):
            scene_id = scene_elem.attrib.get("id", "?")
            location = scene_elem.attrib.get("location", "")
            # Get all text within this scene
            full_text = " ".join(scene_elem.itertext()).strip()
            full_text = re.sub(r'\s+', ' ', full_text)
            if full_text:
                scenes.append({
                    "act": act_id,
                    "scene": scene_id,
                    "location": location,
                    "text": full_text
                })
    return scenes

def extract_cast(xml_content):
    """Extract the official character list from the <Cast> section."""
    tree = ET.fromstring(xml_content)
    characters = []
    for char in tree.findall(".//Character"):
        raw_name = char.attrib.get("name", "")
        # Take only the part before a comma (e.g. "MACBETH, General..." → "MACBETH")
        name = raw_name.split(",")[0].strip()
        if name:
            characters.append(name)
    return characters

# ─────────────────────────────────────────────
# 2. RUN DEFAULT spaCy NER
# ─────────────────────────────────────────────

def run_default_ner(scenes):
    """
    Run spaCy's en_core_web_md on all scene dialogue.
    Returns a list of entity dicts.
    """
    print("Loading spaCy model: en_core_web_md ...")
    nlp = spacy.load("en_core_web_md")

    all_entities = []
    entity_summary = defaultdict(set)

    for scene in scenes:
        doc = nlp(scene["text"])
        for ent in doc.ents:
            record = {
                "act": scene["act"],
                "scene": scene["scene"],
                "location": scene["location"],
                "entity_text": ent.text.strip(),
                "spacy_label": ent.label_,
                "label_description": spacy.explain(ent.label_),
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            }
            all_entities.append(record)
            entity_summary[ent.label_].add(ent.text.strip())

    return all_entities, entity_summary

# ─────────────────────────────────────────────
# 3. IDENTIFY MISLABELED / MISSING ENTITIES
# ─────────────────────────────────────────────

def analyse_labels(entity_summary, official_characters):
    """
    Compare spaCy output against the official cast list.
    Flags:
      - Characters spaCy labeled as something other than PERSON
      - Characters spaCy missed entirely
    """
    char_set = {c.upper() for c in official_characters}

    print("\n=== LABEL SUMMARY (default model) ===")
    for label, ents in sorted(entity_summary.items()):
        print(f"\n[{label}] ({len(ents)} unique):")
        for e in sorted(ents)[:15]:  # show up to 15 per label
            print(f"   {e}")
        if len(ents) > 15:
            print(f"   ... and {len(ents)-15} more")

    # Check for characters mislabeled
    print("\n=== MISLABELED CHARACTERS (found by spaCy but wrong label) ===")
    mislabeled = []
    for label, ents in entity_summary.items():
        if label != "PERSON":
            for e in ents:
                if e.upper() in char_set or any(c in e.upper() for c in char_set):
                    print(f"  '{e}' labeled as [{label}] — should be PERSON")
                    mislabeled.append((e, label, "PERSON"))

    # Check for characters completely missed
    print("\n=== MISSING CHARACTERS (in cast list but not found by spaCy) ===")
    all_found = {e.upper() for ents in entity_summary.values() for e in ents}
    for char in official_characters:
        if char.upper() not in all_found:
            print(f"  '{char}' — not found at all by spaCy")

    return mislabeled

# ─────────────────────────────────────────────
# 4. SAVE TO CSV
# ─────────────────────────────────────────────

def save_csv(entities, path):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["act", "scene", "location", "entity_text", "spacy_label",
                  "label_description", "start_char", "end_char"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entities)
    print(f"\nSaved {len(entities)} entity records to: {path}")
