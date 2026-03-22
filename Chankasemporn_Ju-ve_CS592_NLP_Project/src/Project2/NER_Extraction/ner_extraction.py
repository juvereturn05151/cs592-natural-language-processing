"""
CS 374 – Assignment 2
Notebook 1: Default NER Extraction using spaCy en_core_web_md
Runs Named-Entity Recognition on Shakespeare's Macbeth using the
default spaCy medium model and outputs a CSV of all found entities.
"""

import spacy
import csv
from pathlib import Path
from collections import defaultdict

# run DEFAULT spaCy NER
def run_default_ner(scenes: list, nlp, play_title: str) -> tuple:
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


# label analysis
#print label summary, flag mislabeled and missing characters.
#returns list of (entity_text, wrong_label, correct_label).
def analyse_labels(entity_summary: dict, official_characters: list, play_title: str) -> list:
    # Support both old format (strings) and new format (dicts with "name" key)
    char_names = [
        c["name"] if isinstance(c, dict) else c
        for c in official_characters
    ]
    char_set = {c.upper() for c in char_names}

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
                    print(f"    '{e}'  [{label}] -> should be PERSON")
                    mislabeled.append((e, label, "PERSON"))
    if not mislabeled:
        print("    (none detected)")

    print(f"\n  Missing characters (in cast but not found by spaCy):")
    all_found = {e.upper() for ents in entity_summary.values() for e in ents}
    missing = [c for c in char_names if c.upper() not in all_found]
    for c in missing:
        print(f"    '{c}'")
    if not missing:
        print("    (none)")

    return mislabeled


#save into a csv
ENTITY_FIELDS = [
    "play", "act", "scene", "location",
    "entity_text", "spacy_label", "label_description",
    "start_char", "end_char"
]

REL_FIELDS = [
    "play", "source", "rel_type", "target", "description"
]

def save_csv(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENTITY_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"  -> Saved {len(records):,} records  :  {path.name}")

def save_cast_relationships_csv(relationships: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REL_FIELDS)
        writer.writeheader()
        writer.writerows(relationships)
    print(f"  -> Saved {len(relationships):,} cast relationships  :  {path.name}")