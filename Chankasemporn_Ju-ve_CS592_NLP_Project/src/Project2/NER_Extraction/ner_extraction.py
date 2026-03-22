"""
File Name:    ner_extraction.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import spacy
import csv
from pathlib import Path
from collections import defaultdict

import src.Project2.Project2Globals as Globals
import src.Project2.Data_Extraction.data_extractor as DataExtractor

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

def run(nlp, play_files):
    all_records = []
    all_mislabel = []
    all_cast_rels = []

    for play_file in play_files:
        print(f"\n{'─' * 60}")
        print(f"  {play_file.name}")
        print(f"{'─' * 60}")

        root, _ = DataExtractor.load_play(play_file)
        title = DataExtractor.get_title(root)
        characters = DataExtractor.extract_cast(root)
        scenes = DataExtractor.extract_scenes(root)

        print(f"  Title      : {title}")
        print(f"  Characters : {len(characters)}")
        print(f"  Scenes     : {len(scenes)}")

        records, summary = run_default_ner(scenes, nlp, title)
        print(f"  Entities   : {len(records):,} found by default model")

        mislabeled = analyse_labels(summary, characters, title)
        all_mislabel.extend(mislabeled)

        save_csv(records, Globals.OUTPUT_DIR / f"01_{play_file.stem}_default_ner.csv")
        all_records.extend(records)

        # Extract relationships from cast list descriptions
        print(f"\n  Cast relationships found:")
        cast_rels = DataExtractor.extract_cast_relationships(characters, title)
        if not cast_rels:
            print("    (none found)")
        all_cast_rels.extend(cast_rels)

    save_csv(all_records, Globals.OUTPUT_DIR / "01_ALL_default_ner.csv")
    save_cast_relationships_csv(
        all_cast_rels,
        Globals.OUTPUT_DIR / "01_ALL_cast_relationships.csv"
    )

    return all_records, all_mislabel, all_cast_rels