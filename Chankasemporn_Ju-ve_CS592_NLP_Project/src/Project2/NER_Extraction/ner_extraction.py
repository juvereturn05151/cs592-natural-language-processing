"""
File Name:    ner_extraction.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import spacy
import re
import csv
from pathlib import Path
from collections import defaultdict

import src.Project2.Project2Globals as Globals
import src.Project2.Data_Extraction.data_extractor as DataExtractor

# ─────────────────────────────────────────────
# ENTITY NORMALISATION
# ─────────────────────────────────────────────

# Archaic Shakespeare suffixes that indicate a verb/adjective, not an entity
# e.g. "carv'd", "Dismay'd", "hors'd", "return'd", "might'st"
ARCHAIC_SUFFIX_RE = re.compile(
    r"'(d|st|t|est|eth|s|n|er|ing)\b", re.IGNORECASE
)

# Labels where we normalise to uppercase (character names)
PERSON_LABELS = {"PERSON"}

# Labels where we apply lemmatization (places, organisations)
LEMMA_LABELS = {"GPE", "ORG", "LOC", "NORP"}

def is_noise_entity(text: str, label: str) -> bool:
    """
    Returns True if the entity is likely noise and should be filtered out.
    Catches:
      - Archaic verb forms picked up as entities (carv'd, Dismay'd)
      - Stage directions picked up as entities (Exit Captain, Alarum)
      - Very short tokens that are unlikely to be real entities
    """
    # Filter archaic inflected verb forms
    if ARCHAIC_SUFFIX_RE.search(text):
        return True

    # Filter stage directions commonly misidentified as PERSON
    stage_directions = {
        "alarum", "alarums", "exeunt", "exit", "enter",
        "flourish", "hautboys", "sennet", "aside", "within"
    }
    if text.lower().strip() in stage_directions:
        return True

    # Filter entities that start with stage direction words
    if re.match(r'^(Exit|Enter|Exeunt)\s', text, re.IGNORECASE):
        return True

    # Filter single characters and very short noise tokens
    if len(text.strip()) < 2:
        return True

    return False

def normalise_entity(text: str, label: str, nlp) -> str:
    """
    Normalise entity text:
      - All labels → uppercase to eliminate casing duplicates
        (Duncan/DUNCAN, one/One/ONE, first/First/FIRST)
      - GPE/ORG/LOC → also lemmatize via spaCy to get root form
      - Strip possessives e.g. "Hecate's" → "Hecate"
    """
    # Strip possessives first e.g. "Pale Hecate's" → "Pale Hecate"
    text = re.sub(r"'s$", "", text).strip()

    if label in LEMMA_LABELS:
        doc = nlp(text)
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        if tokens:
            text = " ".join(t.lemma_.capitalize() for t in tokens)

    # Uppercase everything to eliminate all casing duplicates
    return text.upper()

def run_default_ner(scenes: list, nlp, play_title: str) -> tuple:
    """
    Run spaCy NER on all scenes.
    Returns deduplicated entity records — one row per unique
    (entity_text, spacy_label) pair per play, not one row per occurrence.
    """
    # Use a set to track seen (normalised_text, label) pairs — prevents duplicates
    seen        = set()
    all_entities   = []
    entity_summary = defaultdict(set)

    for act_id, scene_id, location, text in scenes:
        doc = nlp(text)
        for ent in doc.ents:
            raw_text = ent.text.strip()

            # Step 1: filter out noise entities
            if is_noise_entity(raw_text, ent.label_):
                continue

            # Step 2: normalise — uppercase everything, lemmatize GPE/ORG/LOC
            normalised_text = normalise_entity(raw_text, ent.label_, nlp)

            # Step 3: deduplicate — skip if we've already seen this entity+label
            dedup_key = (normalised_text, ent.label_)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            record = {
                "play":              play_title,
                "act":               act_id,
                "scene":             scene_id,
                "location":          location,
                "entity_text":       normalised_text,
                "spacy_label":       ent.label_,
                "label_description": spacy.explain(ent.label_) or ent.label_,
                "start_char":        ent.start_char,
                "end_char":          ent.end_char,
            }
            all_entities.append(record)
            entity_summary[ent.label_].add(normalised_text)

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