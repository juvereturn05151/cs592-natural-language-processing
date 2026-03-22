"""
File Name:    fine_tuning.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import re
import random
import csv
from collections import defaultdict

import src.Project2.Project2Globals as Globals
import src.Project2.Data_Extraction.data_extractor as DataExtractor
from src.Project2.Play_Configs.play_configs import PLAY_CONFIGS

#find all non-overlapping occurrences of phrase in text (case-insensitive).
def find_all_spans(text: str, phrase: str) -> list:
    spans      = []
    start      = 0
    text_lower = text.lower()
    phr_lower  = phrase.lower()
    while True:
        idx = text_lower.find(phr_lower, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(phrase)))
        start = idx + len(phrase)
    return spans

#keep first entity when spans overlap.
def remove_overlaps(entities: list) -> list:
    result, last_end = [], -1
    for ent in sorted(entities, key=lambda x: x[0]):
        if ent[0] >= last_end:
            result.append(ent)
            last_end = ent[1]
    return result

def build_training_data_for_play(scenes: list, characters: list, config: dict, nlp) -> tuple:
    known_gpes      = config.get("known_gpes", set())
    known_locations = config.get("known_locations", set())
    known_titles    = config.get("known_titles", set())

    #extract just the names for span matching
    char_names = [c["name"] if isinstance(c, dict) else c for c in characters]

    examples = []
    skipped  = 0

    for _, _, _, text in scenes:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            entities = []
            for name in char_names:
                for span in find_all_spans(sentence, name):
                    entities.append((span[0], span[1], "PERSON"))
            for gpe in known_gpes:
                for span in find_all_spans(sentence, gpe):
                    entities.append((span[0], span[1], "GPE"))
            for loc in known_locations:
                for span in find_all_spans(sentence, loc):
                    entities.append((span[0], span[1], "LOCATION"))
            for title in known_titles:
                for span in find_all_spans(sentence, title):
                    entities.append((span[0], span[1], "TITLE"))

            entities = remove_overlaps(entities)
            if not entities:
                continue

            try:
                doc     = nlp.make_doc(sentence)
                example = Example.from_dict(doc, {"entities": entities})
                examples.append(example)
            except Exception:
                skipped += 1

    return examples, skipped

#generate training examples from the cast list and cast relationships.
def build_training_data_from_cast(characters: list, cast_relationships: list, nlp) -> tuple:
    examples = []
    skipped  = 0

    # ── Source 1: simple character sentences ──────────────────────
    for char in characters:
        name = char["name"] if isinstance(char, dict) else char
        desc = char.get("desc", "") if isinstance(char, dict) else ""

        # "MACBETH is a character."
        sentence = f"{name} is a character."
        entities = [(0, len(name), "PERSON")]

        # If description exists, also label it as context
        # "MACBETH, General in the King's Army."
        if desc:
            sentence = f"{name}, {desc}."
            entities = [(0, len(name), "PERSON")]

        try:
            doc     = nlp.make_doc(sentence)
            example = Example.from_dict(doc, {"entities": entities})
            examples.append(example)
        except Exception:
            skipped += 1

    # ── Source 2: relationship sentences ──────────────────────────
    for rel in cast_relationships:
        source   = rel["source"]    # e.g. "HERO"
        rel_type = rel["rel_type"]  # e.g. "DAUGHTER_TO"
        target   = rel["target"]    # e.g. "Leonato"
        desc     = rel["description"]  # e.g. "Daughter to Leonato"

        # Build a natural sentence:
        # "HERO is Daughter to Leonato."
        sentence = f"{source} is {desc}."

        # Find character span (PERSON)
        source_start = 0
        source_end   = len(source)

        # Find the relationship word span (REL)
        # The rel word is everything in desc before the preposition
        # e.g. desc="Daughter to Leonato" → rel_word="Daughter"
        rel_word_match = re.match(
            r'^(?P<role>.+?)\s+(?:to|of|on|with|unto)\s+', desc, re.IGNORECASE
        )
        rel_word     = rel_word_match.group("role").strip() if rel_word_match else None
        is_start     = len(f"{source} is ")
        rel_start    = is_start
        rel_end      = is_start + len(rel_word) if rel_word else None

        # Find target span (PERSON)
        target_start = sentence.lower().find(target.lower())
        target_end   = target_start + len(target) if target_start != -1 else None

        entities = [(source_start, source_end, "PERSON")]
        if rel_end:
            entities.append((rel_start, rel_end, "REL"))
        if target_end and target_start != -1:
            entities.append((target_start, target_end, "PERSON"))

        entities = remove_overlaps(entities)

        try:
            doc     = nlp.make_doc(sentence)
            example = Example.from_dict(doc, {"entities": entities})
            examples.append(example)
        except Exception:
            skipped += 1

    return examples, skipped

def fine_tune(nlp, all_examples: list, n_iter: int = 40):
    print("\nLoading base model: en_core_web_md ...")
    ner = nlp.get_pipe("ner")

    #add all custom labels including new REL label
    for label in ["PERSON", "GPE", "LOCATION", "TITLE", "REL"]:
        ner.add_label(label)

    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    print(f"Fine-tuning on {len(all_examples):,} examples for {n_iter} iterations ...")
    with nlp.disable_pipes(*other_pipes):
        optimizer            = nlp.resume_training()
        optimizer.learn_rate = 0.001

        for i in range(n_iter):
            random.shuffle(all_examples)
            losses  = {}
            batches = minibatch(all_examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=0.35, losses=losses)
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1:3d}  NER loss: {losses.get('ner', 0):.4f}")

    return nlp

ENTITY_FIELDS = [
    "play", "act", "scene", "location",
    "entity_text", "label", "label_description"
]

#run fine-tuned model on scenes and save per-play CSV
def extract_and_save(nlp_ft, scenes: list, play_title: str, stem: str) -> list:
    records        = []
    entity_summary = defaultdict(set)

    for act_id, scene_id, location, text in scenes:
        doc = nlp_ft(text)
        for ent in doc.ents:
            records.append({
                "play":              play_title,
                "act":               act_id,
                "scene":             scene_id,
                "location":          location,
                "entity_text":       ent.text.strip(),
                "label":             ent.label_,
                "label_description": spacy.explain(ent.label_) or ent.label_,
            })
            entity_summary[ent.label_].add(ent.text.strip())

    path = Globals.OUTPUT_DIR / f"02_{stem}_finetuned_ner.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENTITY_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"  [{play_title}]  {len(records):,} entities  → {path.name}")
    for label, ents in sorted(entity_summary.items()):
        print(f"    {label:12s}: {len(ents)} unique")

    return records

def run(nlp_base, play_files: list, cast_relationships: list, n_iter: int = 40):
    all_examples = []
    play_data    = []

    for play_file in play_files:
        print(f"\n{'─'*60}")
        print(f"  Building training data: {play_file.name}")

        root       = DataExtractor.load_play(play_file)[0]
        title      = DataExtractor.get_title(root)
        characters = DataExtractor.extract_cast(root)
        scenes     = DataExtractor.extract_scenes(root)
        play_data.append((play_file, title, characters, scenes))

        config = PLAY_CONFIGS.get(play_file.name, {})
        if not config:
            print(f"  WARNING: No config found for {play_file.name}")

        # Source 1: scene dialogue
        scene_examples, skipped = build_training_data_for_play(
            scenes, characters, config, nlp_base
        )
        print(f"  Scene examples   : {len(scene_examples):,}  ({skipped} skipped)")

        # Source 2: cast list + cast relationships
        play_cast_rels = [r for r in cast_relationships if r["play"] == title]
        cast_examples, cast_skipped = build_training_data_from_cast(
            characters, play_cast_rels, nlp_base
        )
        print(f"  Cast examples    : {len(cast_examples):,}  ({cast_skipped} skipped)")
        print(f"    └─ incl. {len(play_cast_rels)} REL relationship sentences")

        all_examples.extend(scene_examples)
        all_examples.extend(cast_examples)

    print(f"\nTotal training examples: {len(all_examples):,}")

    # Fine-tune
    nlp_ft = fine_tune(nlp_base, all_examples, n_iter=n_iter)

    # Save model
    Globals.MODEL_OUT.mkdir(parents=True, exist_ok=True)
    nlp_ft.to_disk(Globals.MODEL_OUT)
    print(f"\nFine-tuned model saved → {Globals.MODEL_OUT}")

    # Extract and save per-play entities
    print("\n=== Extracting entities with fine-tuned model ===")
    all_ft_records = []
    for play_file, title, characters, scenes in play_data:
        records = extract_and_save(nlp_ft, scenes, title, play_file.stem)
        all_ft_records.extend(records)

    # Save combined CSV
    combined_path = Globals.OUTPUT_DIR / "02_ALL_finetuned_ner.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENTITY_FIELDS)
        writer.writeheader()
        writer.writerows(all_ft_records)
    print(f"  Combined CSV → {combined_path.name}")

    return nlp_ft, all_ft_records