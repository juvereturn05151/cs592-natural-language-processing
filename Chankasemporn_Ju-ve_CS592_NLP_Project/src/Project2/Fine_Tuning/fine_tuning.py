"""
CS 374 – Assignment 2
Notebook 2: Fine-Tuning spaCy NER for Shakespeare's Macbeth
Adds custom labels (PERSON corrections, GPE, LOCATION, TITLE)
and trains on auto-generated examples from the XML cast/scene data.
"""

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import xml.etree.ElementTree as ET
import re
import random
import csv
from collections import defaultdict
from pathlib import Path
from src.Project2.NER_Extraction.data_extractor import find_repo_root

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

REPO_ROOT, _ = find_repo_root()
DATA_DIR   = REPO_ROOT / "data"
FILE_PATH  = DATA_DIR / "train" / "Shakespeare_Macbeth.txt"
MODEL_OUT  = REPO_ROOT / "models" / "shakespeare_ner"
OUTPUT_CSV = DATA_DIR / "output" / "02_finetuned_ner_entities.csv"

# Real geographic places that should be GPE
KNOWN_GPES = {
    "Scotland", "England", "Forres", "Inverness", "Fife",
    "Dunsinane", "Northumberland", "Norway", "Aleppo",
    "Saint Colme's Inch", "Birnam Wood", "Birnam"
}

# Descriptive/fictional places → LOCATION (not a real GPE)
KNOWN_LOCATIONS = {
    "the Castle", "the Palace", "the heath", "A dark Cave",
    "the Plain", "the field", "a Wood", "the Court",
    "A Camp", "the Lobby"
}

# Titles / ranks — add a custom TITLE label
KNOWN_TITLES = {
    "Thane of Glamis", "Thane of Cawdor", "Thane of Fife",
    "King of Scotland", "Earl of Northumberland",
    "General of the English Forces"
}

# ─────────────────────────────────────────────
# 1. PARSE XML
# ─────────────────────────────────────────────

def load_and_clean(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = raw.replace("&#8217;", "'").replace("&#8216;", "'")
    raw = raw.replace("&#8220;", '"').replace("&#8221;", '"')
    raw = raw.replace("&#8212;", "—").replace("&#8211;", "–")
    return raw

def extract_cast(xml_content):
    """Return cleaned character name list from <Cast> section."""
    tree = ET.fromstring(xml_content)
    characters = []
    for char in tree.findall(".//Character"):
        raw = char.attrib.get("name", "")
        name = raw.split(",")[0].strip()
        # Skip purely generic roles like "A Soldier", "A Porter"
        if name and not re.match(r'^(A |An |The )(Soldier|Porter|Doctor|Man|Boy|Captain)', name):
            characters.append(name)
    return characters

def extract_scenes(xml_content):
    """Return list of (act_id, scene_id, location, text) tuples."""
    tree = ET.fromstring(xml_content)
    scenes = []
    for act in tree.findall(".//Act"):
        act_id = act.attrib.get("id")
        for scene in act.findall(".//Scene"):
            scene_id  = scene.attrib.get("id")
            location  = scene.attrib.get("location", "")
            text      = " ".join(scene.itertext()).strip()
            text      = re.sub(r'\s+', ' ', text)
            scenes.append((act_id, scene_id, location, text))
    return scenes

# ─────────────────────────────────────────────
# 2. AUTO-GENERATE TRAINING EXAMPLES
# ─────────────────────────────────────────────

def find_all_spans(text, phrase):
    """Find all non-overlapping occurrences of phrase in text."""
    spans = []
    start = 0
    phrase_lower = phrase.lower()
    text_lower   = text.lower()
    while True:
        idx = text_lower.find(phrase_lower, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(phrase)))
        start = idx + len(phrase)
    return spans

def remove_overlaps(entities):
    """Remove overlapping entity spans (keep first found)."""
    entities = sorted(entities, key=lambda x: x[0])
    result, last_end = [], -1
    for ent in entities:
        if ent[0] >= last_end:
            result.append(ent)
            last_end = ent[1]
    return result

def build_training_data(scenes, characters, nlp):
    """
    For each scene, scan the text for:
      - character names  → PERSON
      - known GPEs       → GPE
      - known LOCATIONs  → LOCATION
      - known TITLEs     → TITLE
    Returns a list of spaCy Example objects.
    """
    examples = []
    skipped  = 0

    # Split long scenes into sentences for better training granularity
    for act_id, scene_id, location, text in scenes:
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            entities = []

            for char in characters:
                for span in find_all_spans(sentence, char):
                    entities.append((span[0], span[1], "PERSON"))

            for gpe in KNOWN_GPES:
                for span in find_all_spans(sentence, gpe):
                    entities.append((span[0], span[1], "GPE"))

            for loc in KNOWN_LOCATIONS:
                for span in find_all_spans(sentence, loc):
                    entities.append((span[0], span[1], "LOCATION"))

            for title in KNOWN_TITLES:
                for span in find_all_spans(sentence, title):
                    entities.append((span[0], span[1], "TITLE"))

            entities = remove_overlaps(entities)

            if not entities:
                continue

            try:
                doc     = nlp.make_doc(sentence)
                example = Example.from_dict(doc, {"entities": entities})
                examples.append(example)
            except Exception as e:
                skipped += 1

    print(f"Built {len(examples)} training examples ({skipped} skipped due to alignment errors)")
    return examples

# ─────────────────────────────────────────────
# 3. FINE-TUNE THE MODEL
# ─────────────────────────────────────────────

def fine_tune(examples, n_iter=40):
    """
    Load en_core_web_md, add custom labels, and fine-tune.
    Uses dropout=0.35 and batch compounding for stable training.
    """
    print("Loading base model: en_core_web_md ...")
    nlp = spacy.load("en_core_web_md")

    ner = nlp.get_pipe("ner")

    # Add custom labels
    for label in ["PERSON", "GPE", "LOCATION", "TITLE"]:
        ner.add_label(label)

    # Disable other pipes during training (only train NER)
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    print(f"Fine-tuning for {n_iter} iterations ...")
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        optimizer.learn_rate = 0.001

        for i in range(n_iter):
            random.shuffle(examples)
            losses = {}
            batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=0.35, losses=losses)

            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1:3d}  NER loss: {losses.get('ner', 0):.4f}")

    return nlp

# ─────────────────────────────────────────────
# 4. EVALUATE AND SAVE FINE-TUNED ENTITIES
# ─────────────────────────────────────────────

def extract_finetuned_entities(nlp, scenes):
    """Run the fine-tuned model and return entity records."""
    records = []
    entity_summary = defaultdict(set)

    for act_id, scene_id, location, text in scenes:
        doc = nlp(text)
        for ent in doc.ents:
            records.append({
                "act": act_id,
                "scene": scene_id,
                "location": location,
                "entity_text": ent.text.strip(),
                "label": ent.label_,
                "label_description": spacy.explain(ent.label_) or ent.label_,
            })
            entity_summary[ent.label_].add(ent.text.strip())

    return records, entity_summary

def save_csv(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["act", "scene", "location", "entity_text", "label", "label_description"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} entity records → {path}")
