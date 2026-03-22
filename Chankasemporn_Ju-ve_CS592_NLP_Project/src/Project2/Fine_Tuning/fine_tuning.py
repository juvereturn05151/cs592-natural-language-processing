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
from src.Project2.Data_Extraction.data_extractor import find_repo_root

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

REPO_ROOT, _ = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = DATA_DIR / "output"
MODEL_OUT = REPO_ROOT / "models" / "shakespeare_ner"


# ─────────────────────────────────────────────
# 1. XML PARSING  (shared with 01)
# ─────────────────────────────────────────────

def clean_xml(raw: str) -> str:
    return (raw
            .replace("&#8217;", "'").replace("&#8216;", "'")
            .replace("&#8220;", '"').replace("&#8221;", '"')
            .replace("&#8212;", "—").replace("&#8211;", "–"))


def load_play(filepath: Path):
    raw = filepath.read_text(encoding="utf-8")
    raw = clean_xml(raw)
    root = ET.fromstring(raw)
    return root


def get_title(root) -> str:
    elem = root.find(".//Title")
    return elem.text.strip() if elem is not None and elem.text else "Unknown"


def extract_cast(root) -> list:
    characters = []
    for char in root.findall(".//Character"):
        raw = char.attrib.get("name", "")
        name = raw.split(",")[0].strip()
        if name and not re.match(
                r'^(A |An |The )(Soldier|Porter|Doctor|Man|Boy|Captain|Servant|'
                r'Sexton|Officer|Apothecary|Messenger|Attendant|Musician)', name
        ):
            characters.append(name)
    return characters


def extract_scenes(root) -> list:
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
            text = re.sub(r'\s+', ' ', " ".join(act.itertext()).strip())
            if text:
                scenes.append((act_id, "0", "Prologue", text))
    return scenes


# ─────────────────────────────────────────────
# 2. TRAINING DATA GENERATION
# ─────────────────────────────────────────────

def find_all_spans(text: str, phrase: str) -> list:
    """Find all non-overlapping occurrences of phrase in text (case-insensitive)."""
    spans = []
    start = 0
    text_lower = text.lower()
    phr_lower = phrase.lower()
    while True:
        idx = text_lower.find(phr_lower, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(phrase)))
        start = idx + len(phrase)
    return spans


def remove_overlaps(entities: list) -> list:
    """Keep first entity when spans overlap."""
    result, last_end = [], -1
    for ent in sorted(entities, key=lambda x: x[0]):
        if ent[0] >= last_end:
            result.append(ent)
            last_end = ent[1]
    return result


def build_training_data_for_play(scenes: list, characters: list,
                                 config: dict, nlp) -> list:
    """
    Scan every sentence in a play for known characters, GPEs,
    locations, and titles, and return spaCy Example objects.
    """
    known_gpes = config.get("known_gpes", set())
    known_locations = config.get("known_locations", set())
    known_titles = config.get("known_titles", set())

    examples = []
    skipped = 0

    for _, _, _, text in scenes:
        # Split into sentences for finer granularity
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            entities = []
            for name in characters:
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
                doc = nlp.make_doc(sentence)
                example = Example.from_dict(doc, {"entities": entities})
                examples.append(example)
            except Exception:
                skipped += 1

    return examples, skipped


# ─────────────────────────────────────────────
# 3. FINE-TUNE
# ─────────────────────────────────────────────

def fine_tune(all_examples: list, n_iter: int = 40):
    """
    Load en_core_web_md, register all custom labels,
    and fine-tune on the combined example set from all plays.
    """
    print("\nLoading base model: en_core_web_md ...")
    nlp = spacy.load("en_core_web_md")
    ner = nlp.get_pipe("ner")

    for label in ["PERSON", "GPE", "LOCATION", "TITLE"]:
        ner.add_label(label)

    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    print(f"Fine-tuning on {len(all_examples):,} examples for {n_iter} iterations ...")
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        optimizer.learn_rate = 0.001

        for i in range(n_iter):
            random.shuffle(all_examples)
            losses = {}
            batches = minibatch(all_examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=0.35, losses=losses)
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1:3d}  NER loss: {losses.get('ner', 0):.4f}")

    return nlp


# ─────────────────────────────────────────────
# 4. EXTRACT + SAVE FINE-TUNED ENTITIES
# ─────────────────────────────────────────────

ENTITY_FIELDS = [
    "play", "act", "scene", "location",
    "entity_text", "label", "label_description"
]


def extract_and_save(nlp_ft, scenes: list, play_title: str,
                     stem: str) -> list:
    """Run fine-tuned model on scenes and save per-play CSV."""
    records = []
    entity_summary = defaultdict(set)

    for act_id, scene_id, location, text in scenes:
        doc = nlp_ft(text)
        for ent in doc.ents:
            records.append({
                "play": play_title,
                "act": act_id,
                "scene": scene_id,
                "location": location,
                "entity_text": ent.text.strip(),
                "label": ent.label_,
                "label_description": spacy.explain(ent.label_) or ent.label_,
            })
            entity_summary[ent.label_].add(ent.text.strip())

    path = OUTPUT_DIR / f"02_{stem}_finetuned_ner.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENTITY_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"  [{play_title}]  {len(records):,} entities  → {path.name}")

    # Print label breakdown
    for label, ents in sorted(entity_summary.items()):
        print(f"    {label:12s}: {len(ents)} unique")

    return records
