"""
CS 374 – Assignment 2
Notebook 3: Custom Pronoun / Coreference Resolution
Goes beyond the 1-sentence-back model:
  - Tracks a rolling window of N sentences
  - Disambiguates he/him vs she/her using gender heuristics
  - Uses the official character list for candidate validation
  - Outputs a CSV showing every pronoun and its resolved antecedent
"""

import spacy
import xml.etree.ElementTree as ET
import re
import csv
from pathlib import Path
from src.Project2.NER_Extraction.data_extractor import find_repo_root
from src.Project2.Play_Configs.play_configs import PLAY_CONFIGS

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

REPO_ROOT, _ = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = DATA_DIR / "output"
MODEL_PATH = REPO_ROOT / "models" / "shakespeare_ner"

WINDOW_SIZE = 7  # sentences to look back for antecedents

# Pronoun groups
MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
NEUTRAL_PRONOUNS = {"they", "them", "their", "themselves", "it", "its"}
ALL_PRONOUNS = MALE_PRONOUNS | FEMALE_PRONOUNS | NEUTRAL_PRONOUNS


# ─────────────────────────────────────────────
# 1. XML PARSING  (shared helpers)
# ─────────────────────────────────────────────

def clean_xml(raw: str) -> str:
    return (raw
            .replace("&#8217;", "'").replace("&#8216;", "'")
            .replace("&#8220;", '"').replace("&#8221;", '"')
            .replace("&#8212;", "—").replace("&#8211;", "–"))


def load_play(filepath: Path):
    raw = filepath.read_text(encoding="utf-8")
    root = ET.fromstring(clean_xml(raw))
    return root


def get_title(root) -> str:
    elem = root.find(".//Title")
    return elem.text.strip() if elem is not None and elem.text else "Unknown"


def extract_scenes(root) -> list:
    scenes = []
    for act in root.findall(".//Act"):
        act_id = act.attrib.get("id", "?")
        scene_elems = act.findall("Scene")
        if scene_elems:
            for scene in scene_elems:
                text = re.sub(r'\s+', ' ', " ".join(scene.itertext()).strip())
                if text:
                    scenes.append((
                        act_id,
                        scene.attrib.get("id", "?"),
                        scene.attrib.get("location", ""),
                        text
                    ))
        else:
            text = re.sub(r'\s+', ' ', " ".join(act.itertext()).strip())
            if text:
                scenes.append((act_id, "0", "Prologue", text))
    return scenes


# ─────────────────────────────────────────────
# 2. COREFERENCE RESOLVER
# ─────────────────────────────────────────────

SPEAKER_RE = re.compile(r'([A-Z][A-Z\s]+)\.\s+')


def split_into_utterances(scene_text: str) -> list:
    """Split scene text into list of {speaker, text} dicts."""
    parts = SPEAKER_RE.split(scene_text)
    utterances = []
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].strip()
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if speaker and text:
            utterances.append({"speaker": speaker, "text": text})
        i += 2
    return utterances


class CoreferenceResolver:
    """
    Sliding-window coreference resolver.
    Gender maps are loaded per-play from play_configs.py.
    """

    def __init__(self, nlp, male_chars: set, female_chars: set,
                 window_size: int = WINDOW_SIZE):
        self.nlp = nlp
        self.male_chars = {c.upper() for c in male_chars}
        self.female_chars = {c.upper() for c in female_chars}
        self.window_size = window_size
        self.recent_persons = []  # list of (name, gender)

    def reset(self):
        """Clear window between scenes."""
        self.recent_persons = []

    def _gender_of(self, name: str) -> str:
        uname = name.upper()
        if any(m in uname for m in self.male_chars):
            return "male"
        if any(f in uname for f in self.female_chars):
            return "female"
        return "neutral"

    def _update_window(self, doc, speaker: str = None):
        if speaker:
            self.recent_persons.append((speaker, self._gender_of(speaker)))
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                self.recent_persons.append(
                    (ent.text.strip(), self._gender_of(ent.text))
                )
        # Trim to rolling window
        self.recent_persons = self.recent_persons[-(self.window_size * 3):]

    def _best_candidate(self, pronoun_lower: str):
        candidates = list(reversed(self.recent_persons))
        if not candidates:
            return "UNKNOWN", "unknown"

        if pronoun_lower in MALE_PRONOUNS:
            for name, gender in candidates:
                if gender == "male":
                    return name, "male"
        elif pronoun_lower in FEMALE_PRONOUNS:
            for name, gender in candidates:
                if gender == "female":
                    return name, "female"

        # Neutral or fallback: most recent person of any gender
        return candidates[0][0], candidates[0][1]

    def resolve_scene(self, scene_text: str, act_id: str,
                      scene_id: str, play_title: str) -> list:
        records = []
        utterances = split_into_utterances(scene_text)

        for utt in utterances:
            speaker = utt["speaker"]
            sent_doc = self.nlp(utt["text"])

            for sent in sent_doc.sents:
                sent_nlp = self.nlp(sent.text)
                self._update_window(sent_nlp, speaker=speaker)

                for token in sent_nlp:
                    if token.pos_ == "PRON" and token.text.lower() in ALL_PRONOUNS:
                        if not self.recent_persons:
                            continue
                        resolved, gender_group = self._best_candidate(
                            token.text.lower()
                        )
                        records.append({
                            "play": play_title,
                            "act": act_id,
                            "scene": scene_id,
                            "speaker": speaker,
                            "sentence": sent.text.strip()[:120],
                            "pronoun": token.text,
                            "resolved_to": resolved,
                            "gender_group": gender_group,
                            "window_size_used": min(
                                self.window_size,
                                len(self.recent_persons)
                            ),
                        })
        return records


# ─────────────────────────────────────────────
# 3. SAVE CSV
# ─────────────────────────────────────────────

COREF_FIELDS = [
    "play", "act", "scene", "speaker", "sentence",
    "pronoun", "resolved_to", "gender_group", "window_size_used"
]


def save_csv(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COREF_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"  → Saved {len(records):,} records  :  {path.name}")
