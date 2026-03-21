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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

REPO_ROOT, _ = find_repo_root()
DATA_DIR   = REPO_ROOT / "data"
FILE_PATH  = DATA_DIR / "train" / "Shakespeare_Macbeth.txt"
MODEL_PATH = REPO_ROOT / "models" / "shakespeare_ner"
OUTPUT_CSV = DATA_DIR / "output" / "03_coreference_resolved.csv"

# Window size: how many previous sentences to scan for antecedents
WINDOW_SIZE = 7

# Gender mapping for Macbeth characters
# Male characters → resolved by he/him/his
MALE_CHARACTERS = {
    "MACBETH", "DUNCAN", "MALCOLM", "DONALBAIN", "BANQUO", "MACDUFF",
    "LENNOX", "ROSS", "MENTEITH", "ANGUS", "CAITHNESS", "FLEANCE",
    "SIWARD", "YOUNG SIWARD", "SEYTON", "BOY"
}

# Female characters → resolved by she/her/hers
FEMALE_CHARACTERS = {
    "LADY MACBETH", "LADY MACDUFF", "HECATE"
}

# Pronoun gender groups
MALE_PRONOUNS   = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
NEUTRAL_PRONOUNS = {"they", "them", "their", "themselves", "it", "its"}

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_and_clean(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    raw = raw.replace("&#8217;", "'").replace("&#8216;", "'")
    raw = raw.replace("&#8220;", '"').replace("&#8221;", '"')
    raw = raw.replace("&#8212;", "—").replace("&#8211;", "–")
    return raw

def extract_scenes(xml_content):
    tree = ET.fromstring(xml_content)
    scenes = []
    for act in tree.findall(".//Act"):
        act_id = act.attrib.get("id")
        for scene in act.findall(".//Scene"):
            scene_id = scene.attrib.get("id")
            location = scene.attrib.get("location", "")
            text     = " ".join(scene.itertext()).strip()
            text     = re.sub(r'\s+', ' ', text)
            scenes.append((act_id, scene_id, location, text))
    return scenes

# ─────────────────────────────────────────────
# 2. SENTENCE SPLITTER WITH SPEAKER DETECTION
# ─────────────────────────────────────────────

def split_into_utterances(scene_text):
    """
    Split scene text into utterances. Each utterance is
    a dict: {speaker, text, sentences}
    Speaker lines are in ALL CAPS followed by a period.
    """
    # Pattern: ALL_CAPS speaker name followed by period/newline
    pattern = re.compile(r'([A-Z][A-Z\s]+)\.\s+')
    parts   = pattern.split(scene_text)

    utterances = []
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].strip()
        text    = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if speaker and text:
            utterances.append({"speaker": speaker, "text": text})
        i += 2

    return utterances

# ─────────────────────────────────────────────
# 3. CORE COREFERENCE RESOLUTION ENGINE
# ─────────────────────────────────────────────

class CoreferenceResolver:
    """
    Sliding-window coreference resolver.

    Maintains a rolling buffer of recent PERSON entities.
    For each pronoun found in the current sentence:
      - Male pronouns → look for most recent MALE character
      - Female pronouns → look for most recent FEMALE character
      - Neutral pronouns → use most recently mentioned character of any gender
    """

    def __init__(self, nlp, window_size=7):
        self.nlp         = nlp
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset state between scenes."""
        self.recent_persons = []   # list of (name, gender) tuples, most recent last
        self.recent_speaker = None

    def _gender_of(self, name):
        uname = name.upper()
        for m in MALE_CHARACTERS:
            if m in uname:
                return "male"
        for f in FEMALE_CHARACTERS:
            if f in uname:
                return "female"
        return "neutral"

    def _update_window(self, doc, speaker=None):
        """Add PERSON entities from doc (and current speaker) to the window."""
        # Add current speaker to window
        if speaker:
            gender = self._gender_of(speaker)
            self.recent_persons.append((speaker, gender))

        # Add entities recognised in this sentence
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                gender = self._gender_of(ent.text)
                self.recent_persons.append((ent.text.strip(), gender))

        # Trim to window size
        self.recent_persons = self.recent_persons[-self.window_size * 3:]

    def _best_candidate(self, pronoun_lower):
        """Return the best antecedent for a given pronoun."""
        candidates = list(reversed(self.recent_persons))  # most recent first

        if pronoun_lower in MALE_PRONOUNS:
            for name, gender in candidates:
                if gender == "male":
                    return name, "male"
            # Fall back to any recent person
            return (candidates[0][0], "unknown") if candidates else ("UNKNOWN", "unknown")

        elif pronoun_lower in FEMALE_PRONOUNS:
            for name, gender in candidates:
                if gender == "female":
                    return name, "female"
            return (candidates[0][0], "unknown") if candidates else ("UNKNOWN", "unknown")

        else:  # neutral
            return (candidates[0][0], "neutral") if candidates else ("UNKNOWN", "neutral")

    def resolve_scene(self, scene_text, act_id, scene_id):
        """
        Process one scene and return a list of resolution records.
        Each record: {act, scene, sentence, pronoun, resolved_to, gender_match}
        """
        records    = []
        utterances = split_into_utterances(scene_text)

        for utterance in utterances:
            speaker   = utterance["speaker"]
            utt_text  = utterance["text"]
            sentences = list(self.nlp(utt_text).sents)

            for sent in sentences:
                sent_doc = self.nlp(sent.text)

                # Update window with named entities in this sentence
                self._update_window(sent_doc, speaker=speaker)

                # Find pronouns and resolve
                for token in sent_doc:
                    if token.pos_ == "PRON":
                        pron_lower = token.text.lower()
                        if pron_lower in (MALE_PRONOUNS | FEMALE_PRONOUNS | NEUTRAL_PRONOUNS):
                            if not self.recent_persons:
                                continue  # nothing to resolve to yet
                            resolved, gender_group = self._best_candidate(pron_lower)
                            records.append({
                                "act": act_id,
                                "scene": scene_id,
                                "speaker": speaker,
                                "sentence": sent.text.strip()[:120],
                                "pronoun": token.text,
                                "resolved_to": resolved,
                                "gender_group": gender_group,
                                "window_size_used": min(self.window_size, len(self.recent_persons))
                            })

        return records

# ─────────────────────────────────────────────
# 4. APPLY TO ALL SCENES
# ─────────────────────────────────────────────

def resolve_all_scenes(scenes, nlp):
    resolver     = CoreferenceResolver(nlp, window_size=WINDOW_SIZE)
    all_records  = []
    total_pronouns = 0

    for act_id, scene_id, location, text in scenes:
        resolver.reset()  # fresh window per scene
        records = resolver.resolve_scene(text, act_id, scene_id)
        all_records.extend(records)
        total_pronouns += len(records)
        print(f"  Act {act_id} Scene {scene_id}: {len(records)} pronouns resolved")

    print(f"\nTotal pronouns resolved: {total_pronouns}")
    return all_records

def save_csv(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["act", "scene", "speaker", "sentence", "pronoun",
                  "resolved_to", "gender_group", "window_size_used"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved coreference records → {path}")
