"""
File Name:    coreference.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import re
import csv
from pathlib import Path

import src.Project2.Project2Globals as Globals
import src.Project2.Data_Extraction.data_extractor as DataExtractor
from src.Project2.Play_Configs.play_configs import PLAY_CONFIGS

WINDOW_SIZE = 7

MALE_PRONOUNS    = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS  = {"she", "her", "hers", "herself"}
NEUTRAL_PRONOUNS = {"they", "them", "their", "themselves", "it", "its"}
ALL_PRONOUNS     = MALE_PRONOUNS | FEMALE_PRONOUNS | NEUTRAL_PRONOUNS

SPEAKER_RE = re.compile(r'([A-Z][A-Z\s]+)\.\s+')

def split_into_utterances(scene_text: str) -> list:
    """Split scene text into list of {speaker, text} dicts."""
    parts      = SPEAKER_RE.split(scene_text)
    utterances = []
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].strip()
        text    = parts[i + 1].strip() if i + 1 < len(parts) else ""
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
        self.nlp          = nlp
        self.male_chars   = {c.upper() for c in male_chars}
        self.female_chars = {c.upper() for c in female_chars}
        self.window_size  = window_size
        self.recent_persons = []

    def reset(self):
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
        return candidates[0][0], candidates[0][1]

    def resolve_scene(self, scene_text: str, act_id: str, scene_id: str, play_title: str) -> list:
        records    = []
        utterances = split_into_utterances(scene_text)

        for utt in utterances:
            speaker  = utt["speaker"]
            sent_doc = self.nlp(utt["text"])
            sentences = list(sent_doc.sents)

            for i, sent in enumerate(sentences):
                sent_nlp = self.nlp(sent.text)
                self._update_window(sent_nlp, speaker=speaker)

                # Find pronouns in the current sentence
                pronouns_found = [
                    token for token in sent_nlp
                    if token.pos_ == "PRON" and token.text.lower() in ALL_PRONOUNS
                ]

                if not pronouns_found or not self.recent_persons:
                    continue

                sentence_text = sent.text

                for pronoun_token in pronouns_found:
                    resolved, gender_group = self._best_candidate(
                        pronoun_token.text.lower()
                    )

                    # Replace pronoun with resolved name in the sentence text
                    resolved_text = re.sub(
                        r'\b' + re.escape(pronoun_token.text) + r'\b',
                        resolved,
                        sentence_text,
                        count=1,
                        flags=re.IGNORECASE
                    )

                    print(
                        f"[Resolved] '{sent.text.strip()}'\n"
                        f"        -> '{resolved_text.strip()}'\n"
                        f"           (pronoun '{pronoun_token.text}' -> '{resolved}')\n"
                    )

                    records.append({
                        "play":              play_title,
                        "act":               act_id,
                        "scene":             scene_id,
                        "speaker":           speaker,
                        "original_sentence": sent.text.strip()[:120],
                        "pronoun":           pronoun_token.text,
                        "resolved_to":       resolved,
                        "resolved_sentence": resolved_text.strip()[:120],
                        "gender_group":      gender_group,
                        "window_size_used":  min(self.window_size, len(self.recent_persons)),
                    })

                    # Update sentence_text so subsequent pronouns in same sentence
                    # use the already-resolved version
                    sentence_text = resolved_text

        return records

COREF_FIELDS = [
    "play", "act", "scene", "speaker",
    "original_sentence", "pronoun", "resolved_to",
    "resolved_sentence", "gender_group", "window_size_used"
]

def save_csv(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COREF_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"  → Saved {len(records):,} records  :  {path.name}")

# ─────────────────────────────────────────────
# RUN  (called from Project2Runner)
# ─────────────────────────────────────────────

def run(nlp, play_files: list) -> list:
    all_coref_records = []

    for play_file in play_files:
        print(f"\n{'─'*60}")
        print(f"  {play_file.name}")
        print(f"{'─'*60}")

        root   = DataExtractor.load_play(play_file)[0]
        title  = DataExtractor.get_title(root)
        scenes = DataExtractor.extract_scenes(root)

        config       = PLAY_CONFIGS.get(play_file.name, {})
        male_chars   = config.get("male_characters", set())
        female_chars = config.get("female_characters", set())

        if not config:
            print(f"  WARNING: No config in play_configs.py for {play_file.name}")
            print(f"           Pronoun resolution will use fallback (most-recent only)")

        resolver     = CoreferenceResolver(nlp, male_chars, female_chars)
        play_records = []

        for act_id, scene_id, location, text in scenes:
            resolver.reset()
            records = resolver.resolve_scene(text, act_id, scene_id, title)
            play_records.extend(records)

        print(f"  Title             : {title}")
        print(f"  Scenes            : {len(scenes)}")
        print(f"  Pronouns resolved : {len(play_records):,}")

        save_csv(play_records, Globals.OUTPUT_DIR / f"03_{play_file.stem}_coreference.csv")
        all_coref_records.extend(play_records)

    save_csv(all_coref_records, Globals.OUTPUT_DIR / "03_ALL_coreference.csv")
    return all_coref_records