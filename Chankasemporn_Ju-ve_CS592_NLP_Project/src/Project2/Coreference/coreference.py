"""
File Name:    coreference.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import re
import csv
from pathlib import Path
from collections import defaultdict

import src.Project2.Project2Globals as Globals
import src.Project2.Data_Extraction.data_extractor as DataExtractor
from src.Project2.Play_Configs.play_configs import PLAY_CONFIGS

MALE_PRONOUNS    = {"he", "him", "himself"}
FEMALE_PRONOUNS  = {"she", "her", "hers", "herself"}
ALL_PRONOUNS     = MALE_PRONOUNS | FEMALE_PRONOUNS

COREF_FIELDS = [
    "play", "act", "scene", "speaker",
    "original_sentence", "pronoun", "resolved_to",
    "resolved_sentence", "gender_group", "resolution_source"
]

SPEAKER_RE = re.compile(r'([A-Z][A-Z\s]+)\.\s+')

#split scene text into list of {speaker, text} dicts.
def split_into_utterances(scene_text: str) -> list:
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


#relationship-aware coreference resolver.
class CoreferenceResolver:
    def __init__(self, nlp, male_chars: set, female_chars: set,
                 relationships: list):
        self.nlp          = nlp
        self.male_chars   = {c.upper() for c in male_chars}
        self.female_chars = {c.upper() for c in female_chars}

        # Build adjacency: speaker_upper -> list of (target, rel_type)
        # Only PERSON->PERSON edges are useful for pronoun resolution
        self.rel_targets: dict = defaultdict(list)
        for rel in relationships:
            src, src_type, rel_type, tgt, tgt_type = rel
            if src_type == "PERSON" and tgt_type == "PERSON":
                self.rel_targets[src.upper()].append(tgt.upper())
            # Bidirectional — also index the target side
            if tgt_type == "PERSON" and src_type == "PERSON":
                self.rel_targets[tgt.upper()].append(src.upper())

    def _gender_of(self, name: str) -> str:
        uname = name.upper()
        if any(m in uname for m in self.male_chars):
            return "male"
        if any(f in uname for f in self.female_chars):
            return "female"
        return "none"

    def _gender_matches(self, name: str, pronoun_lower: str) -> bool:
        g = self._gender_of(name)
        if pronoun_lower in MALE_PRONOUNS:
            return g == "male"
        if pronoun_lower in FEMALE_PRONOUNS:
            return g == "female"
        return False

    #return list of unique PERSON entity names in appearance order.
    def _persons_in_doc(self, doc) -> list:
        seen = set()
        result = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip().upper()
                if name not in seen:
                    seen.add(name)
                    result.append(name)
        return result

    #return last gender-matching name from a list, or None.
    def _pick_from_list(self, names: list, pronoun_lower: str):
        for name in reversed(names):
            if self._gender_matches(name, pronoun_lower):
                return name
        return None

    def _resolve_via_relationships(self, speaker: str, pronoun_lower: str, prev_names: list):
        candidates = self.rel_targets.get(speaker.upper(), [])
        if not candidates:
            return None

        #prefer candidates that were also in the previous utterance
        prev_set = {n.upper() for n in prev_names}
        prioritised = [c for c in candidates if c in prev_set] + \
                      [c for c in candidates if c not in prev_set]

        for cand in prioritised:
            if self._gender_matches(cand, pronoun_lower):
                return cand
        return None

    def resolve_scene(self, scene_text: str, act_id: str,
                      scene_id: str, play_title: str) -> list:
        records    = []
        utterances = split_into_utterances(scene_text)
        prev_utt_names: list = []   # PERSON names from the previous utterance

        for utt in utterances:
            speaker  = utt["speaker"]
            utt_doc  = self.nlp(utt["text"])
            utt_names = self._persons_in_doc(utt_doc)

            for sent in utt_doc.sents:
                sent_nlp = self.nlp(sent.text)

                # Skip possessive adjectives — dep=poss means adjectival
                # ("his sword", "her face") not referential
                pronouns_found = [
                    token for token in sent_nlp
                    if token.pos_ == "PRON"
                    and token.text.lower() in ALL_PRONOUNS
                    and token.dep_ != "poss"
                ]

                if not pronouns_found:
                    continue

                sentence_text = sent.text

                for pronoun_token in pronouns_found:
                    pron = pronoun_token.text.lower()
                    resolved = None
                    source   = None

                    #priority 1: named PERSON in this utterance
                    candidate = self._pick_from_list(utt_names, pron)
                    if candidate:
                        resolved = candidate
                        source   = "intra-utterance"

                    #priority 2: relationship graph from speaker
                    if resolved is None:
                        candidate = self._resolve_via_relationships(
                            speaker, pron, prev_utt_names
                        )
                        if candidate:
                            resolved = candidate
                            source   = "relationship-graph"

                    #priority 3: named PERSON in previous utterance,
                    if resolved is None:
                        candidate = self._pick_from_list(prev_utt_names, pron)
                        if candidate and self._gender_matches(candidate, pron):
                            resolved = candidate
                            source   = "prev-utterance"

                    if resolved is None:
                        continue

                    resolved_text = re.sub(
                        r'\b' + re.escape(pronoun_token.text) + r'\b',
                        resolved,
                        sentence_text,
                        count=1,
                        flags=re.IGNORECASE
                    )

                    print(
                        f"[{source}] '{sent.text.strip()}'\n"
                        f"        -> '{resolved_text.strip()}'\n"
                        f"           ('{pronoun_token.text}' -> '{resolved}')\n"
                    )

                    records.append({
                        "play":              play_title,
                        "act":              act_id,
                        "scene":             scene_id,
                        "speaker":           speaker,
                        "original_sentence": sent.text.strip()[:120],
                        "pronoun":           pronoun_token.text,
                        "resolved_to":       resolved,
                        "resolved_sentence": resolved_text.strip()[:120],
                        "gender_group":      self._gender_of(resolved),
                        "resolution_source": source,
                    })

                    sentence_text = resolved_text

            prev_utt_names = utt_names

        return records


def save_csv(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COREF_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f" -> Saved {len(records):,} records  :  {path.name}")


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
        relationships = config.get("relationships", [])

        if not config:
            print(f"  WARNING: No config in play_configs.py for {play_file.name}")

        resolver     = CoreferenceResolver(nlp, male_chars, female_chars, relationships)
        play_records = []

        for act_id, scene_id, location, text in scenes:
            records = resolver.resolve_scene(text, act_id, scene_id, title)
            play_records.extend(records)

        print(f"  Title             : {title}")
        print(f"  Scenes            : {len(scenes)}")
        print(f"  Pronouns resolved : {len(play_records):,}")

        save_csv(play_records, Globals.OUTPUT_DIR / f"03_{play_file.stem}_coreference.csv")
        all_coref_records.extend(play_records)

    save_csv(all_coref_records, Globals.OUTPUT_DIR / "03_ALL_coreference.csv")
    return all_coref_records