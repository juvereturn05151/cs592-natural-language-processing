from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Tuple
import html
import xml.etree.ElementTree as ET

import pandas as pd
import spacy
from spacy.training.example import Example


def split_text_into_sentences(text: str, model: str = "en_core_web_md") -> List[str]:
    nlp = spacy.load(model, disable=["ner"])
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        s = re.sub(r"\s+", " ", sent.text).strip()
        if s:
            sentences.append(s)
    return sentences


def find_all_non_overlapping_spans(text: str, substring: str):
    matches = []
    for m in re.finditer(re.escape(substring), text):
        matches.append((m.start(), m.end()))
    return matches


def make_example_from_spans(text: str, spans: List[Tuple[int, int, str]]):
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    return (text, {"entities": spans})


def sentence_has_stage_direction_noise(sentence: str) -> bool:
    lowered = sentence.casefold()
    noisy_patterns = [
        "enter ",
        "exit ",
        "exeunt",
        " flourish",
        "music",
        "within",
        "aside",
        " alarum",
        "drum",
        "trumpet",
        "hautboys",
        "sennet",
    ]
    return any(p in lowered for p in noisy_patterns)


def mine_training_data_from_corpus(
    corpus_text: str,
    expected_entities: Dict[str, str],
    max_positive: int = 200,
    max_negative: int = 100,
    min_sentence_len: int = 15,
    max_sentence_len: int = 220,
    base_model: str = "en_core_web_md",
    seed: int = 42,
):
    """
    Semi-automatic mining:
    - Positive examples: sentences containing high-confidence exact matches from expected_entities
    - Negative examples: sentences with no matched expected entities
    """
    random.seed(seed)

    sentences = split_text_into_sentences(corpus_text, model=base_model)

    # Sort longer entities first so "Lady Macbeth" is preferred over "Macbeth"
    entity_items = sorted(
        expected_entities.items(),
        key=lambda kv: len(kv[0]),
        reverse=True
    )

    positive_examples = []
    negative_examples = []

    for sentence in sentences:
        sent = re.sub(r"\s+", " ", sentence).strip()

        if len(sent) < min_sentence_len or len(sent) > max_sentence_len:
            continue

        spans = []
        used_ranges = []

        for entity_text, label in entity_items:
            for start, end in find_all_non_overlapping_spans(sent, entity_text):
                # word boundary safety
                left_ok = start == 0 or not sent[start - 1].isalnum()
                right_ok = end == len(sent) or not sent[end].isalnum()
                if not (left_ok and right_ok):
                    continue

                overlap = False
                for s, e, _ in used_ranges:
                    if not (end <= s or start >= e):
                        overlap = True
                        break

                if not overlap:
                    spans.append((start, end, label))
                    used_ranges.append((start, end, label))

        spans = sorted(spans, key=lambda x: (x[0], x[1]))

        if spans:
            positive_examples.append(make_example_from_spans(sent, spans))
        else:
            # keep some clean negatives
            if not sentence_has_stage_direction_noise(sent):
                negative_examples.append(make_example_from_spans(sent, []))

    # Deduplicate by text
    def dedupe_examples(examples):
        seen = set()
        out = []
        for text, ann in examples:
            key = (text, tuple(ann["entities"]))
            if key not in seen:
                seen.add(key)
                out.append((text, ann))
        return out

    positive_examples = dedupe_examples(positive_examples)
    negative_examples = dedupe_examples(negative_examples)

    random.shuffle(positive_examples)
    random.shuffle(negative_examples)

    positive_examples = positive_examples[:max_positive]
    negative_examples = negative_examples[:max_negative]

    return {
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "all_examples": positive_examples + negative_examples,
    }


def load_shakespeare_test_text(data_dir="../../data/train"):
    data_path = Path(data_dir)
    texts = []

    files = [
        path for path in data_path.rglob("*.txt")
        if "shakespeare" in path.name.lower()
    ]

    for path in files:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        raw_text = html.unescape(raw_text)

        try:
            root = ET.fromstring(raw_text)
            body = root.find("Body")
            if body is not None:
                body_text = " ".join(body.itertext())
                body_text = re.sub(r"\s+", " ", body_text).strip()
                texts.append(body_text)
        except ET.ParseError:
            continue

    return "\n".join(texts)


# --------------------------------------------------
# Manual curated TRAIN_DATA only
# --------------------------------------------------
def make_example(text, entities):
    """
    entities: list of (substring, label)
    returns: (text, {"entities": [(start, end, label), ...]})
    """
    spans = []
    used = []

    for substring, label in entities:
        search_start = 0
        found = False

        while True:
            start = text.find(substring, search_start)
            if start == -1:
                break

            end = start + len(substring)

            # accept only if it does not overlap an existing span
            overlaps = False
            for s, e, _ in used:
                if not (end <= s or start >= e):
                    overlaps = True
                    break

            if not overlaps:
                spans.append((start, end, label))
                used.append((start, end, label))
                found = True
                break

            search_start = start + 1

        if not found:
            raise ValueError(f"Substring '{substring}' not found without overlap in: {text}")

    return (text, {"entities": spans})


TRAIN_DATA = [
    # =========================
    # MACBETH
    # =========================
    make_example("Macbeth spoke with Banquo in Scotland.", [
        ("Macbeth", "PERSON"),
        ("Banquo", "PERSON"),
        ("Scotland", "GPE"),
    ]),
    make_example("Lady Macbeth read the letter at Inverness.", [
        ("Lady Macbeth", "PERSON"),
        ("Inverness", "GPE"),
    ]),
    make_example("Macduff traveled from Fife to England.", [
        ("Macduff", "PERSON"),
        ("Fife", "GPE"),
        ("England", "GPE"),
    ]),
    make_example("King Duncan praised Macbeth at Forres.", [
        ("King Duncan", "PERSON"),
        ("Macbeth", "PERSON"),
        ("Forres", "GPE"),
    ]),
    make_example("Prince Malcolm returned to Scotland with Macduff.", [
        ("Prince Malcolm", "PERSON"),
        ("Scotland", "GPE"),
        ("Macduff", "PERSON"),
    ]),
    make_example("Banquo and Fleance rode toward Forres.", [
        ("Banquo", "PERSON"),
        ("Fleance", "PERSON"),
        ("Forres", "GPE"),
    ]),
    make_example("The Messenger warned Macbeth of approaching soldiers.", [
        ("Messenger", "OCC"),
        ("Macbeth", "PERSON"),
    ]),
    make_example("A Servant opened the gate for Duncan.", [
        ("Servant", "OCC"),
        ("Duncan", "PERSON"),
    ]),
    make_example("The Porter laughed at Inverness.", [
        ("Porter", "OCC"),
        ("Inverness", "GPE"),
    ]),
    make_example("The Doctor treated Lady Macbeth.", [
        ("Doctor", "OCC"),
        ("Lady Macbeth", "PERSON"),
    ]),
    make_example("The Gentlewoman watched Lady Macbeth sleepwalk.", [
        ("Gentlewoman", "OCC"),
        ("Lady Macbeth", "PERSON"),
    ]),
    make_example("First Witch greeted Macbeth on the heath.", [
        ("First Witch", "OCC"),
        ("Macbeth", "PERSON"),
    ]),
    make_example("Second Witch spoke to Banquo.", [
        ("Second Witch", "OCC"),
        ("Banquo", "PERSON"),
    ]),
    make_example("Third Witch vanished before Macbeth arrived.", [
        ("Third Witch", "OCC"),
        ("Macbeth", "PERSON"),
    ]),
    make_example("Ross met Macduff in Scotland.", [
        ("Ross", "PERSON"),
        ("Macduff", "PERSON"),
        ("Scotland", "GPE"),
    ]),
    make_example("Lady Macduff stayed in Fife with her son.", [
        ("Lady Macduff", "PERSON"),
        ("Fife", "GPE"),
        ("son", "REL"),
    ]),
    make_example("Lady Macbeth is the wife of Macbeth.", [
        ("Lady Macbeth", "PERSON"),
        ("wife", "REL"),
        ("Macbeth", "PERSON"),
    ]),
    make_example("Banquo is the father of Fleance.", [
        ("Banquo", "PERSON"),
        ("father", "REL"),
        ("Fleance", "PERSON"),
    ]),
    make_example("Macduff is the husband of Lady Macduff.", [
        ("Macduff", "PERSON"),
        ("husband", "REL"),
        ("Lady Macduff", "PERSON"),
    ]),
    make_example("Malcolm is the son of Duncan.", [
        ("Malcolm", "PERSON"),
        ("son", "REL"),
        ("Duncan", "PERSON"),
    ]),

    # =========================
    # ROMEO AND JULIET
    # =========================
    make_example("Romeo loved Juliet in Verona.", [
        ("Romeo", "PERSON"),
        ("Juliet", "PERSON"),
        ("Verona", "GPE"),
    ]),
    make_example("Paris wished to marry Juliet.", [
        ("Paris", "PERSON"),
        ("Juliet", "PERSON"),
    ]),
    make_example("Mercutio joked with Romeo in Verona.", [
        ("Mercutio", "PERSON"),
        ("Romeo", "PERSON"),
        ("Verona", "GPE"),
    ]),
    make_example("Tybalt challenged Romeo before the Capulet house.", [
        ("Tybalt", "PERSON"),
        ("Romeo", "PERSON"),
        ("Capulet", "PERSON"),
    ]),
    make_example("Friar Laurence helped Romeo escape.", [
        ("Friar Laurence", "PERSON"),
        ("Romeo", "PERSON"),
    ]),
    make_example("Nurse comforted Juliet in Verona.", [
        ("Nurse", "OCC"),
        ("Juliet", "PERSON"),
        ("Verona", "GPE"),
    ]),
    make_example("Benvolio searched for Romeo in Verona.", [
        ("Benvolio", "PERSON"),
        ("Romeo", "PERSON"),
        ("Verona", "GPE"),
    ]),
    make_example("Lord Capulet arranged a feast in Verona.", [
        ("Lord Capulet", "PERSON"),
        ("Verona", "GPE"),
    ]),
    make_example("Lady Capulet spoke with Juliet.", [
        ("Lady Capulet", "PERSON"),
        ("Juliet", "PERSON"),
    ]),
    make_example("Prince Escalus judged the families in Verona.", [
        ("Prince Escalus", "PERSON"),
        ("Verona", "GPE"),
    ]),
    make_example("The Apothecary sold poison to Romeo.", [
        ("Apothecary", "OCC"),
        ("Romeo", "PERSON"),
    ]),
    make_example("Balthasar brought news to Romeo.", [
        ("Balthasar", "PERSON"),
        ("Romeo", "PERSON"),
    ]),
    make_example("Juliet trusted Friar Laurence.", [
        ("Juliet", "PERSON"),
        ("Friar Laurence", "PERSON"),
    ]),
    make_example("Sampson argued in Verona.", [
        ("Sampson", "PERSON"),
        ("Verona", "GPE"),
    ]),
    make_example("Gregory served the Capulet household in Verona.", [
        ("Gregory", "PERSON"),
        ("Capulet", "PERSON"),
        ("Verona", "GPE"),
    ]),
    make_example("Rosaline was admired by Romeo.", [
        ("Rosaline", "PERSON"),
        ("Romeo", "PERSON"),
    ]),
    make_example("Romeo is the son of Montague.", [
        ("Romeo", "PERSON"),
        ("son", "REL"),
        ("Montague", "PERSON"),
    ]),
    make_example("Juliet is the daughter of Capulet.", [
        ("Juliet", "PERSON"),
        ("daughter", "REL"),
        ("Capulet", "PERSON"),
    ]),
    make_example("Romeo is the lover of Juliet.", [
        ("Romeo", "PERSON"),
        ("lover", "REL"),
        ("Juliet", "PERSON"),
    ]),
    make_example("Mercutio is the friend of Romeo.", [
        ("Mercutio", "PERSON"),
        ("friend", "REL"),
        ("Romeo", "PERSON"),
    ]),
    make_example("Capulet and Montague are enemies in Verona.", [
        ("Capulet", "PERSON"),
        ("Montague", "PERSON"),
        ("enemies", "REL"),
        ("Verona", "GPE"),
    ]),

    # =========================
    # A MIDSUMMER NIGHT'S DREAM
    # =========================
    make_example("Hermia loved Lysander in Athens.", [
        ("Hermia", "PERSON"),
        ("Lysander", "PERSON"),
        ("Athens", "GPE"),
    ]),
    make_example("Demetrius pursued Helena in Athens.", [
        ("Demetrius", "PERSON"),
        ("Helena", "PERSON"),
        ("Athens", "GPE"),
    ]),
    make_example("Theseus ruled Athens with Hippolyta.", [
        ("Theseus", "PERSON"),
        ("Athens", "GPE"),
        ("Hippolyta", "PERSON"),
    ]),
    make_example("Oberon argued with Titania in the forest.", [
        ("Oberon", "PERSON"),
        ("Titania", "PERSON"),
    ]),
    make_example("Puck served Oberon faithfully.", [
        ("Puck", "PERSON"),
        ("Oberon", "PERSON"),
    ]),
    make_example("Bottom rehearsed with Quince near Athens.", [
        ("Bottom", "PERSON"),
        ("Quince", "PERSON"),
        ("Athens", "GPE"),
    ]),
    make_example("Flute joined Snout and Snug in the play.", [
        ("Flute", "PERSON"),
        ("Snout", "PERSON"),
        ("Snug", "PERSON"),
    ]),
    make_example("Egeus complained to Theseus in Athens.", [
        ("Egeus", "PERSON"),
        ("Theseus", "PERSON"),
        ("Athens", "GPE"),
    ]),
    make_example("Philostrate prepared the entertainment for Theseus.", [
        ("Philostrate", "PERSON"),
        ("Theseus", "PERSON"),
    ]),
    make_example("Robin Goodfellow misled the lovers in the forest.", [
        ("Robin Goodfellow", "PERSON"),
    ]),
    make_example("Mustardseed attended Titania.", [
        ("Mustardseed", "PERSON"),
        ("Titania", "PERSON"),
    ]),
    make_example("Cobweb followed Titania through the wood.", [
        ("Cobweb", "PERSON"),
        ("Titania", "PERSON"),
    ]),
    make_example("Peaseblossom greeted Bottom politely.", [
        ("Peaseblossom", "PERSON"),
        ("Bottom", "PERSON"),
    ]),
    make_example("Moth served Titania beside Bottom.", [
        ("Moth", "PERSON"),
        ("Titania", "PERSON"),
        ("Bottom", "PERSON"),
    ]),
    make_example("Lysander quarreled with Demetrius in Athens.", [
        ("Lysander", "PERSON"),
        ("Demetrius", "PERSON"),
        ("Athens", "GPE"),
    ]),
    make_example("Helena followed Demetrius into the forest.", [
        ("Helena", "PERSON"),
        ("Demetrius", "PERSON"),
    ]),
    make_example("Hermia is the friend of Helena.", [
        ("Hermia", "PERSON"),
        ("friend", "REL"),
        ("Helena", "PERSON"),
    ]),

    # =========================
    # MUCH ADO ABOUT NOTHING
    # =========================
    make_example("Claudio loved Hero in Messina.", [
        ("Claudio", "PERSON"),
        ("Hero", "PERSON"),
        ("Messina", "GPE"),
    ]),
    make_example("Beatrice argued with Benedick in Messina.", [
        ("Beatrice", "PERSON"),
        ("Benedick", "PERSON"),
        ("Messina", "GPE"),
    ]),
    make_example("Don Pedro spoke with Claudio.", [
        ("Don Pedro", "PERSON"),
        ("Claudio", "PERSON"),
    ]),
    make_example("Don John deceived Claudio in Messina.", [
        ("Don John", "PERSON"),
        ("Claudio", "PERSON"),
        ("Messina", "GPE"),
    ]),
    make_example("Leonato welcomed Don Pedro to Messina.", [
        ("Leonato", "PERSON"),
        ("Don Pedro", "PERSON"),
        ("Messina", "GPE"),
    ]),
    make_example("Friar Francis helped Hero.", [
        ("Friar Francis", "PERSON"),
        ("Hero", "PERSON"),
    ]),
    make_example("Ursula spoke with Hero in the garden.", [
        ("Ursula", "PERSON"),
        ("Hero", "PERSON"),
    ]),
    make_example("Margaret laughed with Beatrice.", [
        ("Margaret", "PERSON"),
        ("Beatrice", "PERSON"),
    ]),
    make_example("Borachio confessed the scheme to Conrade.", [
        ("Borachio", "PERSON"),
        ("Conrade", "PERSON"),
    ]),
    make_example("Dogberry questioned Borachio in Messina.", [
        ("Dogberry", "PERSON"),
        ("Borachio", "PERSON"),
        ("Messina", "GPE"),
    ]),
    make_example("Verges assisted Dogberry.", [
        ("Verges", "PERSON"),
        ("Dogberry", "PERSON"),
    ]),
    make_example("The Watch arrested Borachio.", [
        ("Watch", "OCC"),
        ("Borachio", "PERSON"),
    ]),
    make_example("A Sexton recorded the testimony.", [
        ("Sexton", "OCC"),
    ]),
    make_example("Antonio supported Leonato in Messina.", [
        ("Antonio", "PERSON"),
        ("Leonato", "PERSON"),
        ("Messina", "GPE"),
    ]),
    make_example("Benedick challenged Claudio after the wedding.", [
        ("Benedick", "PERSON"),
        ("Claudio", "PERSON"),
    ]),
    make_example("Hero fainted before Leonato and Beatrice.", [
        ("Hero", "PERSON"),
        ("Leonato", "PERSON"),
        ("Beatrice", "PERSON"),
    ]),
    make_example("Hero is the daughter of Leonato.", [
        ("Hero", "PERSON"),
        ("daughter", "REL"),
        ("Leonato", "PERSON"),
    ]),
    make_example("Beatrice is the niece of Leonato.", [
        ("Beatrice", "PERSON"),
        ("niece", "REL"),
        ("Leonato", "PERSON"),
    ]),
    make_example("Benedick is the friend of Claudio.", [
        ("Benedick", "PERSON"),
        ("friend", "REL"),
        ("Claudio", "PERSON"),
    ]),

    # =========================
    # GENERAL / CROSS-PLAY / NEGATIVES
    # =========================
    make_example("Exit Macbeth.", [
        ("Macbeth", "PERSON"),
    ]),
    make_example("Exeunt all but Benedick.", [
        ("Benedick", "PERSON"),
    ]),
    make_example("Enter Romeo and Juliet.", [
        ("Romeo", "PERSON"),
        ("Juliet", "PERSON"),
    ]),
    make_example("Exit pursued by a bear.", []),
    make_example("Exeunt all but one.", []),
    make_example("Come hither and listen well.", []),
    make_example("Thou shalt not pass unnoticed.", []),
    make_example("Yea, I shall go with thee.", []),
    make_example("Farewell, good sir, until tomorrow.", []),
    make_example("The moon shone brightly above the trees.", []),
    make_example("Music sounded within.", []),
    make_example("A bell rang in the distance.", []),
]


# --------------------------------------------------
# Expected entities / step 2 / step 3
# --------------------------------------------------
def build_expected_entities() -> Dict[str, str]:
    return {
        # =========================
        # MACBETH
        # =========================
        "Macbeth": "PERSON",
        "Lady Macbeth": "PERSON",
        "Banquo": "PERSON",
        "Duncan": "PERSON",
        "King Duncan": "PERSON",
        "Malcolm": "PERSON",
        "Prince Malcolm": "PERSON",
        "Donalbain": "PERSON",
        "Macduff": "PERSON",
        "Lady Macduff": "PERSON",
        "Ross": "PERSON",
        "Lennox": "PERSON",
        "Siward": "PERSON",
        "Young Siward": "PERSON",
        "Fleance": "PERSON",
        "Scotland": "GPE",
        "England": "GPE",
        "Fife": "GPE",
        "Forres": "GPE",
        "Scone": "GPE",
        "Birnam": "GPE",
        "Glamis": "GPE",
        "Inverness": "GPE",

        # =========================
        # ROMEO AND JULIET
        # =========================
        "Romeo": "PERSON",
        "Juliet": "PERSON",
        "Capulet": "PERSON",
        "Lord Capulet": "PERSON",
        "Lady Capulet": "PERSON",
        "Montague": "PERSON",
        "Paris": "PERSON",
        "Mercutio": "PERSON",
        "Benvolio": "PERSON",
        "Tybalt": "PERSON",
        "Friar Lawrence": "PERSON",
        "Friar Laurence": "PERSON",
        "Prince Escalus": "PERSON",
        "Verona": "GPE",
        "Mantua": "GPE",
        "Peter": "PERSON",
        "Balthasar": "PERSON",
        "Rosaline": "PERSON",
        "Sampson": "PERSON",
        "Gregory": "PERSON",

        # =========================
        # MUCH ADO ABOUT NOTHING
        # =========================
        "Benedick": "PERSON",
        "Beatrice": "PERSON",
        "Don Pedro": "PERSON",
        "Don John": "PERSON",
        "Pedro": "PERSON",
        "John": "PERSON",
        "Claudio": "PERSON",
        "Hero": "PERSON",
        "Leonato": "PERSON",
        "Antonio": "PERSON",
        "Margaret": "PERSON",
        "Conrade": "PERSON",
        "Borachio": "PERSON",
        "Ursula": "PERSON",
        "Friar Francis": "PERSON",
        "Messina": "GPE",
        "Dogberry": "PERSON",
        "Verges": "PERSON",

        # =========================
        # A MIDSUMMER NIGHT'S DREAM
        # =========================
        "Lysander": "PERSON",
        "Demetrius": "PERSON",
        "Hermia": "PERSON",
        "Helena": "PERSON",
        "Theseus": "PERSON",
        "Hippolyta": "PERSON",
        "Oberon": "PERSON",
        "Titania": "PERSON",
        "Puck": "PERSON",
        "Robin Goodfellow": "PERSON",
        "Quince": "PERSON",
        "Peter Quince": "PERSON",
        "Bottom": "PERSON",
        "Flute": "PERSON",
        "Snout": "PERSON",
        "Snug": "PERSON",
        "Philostrate": "PERSON",
        "Egeus": "PERSON",
        "Pyramus": "PERSON",
        "Mustardseed": "PERSON",
        "Cobweb": "PERSON",
        "Peaseblossom": "PERSON",
        "Moth": "PERSON",
        "Athens": "GPE",
        "Cupid": "PERSON",

        # =========================
        # OCCUPATIONS / ROLES
        # =========================
        "Friar": "OCC",
        "Nurse": "OCC",
        "Messenger": "OCC",
        "Servant": "OCC",
        "Porter": "OCC",
        "Doctor": "OCC",
        "Apothecary": "OCC",
        "Watch": "OCC",
        "Sexton": "OCC",
        "Attendant": "OCC",
        "Attendants": "OCC",
        "Gentlewoman": "OCC",
        "Musician": "OCC",
        "First Witch": "OCC",
        "Second Witch": "OCC",
        "Third Witch": "OCC",
        "Witch": "OCC",
        "Witches": "OCC",
        "Murderer": "OCC",
        "Murderers": "OCC",
        "First Murderer": "OCC",
        "Second Murderer": "OCC",
        "Third Murderer": "OCC",

        # =========================
        # RELATION WORDS
        # =========================
        "father": "REL",
        "mother": "REL",
        "son": "REL",
        "daughter": "REL",
        "brother": "REL",
        "sister": "REL",
        "husband": "REL",
        "wife": "REL",
        "friend": "REL",
        "lover": "REL",
        "niece": "REL",
        "nephew": "REL",
        "uncle": "REL",
        "aunt": "REL",
        "enemy": "REL",
        "enemies": "REL",
    }


def normalize_entity_for_lookup(entity: str) -> str:
    return re.sub(r"\s+", " ", str(entity).strip()).casefold()


def load_entity_frequency_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"entity", "label", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns {required}, missing {missing}")
    return df


def find_mislabeled_entities(freq_df: pd.DataFrame, expected_entities: Dict[str, str]) -> pd.DataFrame:
    expected_norm = {normalize_entity_for_lookup(k): v for k, v in expected_entities.items()}
    rows = []

    for _, row in freq_df.iterrows():
        entity = row["entity"]
        predicted = row["label"]
        key = normalize_entity_for_lookup(entity)

        if key in expected_norm and predicted != expected_norm[key]:
            rows.append({
                "entity": entity,
                "predicted_label": predicted,
                "expected_label": expected_norm[key],
                "count": int(row["count"]),
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["count", "entity"], ascending=[False, True]).reset_index(drop=True) if not out.empty else out


def find_missing_entities(freq_df: pd.DataFrame, expected_entities: Dict[str, str]) -> pd.DataFrame:
    found = {normalize_entity_for_lookup(x) for x in freq_df["entity"].astype(str).tolist()}
    rows = []

    for entity, expected_label in expected_entities.items():
        if normalize_entity_for_lookup(entity) not in found:
            rows.append({
                "entity": entity,
                "expected_label": expected_label,
                "issue": "Missing from baseline output",
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["expected_label", "entity"]).reset_index(drop=True) if not out.empty else out


def find_no_good_default_label_entities(freq_df: pd.DataFrame, expected_entities: Dict[str, str]) -> pd.DataFrame:
    expected_norm = {normalize_entity_for_lookup(k): v for k, v in expected_entities.items()}
    rows = []

    for _, row in freq_df.iterrows():
        entity = row["entity"]
        intended = expected_norm.get(normalize_entity_for_lookup(entity))

        if intended in {"OCC", "REL"}:
            rows.append({
                "entity": entity,
                "predicted_label": row["label"],
                "intended_label": intended,
                "count": int(row["count"]),
                "reason": "Default spaCy labels do not capture occupation/title/relationship meaning well.",
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["count", "entity"], ascending=[False, True]).reset_index(drop=True) if not out.empty else out


# --------------------------------------------------
# Fine-tuning
# --------------------------------------------------
def add_entity_ruler(nlp, entity_labels: Dict[str, str]):
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")

    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": False})
    patterns = [{"label": label, "pattern": entity} for entity, label in entity_labels.items()]
    ruler.add_patterns(patterns)
    return nlp


def fine_tune_shakespeare_ner(
    train_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]] | None = None,
    output_dir: str = "shakespeare_ner_model",
    base_model: str = "en_core_web_md",
    n_iter: int = 30,
    dropout: float = 0.2,
    entity_labels: Dict[str, str] | None = None,
    seed: int = 42,
):
    random.seed(seed)
    spacy.util.fix_random_seed(seed)

    if train_data is None:
        train_data = TRAIN_DATA
    if entity_labels is None:
        entity_labels = build_expected_entities()

    nlp = spacy.load(base_model)
    nlp = add_entity_ruler(nlp, entity_labels)

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    all_labels = set(entity_labels.values())
    for _, annotations in train_data:
        for _, _, label in annotations["entities"]:
            all_labels.add(label)

    for label in sorted(all_labels):
        ner.add_label(label)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in {"ner", "entity_ruler"}]

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()

        for i in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            examples = []

            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                examples.append(Example.from_dict(doc, annotations))

            nlp.update(examples, sgd=optimizer, drop=dropout, losses=losses)
            print(f"Iteration {i + 1}/{n_iter} - Losses: {losses}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"Saved fine-tuned model to {Path(output_dir).resolve()}")
    return nlp


# --------------------------------------------------
# Custom pronoun / coreference resolution
# --------------------------------------------------
MALE_NAMES = {
    "macbeth", "banquo", "duncan", "king duncan", "malcolm", "prince malcolm",
    "donalbain", "macduff", "ross", "lennox", "siward", "young siward",
    "romeo", "claudio", "benedick", "leonato", "antonio", "theseus",
    "oberon", "lysander", "demetrius", "paris", "mercutio", "benvolio",
    "tybalt", "dogberry", "verges", "puck", "pyramus", "peter", "balthasar",
    "capulet", "montague", "don pedro", "don john", "egeus", "borachio",
    "conrade", "quince", "peter quince", "bottom", "snug", "snout", "flute",
    "sampson", "gregory", "fleance",
}

FEMALE_NAMES = {
    "lady macbeth", "lady macduff", "juliet", "hero", "beatrice", "helena",
    "hermia", "titania", "hippolyta", "ophelia", "gertrude", "margaret",
    "ursula", "rosaline", "mustardseed", "cobweb", "peaseblossom", "moth",
}

NEUTRAL_OCC_NAMES = {
    "messenger", "servant", "attendant", "attendants", "gentlewoman",
    "musician", "witch", "witches", "witness", "murderer", "murderers",
    "first witch", "second witch", "third witch", "first murderer",
    "second murderer", "third murderer", "doctor", "porter", "watch",
    "sexton", "apothecary", "nurse", "friar",
}


def classify_entity_gender_or_number(ent_text: str, ent_label: str) -> str:
    t = ent_text.strip().casefold()

    if t in MALE_NAMES:
        return "male"
    if t in FEMALE_NAMES:
        return "female"
    if t in NEUTRAL_OCC_NAMES:
        return "neutral"
    if ent_label == "GPE":
        return "neutral"
    if ent_label == "REL":
        return "neutral"

    return "unknown"


def resolve_pronouns_custom(doc, memory_window_sentences: int = 5) -> pd.DataFrame:
    resolved = []
    entity_memory = []

    singular_pronouns_male = {"he", "him", "his"}
    singular_pronouns_female = {"she", "her", "hers"}
    plural_pronouns = {"they", "them", "their", "theirs"}
    location_pronouns = {"there"}

    tracked_labels = {"PERSON", "OCC",  "GPE", "REL"}
    person_like_labels = {"PERSON", "OCC"}

    for sent_index, sent in enumerate(doc.sents):
        sent_entities = [ent for ent in sent.ents if ent.label_ in tracked_labels]

        for ent in sent_entities:
            entity_memory.append({
                "text": ent.text,
                "label": ent.label_,
                "class": classify_entity_gender_or_number(ent.text, ent.label_),
                "sent_index": sent_index,
            })

        entity_memory = [
            m for m in entity_memory
            if sent_index - m["sent_index"] < memory_window_sentences
        ]

        for token in sent:
            tok = token.text.lower()
            antecedent = None
            rationale = None
            candidates = list(reversed(entity_memory))

            if tok in singular_pronouns_male:
                for ent in candidates:
                    if ent["class"] == "male":
                        antecedent = ent["text"]
                        rationale = "Most recent male-compatible entity in rolling memory."
                        break

            elif tok in singular_pronouns_female:
                for ent in candidates:
                    if ent["class"] == "female":
                        antecedent = ent["text"]
                        rationale = "Most recent female-compatible entity in rolling memory."
                        break

            elif tok in plural_pronouns:
                recent_people = []
                for ent in candidates:
                    if ent["label"] in person_like_labels and ent["text"] not in recent_people:
                        recent_people.append(ent["text"])
                    if len(recent_people) == 2:
                        break

                if len(recent_people) == 2:
                    antecedent = " and ".join(reversed(recent_people))
                    rationale = "Two most recent person-like entities in rolling memory."

            elif tok in location_pronouns:
                for ent in candidates:
                    if ent["label"] == "GPE":
                        antecedent = ent["text"]
                        rationale = "Most recent location entity in rolling memory."
                        break

            if antecedent:
                resolved.append({
                    "pronoun": token.text,
                    "sentence": sent.text,
                    "resolved_to": antecedent,
                    "rationale": rationale,
                })

    return pd.DataFrame(resolved)


# --------------------------------------------------
# Step 5 final export
# --------------------------------------------------
def export_final_entity_table(
    text: str,
    model_path: str = "shakespeare_ner_model",
    output_csv: str = "final_entities.csv"
) -> pd.DataFrame:
    nlp = spacy.load(model_path)
    doc = nlp(text)

    rows = []
    for ent in doc.ents:
        rows.append({
            "entity": ent.text.strip(),
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No entities found.")
        return df

    final_df = (
        df.groupby(["entity", "label"])
        .size()
        .reset_index(name="frequency")
        .sort_values(by=["label", "entity"])
        .reset_index(drop=True)
    )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved final entity table to {Path(output_csv).resolve()}")
    return final_df


def save_dataframe(df: pd.DataFrame, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved: {Path(output_path).resolve()}")
