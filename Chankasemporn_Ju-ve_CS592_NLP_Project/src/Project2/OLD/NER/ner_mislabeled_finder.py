"""
File Name:    ner_mislabeled_finder.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
import re
import pandas as pd


def load_entity_csv(csv_path="shakespeare_entities.csv"):
    return pd.read_csv(csv_path)


def normalize_entity(entity: str) -> str:
    return re.sub(r"\s+", " ", str(entity).strip()).casefold()


def build_expected_entities():
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
        "Messenger": "OCC",
        "Servant": "OCC",
        "Porter": "OCC",
        "Doctor": "OCC",
        "Gentlewoman": "OCC",
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
        "Nurse": "OCC",
        "Friar": "OCC",
        "Friar Lawrence": "PERSON",
        "Friar Laurence": "PERSON",
        "Prince Escalus": "PERSON",
        "Verona": "GPE",
        "Mantua": "GPE",
        "Peter": "PERSON",
        "Balthasar": "PERSON",
        "Rosaline": "PERSON",
        "Apothecary": "OCC",

        # =========================
        # MUCH ADO ABOUT NOTHING
        # =========================
        "Benedick": "PERSON",
        "BENEDICK": "PERSON",
        "Beatrice": "PERSON",
        "Don Pedro": "PERSON",
        "Don John": "PERSON",
        "Pedro": "PERSON",
        "DON": "PERSON",
        "JOHN": "PERSON",
        "Claudio": "PERSON",
        "CLAUDIO": "PERSON",
        "Hero": "PERSON",
        "HERO": "PERSON",
        "Leonato": "PERSON",
        "Antonio": "PERSON",
        "Margaret": "PERSON",
        "MARGARET": "PERSON",
        "Conrade": "PERSON",
        "Borachio": "PERSON",
        "Ursula": "PERSON",
        "Friar Francis": "PERSON",
        "Messina": "GPE",
        "Dogberry": "PERSON",
        "Verges": "PERSON",
        "Watch": "OCC",
        "Sexton": "OCC",

        # =========================
        # A MIDSUMMER NIGHT'S DREAM
        # =========================
        "Lysander": "PERSON",
        "Demetrius": "PERSON",
        "Hermia": "PERSON",
        "Helena": "PERSON",
        "Theseus": "PERSON",
        "THESEUS": "PERSON",
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
        "Athens": "GPE",
        "Cupid": "PERSON",
        "Mustardseed": "PERSON",
        "Cobweb": "PERSON",
        "Peaseblossom": "PERSON",
        "Moth": "PERSON",

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


def find_mislabeled_entities(freq_df, expected_entities):
    expected_norm = {normalize_entity(k): v for k, v in expected_entities.items()}
    rows = []

    for _, row in freq_df.iterrows():
        entity = row["entity"]
        predicted = row["label"]
        count = row["count"]

        key = normalize_entity(entity)
        if key in expected_norm:
            expected = expected_norm[key]
            if predicted != expected:
                rows.append({
                    "entity": entity,
                    "predicted_label": predicted,
                    "expected_label": expected,
                    "count": count
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(by=["count", "entity"], ascending=[False, True]).reset_index(drop=True)


def find_missing_entities(freq_df, expected_entities):
    found = {normalize_entity(x) for x in freq_df["entity"].tolist()}
    rows = []

    for entity, expected_label in expected_entities.items():
        if normalize_entity(entity) not in found:
            rows.append({
                "entity": entity,
                "expected_label": expected_label,
                "issue": "Missing from spaCy output"
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(by=["expected_label", "entity"]).reset_index(drop=True)


def find_no_good_default_label_entities(freq_df, expected_entities=None):
    """
    Finds entities whose intended labels are custom labels that spaCy's default
    NER schema does not represent well.
    """
    if expected_entities is None:
        expected_entities = build_expected_entities()

    expected_norm = {normalize_entity(k): v for k, v in expected_entities.items()}
    unsupported_custom_labels = {"OCC", "REL"}

    rows = []
    for _, row in freq_df.iterrows():
        entity = row["entity"]
        key = normalize_entity(entity)

        if key in expected_norm and expected_norm[key] in unsupported_custom_labels:
            rows.append({
                "entity": entity,
                "predicted_label": row["label"],
                "intended_label": expected_norm[key],
                "count": row["count"],
                "reason": "Default spaCy labels do not represent occupation/title/relationship meaning well."
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(by=["count", "entity"], ascending=[False, True]).reset_index(drop=True)