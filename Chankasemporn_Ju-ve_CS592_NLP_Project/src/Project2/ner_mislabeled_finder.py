"""
File Name:    ner_mislabeled_finder.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
import pandas as pd

def load_entity_csv(csv_path="shakespeare_entities.csv"):
    return pd.read_csv(csv_path)

def build_expected_entities():
    return {
        "Macbeth": "PERSON",
        "Lady Macbeth": "PERSON",
        "Banquo": "PERSON",
        "Duncan": "PERSON",
        "Malcolm": "PERSON",
        "Donalbain": "PERSON",
        "Macduff": "PERSON",
        "Ross": "PERSON",
        "Lennox": "PERSON",
        "Siward": "PERSON",
        "Scotland": "GPE",
        "England": "GPE",
        "Fife": "GPE",
        "Forres": "GPE",
        "Scone": "GPE",
        "Birnam": "GPE",
        "Glamis": "GPE",

        "Romeo": "PERSON",
        "Juliet": "PERSON",
        "Capulet": "PERSON",
        "Montague": "PERSON",
        "Paris": "PERSON",
        "Mercutio": "PERSON",
        "Benvolio": "PERSON",
        "Tybalt": "PERSON",
        "Nurse": "PERSON",
        "Friar": "PERSON",
        "Lawrence": "PERSON",
        "Verona": "GPE",
        "Mantua": "GPE",
        "Peter": "PERSON",
        "Balthasar": "PERSON",
        "Rosaline": "PERSON",

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

        "Lysander": "PERSON",
        "Demetrius": "PERSON",
        "Hermia": "PERSON",
        "Helena": "PERSON",
        "Theseus": "PERSON",
        "THESEUS": "PERSON",
        "Oberon": "PERSON",
        "Titania": "PERSON",
        "Puck": "PERSON",
        "Quince": "PERSON",
        "Peter Quince": "PERSON",
        "Pyramus": "PERSON",
        "Athens": "GPE",
        "Cupid": "PERSON",
    }

def find_mislabeled_entities(freq_df, expected_entities):
    rows = []
    for _, row in freq_df.iterrows():
        entity = row["entity"]
        predicted = row["label"]
        count = row["count"]

        if entity in expected_entities:
            expected = expected_entities[entity]
            if predicted != expected:
                rows.append({
                    "entity": entity,
                    "predicted_label": predicted,
                    "expected_label": expected,
                    "count": count
                })

    return pd.DataFrame(rows).sort_values(by="count", ascending=False)

def find_missing_entities(freq_df, expected_entities):
    found = set(freq_df["entity"].tolist())
    rows = []

    for entity, expected_label in expected_entities.items():
        if entity not in found:
            rows.append({
                "entity": entity,
                "expected_label": expected_label,
                "issue": "Missing from spaCy output"
            })

    return pd.DataFrame(rows)

def find_no_good_default_label_entities(freq_df):
    role_terms = {
        "MESSENGER", "Messenger",
        "GENTLEWOMAN",
        "MUSICIAN",
        "Attendants",
        "Exit Servant",
        "Witch", "Witches",
        "MURDERER", "BOTH MURDERERS", "Exit Murderer",
        "FIRST", "First", "second", "Second"
    }

    rows = []
    for _, row in freq_df.iterrows():
        entity = row["entity"]
        if entity in role_terms:
            rows.append({
                "entity": entity,
                "label": row["label"],
                "count": row["count"],
                "reason": "Role/stage-direction entity; spaCy default labels do not describe its intended meaning well"
            })

    return pd.DataFrame(rows).sort_values(by="count", ascending=False)