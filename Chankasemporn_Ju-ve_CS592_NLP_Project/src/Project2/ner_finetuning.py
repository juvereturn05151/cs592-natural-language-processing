from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import spacy
from spacy.training.example import Example

# --------------------------------------------------
# Manual curated TRAIN_DATA only
# --------------------------------------------------
TRAIN_DATA = [
    ("Macbeth spoke with Banquo in Scotland.", {
        "entities": [(0, 7, "PERSON"), (19, 25, "PERSON"), (29, 37, "GPE")]
    }),
    ("Lady Macbeth entered the castle.", {
        "entities": [(0, 13, "PERSON")]
    }),
    ("Macduff came from Fife.", {
        "entities": [(0, 7, "PERSON"), (18, 22, "GPE")]
    }),
    ("The army marched toward Birnam.", {
        "entities": [(24, 30, "GPE")]
    }),
    ("Romeo loved Juliet in Verona.", {
        "entities": [(0, 5, "PERSON"), (12, 18, "PERSON"), (22, 28, "GPE")]
    }),
    ("Paris wished to marry Juliet.", {
        "entities": [(0, 5, "PERSON"), (21, 27, "PERSON")]
    }),
    ("Claudio and Hero arrived in Messina.", {
        "entities": [(0, 7, "PERSON"), (12, 16, "PERSON"), (28, 35, "GPE")]
    }),
    ("Beatrice argued with Benedick.", {
        "entities": [(0, 8, "PERSON"), (21, 29, "PERSON")]
    }),
    ("Helena followed Demetrius to Athens.", {
        "entities": [(0, 6, "PERSON"), (16, 26, "PERSON"), (30, 36, "GPE")]
    }),
    ("Theseus spoke with Titania and Oberon.", {
        "entities": [(0, 7, "PERSON"), (19, 26, "PERSON"), (31, 37, "PERSON")]
    }),
    ("The Messenger warned Macbeth.", {
        "entities": [(4, 13, "ROLE"), (21, 28, "PERSON")]
    }),
    ("A Servant opened the gate for Duncan.", {
        "entities": [(2, 9, "ROLE"), (31, 37, "PERSON")]
    }),
    ("An Attendant followed the King.", {
        "entities": [(3, 13, "ROLE")]
    }),
    ("The Gentlewoman waited nearby.", {
        "entities": [(4, 15, "ROLE")]
    }),
    ("Friar Francis helped Hero.", {
        "entities": [(0, 13, "TITLE_PERSON"), (21, 25, "PERSON")]
    }),
    ("King Duncan thanked Macbeth.", {
        "entities": [(0, 12, "TITLE_PERSON"), (21, 28, "PERSON")]
    }),
    ("Prince Malcolm returned to Scotland.", {
        "entities": [(0, 14, "TITLE_PERSON"), (27, 35, "GPE")]
    }),
    ("First Witch greeted Macbeth.", {
        "entities": [(0, 11, "ROLE"), (20, 27, "PERSON")]
    }),
    ("Second Witch spoke to Banquo.", {
        "entities": [(0, 12, "ROLE"), (22, 28, "PERSON")]
    }),
    ("Third Witch vanished suddenly.", {
        "entities": [(0, 11, "ROLE")]
    }),
]

# --------------------------------------------------
# Expected entities / step 2 / step 3
# --------------------------------------------------
def build_expected_entities() -> Dict[str, str]:
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
        "Young Siward": "PERSON",
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
        "Nurse": "ROLE",
        "Friar": "ROLE",
        "Friar Lawrence": "TITLE_PERSON",
        "Friar Laurence": "TITLE_PERSON",
        "Verona": "GPE",
        "Mantua": "GPE",
        "Peter": "PERSON",
        "Balthasar": "PERSON",
        "Rosaline": "PERSON",
        "Benedick": "PERSON",
        "Beatrice": "PERSON",
        "Don Pedro": "TITLE_PERSON",
        "Don John": "TITLE_PERSON",
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
        "Friar Francis": "TITLE_PERSON",
        "Messina": "GPE",
        "Dogberry": "PERSON",
        "Verges": "PERSON",
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
        "Pyramus": "PERSON",
        "Athens": "GPE",
        "Cupid": "PERSON",
        "Messenger": "ROLE",
        "Servant": "ROLE",
        "Attendant": "ROLE",
        "Attendants": "ROLE",
        "Gentlewoman": "ROLE",
        "Musician": "ROLE",
        "First Witch": "ROLE",
        "Second Witch": "ROLE",
        "Third Witch": "ROLE",
        "Witch": "ROLE",
        "Witches": "ROLE",
        "Murderer": "ROLE",
        "Murderers": "ROLE",
        "First Murderer": "ROLE",
        "Second Murderer": "ROLE",
        "Third Murderer": "ROLE",
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
        if intended in {"ROLE", "TITLE_PERSON"}:
            rows.append({
                "entity": entity,
                "predicted_label": row["label"],
                "intended_label": intended,
                "count": int(row["count"]),
                "reason": "Default spaCy labels do not capture theatrical role/title meaning well.",
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
    "macbeth", "banquo", "duncan", "malcolm", "donalbain", "macduff",
    "ross", "lennox", "siward", "romeo", "claudio", "benedick",
    "leonato", "antonio", "theseus", "oberon", "lysander", "demetrius",
    "paris", "mercutio", "benvolio", "tybalt", "dogberry", "verges",
    "puck", "pyramus", "peter", "balthasar",
}
FEMALE_NAMES = {
    "lady macbeth", "juliet", "hero", "beatrice", "helena", "hermia",
    "titania", "hippolyta", "ophelia", "gertrude", "margaret", "ursula", "nurse",
}
NEUTRAL_ROLE_NAMES = {
    "messenger", "servant", "attendant", "attendants", "gentlewoman",
    "musician", "witch", "witness", "murderer", "murderers",
}


def classify_entity_gender_or_number(ent_text: str, ent_label: str) -> str:
    t = ent_text.strip().casefold()
    if t in MALE_NAMES:
        return "male"
    if t in FEMALE_NAMES:
        return "female"
    if t in NEUTRAL_ROLE_NAMES:
        return "neutral"
    if ent_label == "GPE":
        return "neutral"
    return "unknown"


def resolve_pronouns_custom(doc, memory_window_sentences: int = 5) -> pd.DataFrame:
    resolved = []
    entity_memory = []

    singular_pronouns_male = {"he", "him", "his"}
    singular_pronouns_female = {"she", "her", "hers"}
    plural_pronouns = {"they", "them", "their", "theirs"}
    location_pronouns = {"there"}

    for sent_index, sent in enumerate(doc.sents):
        sent_entities = [ent for ent in sent.ents if ent.label_ in {"PERSON", "ROLE", "TITLE_PERSON", "GPE"}]
        for ent in sent_entities:
            entity_memory.append({
                "text": ent.text,
                "label": ent.label_,
                "class": classify_entity_gender_or_number(ent.text, ent.label_),
                "sent_index": sent_index,
            })

        entity_memory = [m for m in entity_memory if sent_index - m["sent_index"] < memory_window_sentences]

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
                    if ent["label"] in {"PERSON", "ROLE", "TITLE_PERSON"} and ent["text"] not in recent_people:
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
def export_final_entity_table(text: str, model_path: str = "shakespeare_ner_model", output_csv: str = "final_entities.csv") -> pd.DataFrame:
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

# --------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------
def run_pipeline(
    baseline_csv: str,
    output_dir: str,
    base_model: str = "en_core_web_md",
    n_iter: int = 20,
):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    expected_entities = build_expected_entities()
    freq_df = load_entity_frequency_csv(baseline_csv)

    mislabeled_df = find_mislabeled_entities(freq_df, expected_entities)
    missing_df = find_missing_entities(freq_df, expected_entities)
    no_default_df = find_no_good_default_label_entities(freq_df, expected_entities)

    save_dataframe(mislabeled_df, str(output_dir_path / "step2_mislabeled_entities.csv"))
    save_dataframe(missing_df, str(output_dir_path / "step3_missing_entities.csv"))
    save_dataframe(no_default_df, str(output_dir_path / "step3_no_good_default_label_entities.csv"))

    model_dir = str(output_dir_path / "shakespeare_ner_model")
    fine_tune_shakespeare_ner(
        train_data=TRAIN_DATA,
        output_dir=model_dir,
        base_model=base_model,
        n_iter=n_iter,
        entity_labels=expected_entities,
    )

    print("Pipeline complete.")
    print(f"Artifacts saved under: {output_dir_path.resolve()}")
    return {
        "mislabeled_df": mislabeled_df,
        "missing_df": missing_df,
        "no_default_df": no_default_df,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Shakespeare NER and export report tables.")
    parser.add_argument("--baseline_csv", type=str, default="shakespeare_entities.csv", help="CSV containing baseline entity, label, count.")
    parser.add_argument("--output_dir", type=str, default="ner_outputs", help="Directory to save outputs.")
    parser.add_argument("--base_model", type=str, default="en_core_web_md", help="spaCy base model to fine-tune.")
    parser.add_argument("--n_iter", type=int, default=20, help="Number of fine-tuning iterations.")
    return parser.parse_args()
