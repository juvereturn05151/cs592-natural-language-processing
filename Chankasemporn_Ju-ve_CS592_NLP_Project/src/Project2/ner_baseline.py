from pathlib import Path
import spacy
import pandas as pd
import xml.etree.ElementTree as ET
import html
import re


def load_shakespeare_files(data_dir="data/train"):
    data_path = Path(data_dir)

    print(f"Looking in: {data_path.resolve()}")

    if not data_path.exists():
        print("Directory does not exist.")
        return []

    files = [
        path for path in data_path.rglob("*.txt")
        if "shakespeare" in path.name.lower()
    ]

    print("Matched files:")
    for f in files:
        print(" -", f.name)

    return files


def extract_body_text(path):
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    raw_text = html.unescape(raw_text)

    try:
        root = ET.fromstring(raw_text)
        body = root.find("Body")

        if body is None:
            print(f"Warning: No <Body> found in {path.name}")
            return ""

        body_text = " ".join(body.itertext())
        body_text = re.sub(r"\s+", " ", body_text).strip()
        return body_text

    except ET.ParseError as e:
        print(f"XML parse error in {path.name}: {e}")
        return ""


def run_baseline_ner(data_dir="data/train"):
    nlp = spacy.load("en_core_web_md")

    rows = []
    files = load_shakespeare_files(data_dir)

    print(f"Loaded {len(files)} text files")

    for path in files:
        print(f"Processing: {path.name}")

        text = extract_body_text(path)

        if not text:
            print(f"Skipping {path.name} because no body text was extracted.")
            continue

        doc = nlp(text)

        for ent in doc.ents:
            entity_text = ent.text.strip()

            # Skip filenames like macbeth.txt
            if re.fullmatch(r"[\w\-]+\.(txt|xml|csv|json)", entity_text.lower()):
                continue

            # Skip path-like strings
            if "/" in entity_text or "\\" in entity_text:
                continue

            rows.append({
                "entity": entity_text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

    df = pd.DataFrame(rows, columns=["entity", "label", "start", "end"])
    return df


def summarize(df):
    if df.empty:
        print("DataFrame is empty.")
        return

    pd.set_option("display.max_rows", None)

    print("\n=== Label Distribution ===")
    print(df["label"].value_counts())

    # Count occurrences
    freq_df = (
        df.groupby(["entity", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    print("\n=== Entities with Frequency ===")
    print(freq_df.to_string(index=False))