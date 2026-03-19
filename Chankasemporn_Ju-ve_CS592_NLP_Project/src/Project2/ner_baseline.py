from pathlib import Path
import spacy
import pandas as pd
import xml.etree.ElementTree as ET
import html
import re

def load_shakespeare_files(data_dir="data/train"):
    """
    Load only Shakespeare .txt files from the given directory.
    """
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
    """
    Extract only the text inside the <Body> tag from the XML-like Shakespeare file.
    This avoids reading Title/Author metadata such as 'William Shakespeare'.
    """
    raw_text = path.read_text(encoding="utf-8", errors="ignore")

    # Decode HTML entities like &#8217;
    raw_text = html.unescape(raw_text)

    try:
        root = ET.fromstring(raw_text)
        body = root.find("Body")

        if body is None:
            print(f"Warning: No <Body> found in {path.name}")
            return ""

        # Collect all text from inside <Body>
        body_text = " ".join(body.itertext())

        # Clean extra whitespace
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
            rows.append({
                "file": path.name,
                "entity": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

    df = pd.DataFrame(rows, columns=["file", "entity", "label", "start", "end"])
    return df


def summarize(df):
    if df.empty:
        print("DataFrame is empty.")
        return

    # Show all rows/columns (no truncation)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("\n=== Label Distribution ===")
    print(df["label"].value_counts())

    print("\n=== All Entities ===")
    print(df)