"""
File Name:    manual_labeling.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import csv
from pathlib import Path

import src.Project2.Project2Globals as Globals

#helper to build correction entries cleanly
def _remove():
    return ("remove", None, None)

def _relabel(label):
    return ("relabel", label, None)

def _rename(text):
    return ("rename", None, text)

def _relabel_rename(label, text):
    return ("relabel+rename", label, text)


CORRECTIONS = {
    # MACBETH
    "MACBETH": {
        ("AROINT",       "GPE"):      _remove(),
        ("SHOUGHS",      "GPE"):      _remove(),
        ("WHATE'ER",     "GPE"):      _remove(),
        ("THIRD",        "GPE"):      _remove(),
        ("RAZE",         "GPE"):      _remove(),
        ("DUNSINANE.—ARM","GPE"):     _rename("DUNSINANE"),
        ("FIRST",               "TITLE"): _remove(),
        ("MAN",                 "TITLE"): _remove(),
        ("MURDERER",            "TITLE"): _remove(),
        ("SERVANT",             "TITLE"): _remove(),
        ("HARK!—WHO",           "TITLE"): _remove(),
        ("SIGHT!—NOW",          "TITLE"): _remove(),
        ("THANE LIVES YET",     "TITLE"): _remove(),
        ("THANE OF FIFE.—DISMISS","TITLE"): _rename("THANE OF FIFE"),
        ("A PEERLESS KINSMAN",  "REL"):   _remove(),
        ("KNOLL'D",             "REL"):   _remove(),
        ("SERVANTS",            "REL"):   _remove(),
        ("SON",                 "REL"):   _remove(),
        ("SERVANT",             "REL"):   _relabel("TITLE"),
        ("WAITING-GENTLEWOMAN", "REL"):   _remove(),
        ("DOCTOR",              "REL"):   _relabel("TITLE"),
        ("LORD;—THE CASTLE",    "LOCATION"): _rename("THE CASTLE"),
    },

    # A MIDSUMMER NIGHT'S DREAM ────────────────────────────────────────────
    "A MIDSUMMER NIGHT'S DREAM": {
        ("PHILLIDA",  "GPE"): _relabel("PERSON"),
        ("QUAIL",     "GPE"): _remove(),
        ("LYSANDER",  "REL"): _relabel("PERSON"),
        ("CAPTAIN",   "REL"): _relabel("TITLE"),
        ("KNIGHT",    "REL"): _relabel("TITLE"),
        ("DANCE",     "REL"): _remove(),
        ("KNACKS",       "TITLE"): _remove(),
        ("MOTH",         "TITLE"): _relabel("PERSON"),
        ("STARVELING",   "TITLE"): _relabel("PERSON"),
    },

    #MUCH ADO ABOUT NOTHING
    "MUCH ADO ABOUT NOTHING": {
        ("CLAUDIO",        "GPE"): _relabel("PERSON"),
        ("CLAUDIO,—WHOSE", "GPE"): _remove(),
        ("QUONDAM",        "GPE"): _remove(),
        ("A MERRY HOUR",   "REL"): _remove(),
        ("AUTHOR",         "REL"): _remove(),
        ("CONRADE",        "REL"): _relabel("PERSON"),
        ("JOVE",           "REL"): _relabel("PERSON"),
        ("PRINCE",         "REL"): _relabel("TITLE"),
        ("DAUGHTER",       "REL"): _remove(),
        ("CHAM",           "TITLE"): _remove(),
        ("WOOSS",          "TITLE"): _remove(),
    },

    #THE TRAGEDY OF ROMEO AND JULIET
    "THE TRAGEDY OF ROMEO AND JULIET": {
        ("CAPULET",   "PERSON"): _relabel("FAMILY"),
        ("MONTAGUE",  "PERSON"): _relabel("FAMILY"),
        ("CAPEL",               "GPE"): _remove(),
        ("GRAZE",               "GPE"): _remove(),
        ("KNOCK!—WHO",          "GPE"): _remove(),
        ("SUSAN GRINDSTONE",    "GPE"): _relabel("PERSON"),
        ("TIBERIO",             "GPE"): _relabel("PERSON"),
        ("UNWIELDY",            "GPE"): _remove(),
        ("WHATE'ER",            "GPE"): _remove(),
        ("SAINT PETER'S CHURCH","GPE"): _relabel("LOCATION"),
        ("TH'EXCHANGE",   "REL"): _remove(),
        ("ATOMIES",       "TITLE"): _remove(),
        ("BRAGS OF HIS",  "TITLE"): _remove(),
        ("NEED'ST",       "TITLE"): _remove(),
        ("WRONG'ST",      "TITLE"): _remove(),
    },
}

ENTITY_FIELDS = [
    "play", "act", "scene", "location",
    "entity_text", "label", "label_description"
]

def correct_csv(input_path: Path, output_path: Path) -> tuple:
    rows_kept    = 0
    rows_removed = 0
    rows_changed = 0
    output_rows  = []

    play_title = None

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            play_title     = row["play"].strip()
            entity_text    = row["entity_text"].strip()
            label          = row["label"].strip()

            play_corrections = CORRECTIONS.get(play_title, {})
            action, new_label, new_text = play_corrections.get(
                (entity_text, label), (None, None, None)
            )

            if action == "remove":
                print(f"  [REMOVE]  [{label}] {entity_text}")
                rows_removed += 1
                continue

            if action in ("relabel", "relabel+rename"):
                print(f"  [RELABEL] [{label}] → [{new_label}] {entity_text}")
                row["label"]            = new_label
                row["label_description"] = new_label
                rows_changed += 1

            if action in ("rename", "relabel+rename"):
                print(f"  [RENAME]  {entity_text} → {new_text}")
                row["entity_text"] = new_text
                rows_changed += 1

            output_rows.append(row)
            rows_kept += 1

    # Deduplicate after corrections (rename may create duplicates)
    seen       = set()
    dedup_rows = []
    for row in output_rows:
        key = (row["entity_text"], row["label"])
        if key not in seen:
            seen.add(key)
            dedup_rows.append(row)
        else:
            rows_removed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENTITY_FIELDS)
        writer.writeheader()
        writer.writerows(dedup_rows)

    return len(dedup_rows), rows_removed, rows_changed


def run(play_files: list):
    all_records   = []
    total_removed = 0
    total_changed = 0

    for play_file in play_files:
        csv_path = Globals.OUTPUT_DIR / f"02_{play_file.stem}_finetuned_ner.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path.name} not found — skipping")
            continue

        print(f"\n{'─'*60}")
        print(f"  Correcting: {csv_path.name}")
        print(f"{'─'*60}")

        kept, removed, changed = correct_csv(csv_path, csv_path)
        total_removed += removed
        total_changed += changed

        print(f"  → {kept} rows kept, {removed} removed, {changed} changed")

        with open(csv_path, "r", encoding="utf-8") as f:
            all_records.extend(list(csv.DictReader(f)))

    combined_path = Globals.OUTPUT_DIR / "02_ALL_finetuned_ner.csv"
    with open(combined_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENTITY_FIELDS)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n  Combined CSV saved → {combined_path.name}")
    print(f"  Total removed : {total_removed}")
    print(f"  Total changed : {total_changed}")
