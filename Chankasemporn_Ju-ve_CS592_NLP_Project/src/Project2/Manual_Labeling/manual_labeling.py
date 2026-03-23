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

    # ── MACBETH ───────────────────────────────────────────────────────────────
    "MACBETH": {
        ("CAPTIVITY.—HAIL", "GPE"):  _remove(),
        ("MASTERDOM",       "GPE"):  _remove(),
        ("NORWAYS",         "GPE"):  _relabel_rename("NORP", "NORWEGIAN"),
        ("PADDOCK",         "GPE"):  _remove(),
        ("SEYTON",          "GPE"):  _relabel("PERSON"),
        ("THANE",           "GPE"):  _remove(),
        ("TILL BIRNAM",     "GPE"):  _rename("BIRNAM"),
        ("GLAMIS",          "GPE"):  _relabel("TITLE"),
        ("NEPTUNE",         "LOC"):  _remove(),
        # NORP noise
        ("ANGERLY",         "NORP"): _remove(),
        ("FIFE.—DISMISS",   "NORP"): _remove(),
        ("HEAVEN",          "NORP"): _remove(),
        ("SCONE",           "NORP"): _relabel("GPE"),
        ("THINE",           "NORP"): _remove(),
        ("THOU",            "NORP"): _remove(),
        ("TUNE",            "NORP"): _remove(),
        ("UNROUGH",         "NORP"): _remove(),
        # ORG noise
        ("ACHERON",         "ORG"):  _relabel("LOC"),
        ("ADMIR'D",         "ORG"):  _remove(),
        ("ARMY",            "ORG"):  _remove(),
        ("BANQUO",          "ORG"):  _relabel("PERSON"),
        ("BELZEBUB",        "ORG"):  _remove(),
        ("COMPT",           "ORG"):  _remove(),
        ("FIL'D",           "ORG"):  _remove(),
        ("GLAMIS",          "ORG"):  _relabel("TITLE"),
        ("GRACE",           "ORG"):  _remove(),
        ("HAST",            "ORG"):  _remove(),
        ("HAUTBOYS",        "ORG"):  _remove(),
        ("ILL",             "ORG"):  _remove(),
        ("LORD",            "ORG"):  _relabel("TITLE"),
        ("LORDS",           "ORG"):  _remove(),
        ("LOYAL",           "ORG"):  _remove(),
        ("NOTE",            "ORG"):  _remove(),
        ("OURSELF",         "ORG"):  _remove(),
        ("PALE",            "ORG"):  _remove(),
        ("POINT",           "ORG"):  _remove(),
        ("ROOT OF HEMLOCK", "ORG"):  _remove(),
        ("SO:—BUT",         "ORG"):  _remove(),
        ("TIME",            "ORG"):  _remove(),
        ("TRUST",           "ORG"):  _remove(),
        ("WHISP'RINGS",     "ORG"):  _remove(),
        ("WIND.—I",         "ORG"):  _remove(),
        # FAC noise
        ("LORD;—THE CASTLE'S","FAC"):_remove(),
        ("THE GHOST OF",    "FAC"):  _relabel("REL"),
        # WORK_OF_ART noise
        ("BANQUO",          "WORK_OF_ART"): _relabel("PERSON"),
        ("BLACK SPIRITS",   "WORK_OF_ART"): _remove(),
        ("FILLET",          "WORK_OF_ART"): _remove(),
        ("FORTUNE",         "WORK_OF_ART"): _remove(),
        ("GHOST RISES",     "WORK_OF_ART"): _remove(),
        ("LOVE",            "WORK_OF_ART"): _remove(),
        ("NATURE",          "WORK_OF_ART"): _remove(),
        # PRODUCT noise
        ("AFTER",           "PRODUCT"): _remove(),
        ("PAYS",            "PRODUCT"): _remove(),
        ("SAUCY",           "PRODUCT"): _remove(),
        ("SEYTON!—I",       "PRODUCT"): _remove(),
        ("WHOLE",           "PRODUCT"): _remove(),
        # DATE noise
        ("ALMOST SLIPP'D THE","DATE"): _remove(),
        ("EVERY ONE",       "DATE"):  _remove(),
        ("FEE",             "DATE"):  _remove(),
        ("FIRST",           "DATE"):  _relabel("ORDINAL"),
        ("MAY",             "DATE"):  _remove(),
        ("SICKEN",          "DATE"):  _remove(),
        ("THIS A QUARTER",  "DATE"):  _remove(),
        # REL noise
        ("IN LOVE",         "REL"):   _remove(),
        ("SERVANT",         "REL"):   _relabel("TITLE"),
        # TIME noise
        ("ALMOST FORGOT THE","TIME"): _remove(),
        ("DIED HEREAFTER",  "TIME"):  _remove(),
        ("LIV'D",           "TIME"):  _remove(),
        ("NIGHTGOWN",       "TIME"):  _remove(),
        ("PALE!—LIGHT",     "TIME"):  _remove(),
        ("TWAS",            "TIME"):  _remove(),
        ("A",               "CARDINAL"): _remove(),
        ("LEAST",           "CARDINAL"): _remove(),
        ("MANY",            "CARDINAL"): _remove(),
        ("NEARLY",          "CARDINAL"): _remove(),
        ("OWE",             "CARDINAL"): _remove(),
        ("THOUSANDS",       "CARDINAL"): _remove(),
        ("LENT",            "EVENT"): _remove(),
    },

    # ── A MIDSUMMER NIGHT'S DREAM ─────────────────────────────────────────────
    "A MIDSUMMER NIGHT'S DREAM": {
        # GPE noise
        ("BEACHÈD",         "GPE"):  _remove(),
        ("CRAZÈD",          "GPE"):  _remove(),
        ("DALE",            "GPE"):  _remove(),
        ("KNACKS",          "GPE"):  _remove(),
        ("SHREWISHNESS",    "GPE"):  _remove(),
        # LOC noise
        ("CRYSTAL",         "LOC"):  _remove(),
        ("ETHIOPE",         "LOC"):  _remove(),
        ("NEPTUNE",         "LOC"):  _remove(),
        ("QUAIL",           "LOC"):  _remove(),
        ("THE WATERY MOON", "LOC"):  _remove(),
        ("WATER",           "LOC"):  _remove(),
        # NORP noise
        ("FAINTNESS",       "NORP"): _remove(),
        ("GLEEK",           "NORP"): _remove(),
        ("MOONSHINE",       "NORP"): _remove(),
        ("THOU",            "NORP"): _remove(),
        # ORG noise
        ("AMAZON",          "ORG"):  _relabel("LOC"),
        ("APOLLO",          "ORG"):  _remove(),
        ("BOTTOM",          "ORG"):  _relabel("PERSON"),
        ("EXIT",            "ORG"):  _remove(),
        ("EYNE",            "ORG"):  _remove(),
        ("GRACE",           "ORG"):  _remove(),
        ("HAST",            "ORG"):  _remove(),
        ("HIPPOLYTA",       "ORG"):  _relabel("PERSON"),
        ("ILL",             "ORG"):  _remove(),
        ("LION AND MOONSHINE","ORG"):_remove(),
        ("LION ARE",        "ORG"):  _remove(),
        ("LION AS",         "ORG"):  _remove(),
        ("LION HIGHT",      "ORG"):  _remove(),
        ("LION ROARS",      "ORG"):  _remove(),
        ("LION VILE WITH",  "ORG"):  _remove(),
        ("LION, MOONSHINE", "ORG"):  _remove(),
        ("LION. DEMETRIUS", "ORG"):  _remove(),
        ("NEDAR",           "ORG"):  _relabel("PERSON"),
        ("NOTE",            "ORG"):  _remove(),
        ("PARD",            "ORG"):  _remove(),
        ("PHIBBUS",         "ORG"):  _remove(),
        ("PROLOGUE",        "ORG"):  _remove(),
        ("PUCK",            "ORG"):  _relabel("PERSON"),
        ("READY",           "ORG"):  _remove(),
        ("STAND",           "ORG"):  _remove(),
        ("THISBE",          "ORG"):  _relabel("PERSON"),
        ("THROTTLE",        "ORG"):  _remove(),
        ("TIME",            "ORG"):  _remove(),
        ("TRUST",           "ORG"):  _remove(),
        ("VENUS",           "ORG"):  _remove(),
        ("WITHDRAW",        "ORG"):  _remove(),
        # FAC noise
        ("NINUS",           "FAC"):  _relabel("PERSON"),
        ("THE FAIRY QUEEN", "FAC"):  _relabel("PERSON"),
        # WORK_OF_ART noise
        ("FAIRY QUEEN",     "WORK_OF_ART"): _relabel("PERSON"),
        ("LOVE",            "WORK_OF_ART"): _remove(),
        ("NATURE",          "WORK_OF_ART"): _remove(),
        ("PERADVENTURE",    "WORK_OF_ART"): _remove(),
        # PRODUCT noise
        ("NOLE",            "PRODUCT"): _remove(),
        ("TARTAR",          "PRODUCT"): _remove(),
        ("TIDE",            "PRODUCT"): _remove(),
        ("TRAIN",           "PRODUCT"): _remove(),
        ("VENUS",           "PRODUCT"): _remove(),
        ("IMPAIRED",        "DATE"):  _remove(),
        ("MOTE",            "DATE"):  _remove(),
        ("PAST",            "DATE"):  _remove(),
        ("THE THROSTLE",    "DATE"):  _remove(),
        ("THIS FOND",       "DATE"):  _remove(),
        ("A DOVE?",         "TIME"):  _remove(),
        ("LION",            "TIME"):  _remove(),
        ("MUSTARDSEED",     "REL"):   _relabel("PERSON"),
        ("DUCHESS",         "LAW"):   _relabel("TITLE"),
        ("SNOUT",           "CARDINAL"): _relabel("PERSON"),
        ("THIS",            "ORDINAL"): _remove(),
    },

    # ── MUCH ADO ABOUT NOTHING ────────────────────────────────────────────────
    "MUCH ADO ABOUT NOTHING": {
        # GPE noise
        ("MILAN",           "GPE"):  _relabel("LOC"),
        ("PHARAOH",         "GPE"):  _relabel("TITLE"),
        ("QUONDAM",         "GPE"):  _remove(),
        ("THE PRINCE WOOSS","GPE"):  _remove(),
        ("VACANT",          "GPE"):  _remove(),
        ("VENICE",          "GPE"):  _relabel("LOC"),
        # LOC noise
        ("ASIA",            "LOC"):  _relabel("GPE"),
        ("ETHIOPE",         "LOC"):  _remove(),
        ("EUROPA",          "LOC"):  _relabel("GPE"),
        # NORP noise
        ("BORACHIO",        "NORP"): _relabel("PERSON"),
        ("CLAUDIO",         "NORP"): _relabel("PERSON"),
        ("MAN,—AS",         "NORP"): _remove(),
        ("SHORTENED,—FOR",  "NORP"): _remove(),
        ("THINE",           "NORP"): _remove(),
        ("THOU",            "NORP"): _remove(),
        ("UNKISSED",        "NORP"): _remove(),
        ("WHIPT",           "NORP"): _remove(),
        # ORG noise
        ("AMEN",            "ORG"):  _remove(),
        ("BALTHASAR",       "ORG"):  _relabel("PERSON"),
        ("BENEDICK",        "ORG"):  _relabel("PERSON"),
        ("CLAUDIO",         "ORG"):  _relabel("PERSON"),
        ("COUNT",           "ORG"):  _relabel("TITLE"),
        ("COUNT COMFECT",   "ORG"):  _remove(),
        ("EXIT",            "ORG"):  _remove(),
        ("EXIT BOY",        "ORG"):  _remove(),
        ("FIRST WATCH",     "ORG"):  _remove(),
        ("GRACE",           "ORG"):  _remove(),
        ("HERO",            "ORG"):  _relabel("PERSON"),
        ("LEAVY",           "ORG"):  _remove(),
        ("LORD",            "ORG"):  _relabel("TITLE"),
        ("LORDSHIP",        "ORG"):  _remove(),
        ("MEG",             "ORG"):  _remove(),
        ("NOTE",            "ORG"):  _remove(),
        ("SECOND WATCH",    "ORG"):  _remove(),
        ("STAND",           "ORG"):  _remove(),
        ("TIME",            "ORG"):  _remove(),
        ("TRUST",           "ORG"):  _remove(),
        ("WITHDRAW",        "ORG"):  _remove(),

        ("THE COUNT CLAUDIO","FAC"): _remove(),
        ("THIS COUNT?",     "FAC"):  _remove(),
        # WORK_OF_ART noise
        ("BOY",             "WORK_OF_ART"): _remove(),
        ("DAUGHTER",        "WORK_OF_ART"): _remove(),
        ("FORBID",          "WORK_OF_ART"): _remove(),
        ("FORTUNE",         "WORK_OF_ART"): _remove(),
        ("LIGHT O'",        "WORK_OF_ART"): _remove(),
        ("LOVE",            "WORK_OF_ART"): _relabel("REL"),
        ("NATURE",          "WORK_OF_ART"): _remove(),
        ("TROILUS",         "WORK_OF_ART"): _relabel("PERSON"),
        ("TROTH",           "WORK_OF_ART"): _remove(),
        ("URSULA",          "WORK_OF_ART"): _relabel("PERSON"),
        # PRODUCT noise
        ("CARDUUS",         "PRODUCT"): _remove(),
        ("ENIGMATICAL",     "PRODUCT"): _remove(),
        ("MEDICINABLE",     "PRODUCT"): _remove(),
        ("SATURN",          "PRODUCT"): _remove(),
        ("VENUS",           "PRODUCT"): _remove(),
        ("VULCAN",          "PRODUCT"): _remove(),
        # REL noise
        ("CONSTABLE",       "REL"):   _relabel("TITLE"),
        # DATE noise
        ("A ONE",           "DATE"):  _remove(),
        ("A SCHOOL",        "DATE"):  _remove(),
        ("A THOUSAND HALFPENCE","DATE"): _remove(),
        ("ALMOST",          "DATE"):  _remove(),
        ("ALMOST SICK",     "DATE"):  _remove(),
        ("OLD",             "DATE"):  _remove(),
        ("THE WINDY",       "DATE"):  _remove(),
        ("TWO OLD",         "DATE"):  _remove(),
        # TIME noise
        ("AN",              "TIME"):  _remove(),
        ("WARS",            "TIME"):  _relabel("EVENT"),
        # CARDINAL noise
        ("ALMOST",          "CARDINAL"): _remove(),
        ("NEARLY",          "CARDINAL"): _remove(),
        ("OWE",             "CARDINAL"): _remove(),
    },

    # ── THE TRAGEDY OF ROMEO AND JULIET ───────────────────────────────────────
    "THE TRAGEDY OF ROMEO AND JULIET": {
        ("CHEERLY",         "GPE"):  _remove(),
        ("COUNTY",          "GPE"):  _relabel("TITLE"),
        ("GRAZE",           "GPE"):  _remove(),
        ("I'FAITH",         "GPE"):  _remove(),
        ("KNOCK!—WHO",      "GPE"):  _remove(),
        ("ROMEO",           "GPE"):  _relabel("PERSON"),
        ("TIBERIO",         "GPE"):  _relabel("PERSON"),
        ("UNFIRM",          "GPE"):  _remove(),
        ("UNWIELDY",        "GPE"):  _remove(),
        ("EARTH",           "LOC"):  _remove(),
        ("ECHO",            "LOC"):  _remove(),
        ("DINE",            "NORP"): _remove(),
        ("DISPRAISE",       "NORP"): _remove(),
        ("EYES",            "NORP"): _remove(),
        ("GLEEK",           "NORP"): _remove(),
        ("NURSE,—O",        "NORP"): _remove(),
        ("PARTIZANS",       "NORP"): _remove(),
        ("SLUTTISH",        "NORP"): _remove(),
        ("SWOUNDED",        "NORP"): _remove(),
        ("THOU",            "NORP"): _remove(),
        ("ABRAM",           "ORG"):  _relabel("PERSON"),
        ("ALONE",           "ORG"):  _remove(),
        ("AMEN",            "ORG"):  _remove(),
        ("ASK'D",           "ORG"):  _remove(),
        ("EXIT",            "ORG"):  _remove(),
        ("FIRST SERVANT",   "ORG"):  _remove(),
        ("FIRST WATCH",     "ORG"):  _remove(),
        ("FRIAR",           "ORG"):  _relabel("TITLE"),
        ("GRIEFS",          "ORG"):  _remove(),
        ("HAST",            "ORG"):  _remove(),
        ("JULIET",          "ORG"):  _relabel("PERSON"),
        ("LOGGERHEAD.—GOOD","ORG"):  _remove(),
        ("LORD",            "ORG"):  _remove(),
        ("MISERY",          "ORG"):  _remove(),
        ("CAPULET",         "PERSON"): _relabel("FAMILY"),
        ("MONTAGUE",        "ORG"):  _relabel("FAMILY"),
        ("NOTE",            "ORG"):  _remove(),
        ("NURSE",           "ORG"):  _relabel("TITLE"),
        ("ORCHARD",         "ORG"):  _relabel("LOCATION"),
        ("PAGE",            "ORG"):  _relabel("PERSON"),
        ("PALE",            "ORG"):  _remove(),
        ("PEACE",           "ORG"):  _remove(),
        ("PHOEBUS",         "ORG"):  _remove(),
        ("PILGRIM",         "ORG"):  _remove(),
        ("PROLOGUE",        "ORG"):  _remove(),
        ("RETIRES",         "ORG"):  _remove(),
        ("REVIVE",          "ORG"):  _remove(),
        ("THE HIGHMOST HILL","ORG"): _relabel("LOCATION"),
        ("THIRD WATCH",     "ORG"):  _remove(),
        ("TIME",            "ORG"):  _remove(),
        ("TITAN",           "ORG"):  _remove(),
        ("TRUST",           "ORG"):  _remove(),
        ("VALENTIO",        "ORG"):  _relabel("PERSON"),
        ("WITHDRAW",        "ORG"):  _remove(),

        ("AN O?",           "FAC"):  _remove(),
        ("MISTA'EN",        "FAC"):  _remove(),
        # WORK_OF_ART noise
        ("AFTER",           "WORK_OF_ART"): _remove(),
        ("FORTUNE",         "WORK_OF_ART"): _remove(),
        ("SNATCHING ROMEO", "WORK_OF_ART"): _remove(),
        ("TROTH",           "WORK_OF_ART"): _remove(),
        ("WIVES",           "WORK_OF_ART"): _remove(),
        ("LIV'D",           "PRODUCT"): _remove(),
        ("TARTAR",          "PRODUCT"): _remove(),
        ("TIDE",            "PRODUCT"): _remove(),
        ("VENUS",           "PRODUCT"): _remove(),
        ("ROMEO",           "REL"):   _relabel("PERSON"),
        ("SAUCE",           "REL"):   _remove(),
        ("A MONTAGUE",      "DATE"):  _remove(),
        ("FIRST",           "DATE"):  _relabel("ORDINAL"),
        ("LIST",            "DATE"):  _remove(),
        ("LOSS",            "DATE"):  _remove(),
        ("PAGE",            "DATE"):  _remove(),
        ("SHARP MISERY HAD","DATE"):  _remove(),
        ("THIS",            "DATE"):  _remove(),
        ("THIS DAY AN",     "DATE"):  _remove(),
        # TIME noise
        ("LORD.—LIGHT",     "TIME"):  _remove(),
        ("SOME DISTEMPERATURE","TIME"):_remove(),
        ("THE NIGHTINGALE", "TIME"):  _remove(),
        ("THIS PALACE OF DIM NIGHT","TIME"): _remove(),
        ("WOLVISH-RAVENING","TIME"):  _remove(),

        ("LEAST",           "CARDINAL"): _remove(),
        ("PENNY",           "CARDINAL"): _remove(),
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
    print(f"\n  Combined CSV saved -> {combined_path.name}")
    print(f"  Total removed : {total_removed}")
    print(f"  Total changed : {total_changed}")