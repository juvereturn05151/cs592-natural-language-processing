"""
File Name:    data_extractor.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

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