"""
File Name:    data_extractor.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import spacy
import sys

import xml.etree.ElementTree as ET

from pathlib import Path

def initialize_nlp(model_name: str = "en_core_web_md"):
    return spacy.load(model_name)

def find_repo_root():
    start_path = Path.cwd()

    current = start_path.resolve()

    while current.name != "src":
        if current.parent == current:
            raise RuntimeError("Could not find 'src' directory in path.")
        current = current.parent

    src_dir = current
    repo_root = src_dir.parent

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    return repo_root, src_dir

def load_corpus(corpus_name: str, dataset: str = "train"):
    repo_root, src_dir = find_repo_root()

    file_path = repo_root / "data" / dataset / corpus_name

    if not file_path.exists():
        raise FileNotFoundError(f"Corpus not found: {file_path}")

    tree = ET.parse(file_path)
    root = tree.getroot()

    return root, file_path

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

