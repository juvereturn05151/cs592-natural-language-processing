"""
File Name:    Project2Globals.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
import sys
from pathlib import Path

def find_repo_root():
    current = Path.cwd().resolve()
    while current.name != "src":
        if current.parent == current:
            raise RuntimeError("Could not find 'src' directory in path.")
        current = current.parent
    src_dir  = current
    repo_root = src_dir.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return repo_root, src_dir

REPO_ROOT, _ = find_repo_root()
DATA_DIR = REPO_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = DATA_DIR / "output"
MODEL_OUT = REPO_ROOT / "models" / "shakespeare_ner"
