"""
File Name:    NLP_Globals.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
from __future__ import annotations
import nltk
import re
from pathlib import Path
from typing import Iterable, Optional

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
VALID_TAGS = ['NN', 'NNP', 'NNPS', 'NNS', 'CD', 'FW', 'JJ', 'JJR', 'JJS']
REGEX_CLEANER = re.compile(r"['—_\“\”\"\”’‘\-)\:!\&]")
REGEX_NONWORD = re.compile(r"[^\w]+")

def _find_repo_root(start: Path, markers: Iterable[str] = ("pyproject.toml", ".git", "requirements.txt")) -> Path:
    """Walk upward from `start` until we find a directory that looks like the repo root. Falls back to `start` if nothing found.
    """
    start = start.resolve()
    for p in (start, *start.parents):
        if any((p / m).exists() for m in markers):
            return p
    return start  # fallback

def _first_existing_dir(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        if p is not None and p.exists() and p.is_dir():
            return p
    return None

def get_default_data_dir(verbose: bool = True) -> str:
    """Find the dataset directory `data/train` robustly."""
    # 0) Optional overrides (nice for grading scripts / different machines)
    import os
    for env_key in ("CS592_DATA_DIR", "DATA_DIR"):
        env_val = os.getenv(env_key)
        if env_val:
            p = Path(env_val).expanduser().resolve()
            if p.exists() and p.is_dir():
                if verbose:
                    print(f"[data_dir] Using {env_key} override: {p}")
                return str(p)

    current_file = Path(__file__).resolve()
    cwd = Path.cwd().resolve()

    # 1) Find a sensible root (repo root if markers exist)
    repo_root = _find_repo_root(current_file.parent)

    # 2) Candidate paths (ordered)
    #    Add your expected “student project folder” variant here.
    candidates = [
        # Common: repo_root/data/train
        repo_root / "data" / "train",

        # Your real expected location: repo_root/<student_project>/data/train
        repo_root / "Chankasemporn_Ju-ve_CS592_NLP_Project" / "data" / "train",

        # Sometimes repo_root is one level above, so try parents too
        repo_root.parent / "data" / "train",
        repo_root.parent / "Chankasemporn_Ju-ve_CS592_NLP_Project" / "data" / "train",

        # Relative to this file (in case markers weren't found)
        current_file.parent / "data" / "train",
        current_file.parent.parent / "data" / "train",
        current_file.parent.parent.parent / "data" / "train",

        # Relative to current working directory
        cwd / "data" / "train",
        cwd / "Chankasemporn_Ju-ve_CS592_NLP_Project" / "data" / "train",
        cwd.parent / "data" / "train",
        cwd.parent / "Chankasemporn_Ju-ve_CS592_NLP_Project" / "data" / "train",
    ]

    found = _first_existing_dir(candidates)

    if found:
        if verbose:
            print(f"[data_dir] Found data at: {found}")
        return str(found)

    # 3) Helpful error message
    if verbose:
        print("[data_dir] Could not find data directory. Tried:")
        for c in candidates:
            print(f"  - {c} (exists: {c.exists()})")

    # Fail loudly (better than returning a wrong path and error later)
    raise FileNotFoundError(
        "Could not locate the training data directory (data/train). "
        "Set CS592_DATA_DIR to the correct folder if needed."
    )