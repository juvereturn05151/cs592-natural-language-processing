"""
File Name:    Project2Runner.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import spacy
from pathlib import Path

import Project2Globals as Globals
import src.Project2.NER_Extraction.ner_extraction as NER_Extraction

class Project2Runner:
    def __init__(self, n_iter: int = 40):
        self.n_iter = n_iter #n_iter = iterations for fine-tuning

        print("Find all Shakespeare play files")
        self.play_files = self._find_plays(Globals.TRAIN_DIR)
        print("Loading spaCy model: en_core_web_md")
        self.nlp = spacy.load("en_core_web_md")

        self.nlp_ft = None  #fine-tuned model (set after step 2)

    # Local
    def _find_plays(self, train_dir: Path) -> list:
        files = sorted([
            p for p in train_dir.rglob("*.txt")
            if "shakespeare" in p.name.lower()
        ])
        if not files:
            raise FileNotFoundError(f"No Shakespeare files found in {train_dir}")
        print(f"\nFound {len(files)} Shakespeare file(s):")
        for p in files:
            print(f"  {p.name}")
        return files

    def _print_header(self, step: int, title: str):
        print(f"\n{'═' * 60}")
        print(f"  STEP {step}: {title}")
        print(f"{'═' * 60}")

    #step 1 NER Extraction
    def run_ner_extraction(self):
        self._print_header(1, "NER EXTRACTION")

        all_records = []
        all_mislabel = []
        all_cast_rels = []

        all_records, all_mislabel,all_cast_rels = NER_Extraction.run(nlp=self.nlp, play_files=self.play_files)

        print(f"\n{'═' * 60}")
        print(f"  COMPLETE")
        print(f"  Total entity records : {len(all_records):,}")
        print(f"  Total mislabelings   : {len(all_mislabel)}")
        print(f"  Total cast_rels   : {len(all_cast_rels)}")
