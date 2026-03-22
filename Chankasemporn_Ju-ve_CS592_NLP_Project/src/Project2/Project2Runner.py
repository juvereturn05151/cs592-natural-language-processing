"""
File Name:    Project2Runner.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import spacy
from pathlib import Path

import src.Project2.Project2Globals as Globals
import src.Project2.NER_Extraction.ner_extraction as NER_Extraction
import src.Project2.Fine_Tuning.fine_tuning as FineTuning
import src.Project2.Coreference.coreference as Coreference

class Project2Runner:
    def __init__(self):
        print("Find all Shakespeare play files")
        self.play_files = self._find_plays(Globals.TRAIN_DIR)
        print("Loading spaCy model: en_core_web_md")
        self.nlp = spacy.load("en_core_web_md")

        self.nlp_ft    = None  # fine-tuned model (set after step 2)
        self.cast_rels = []    # cast relationships (set after step 1)

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

    # ── STEP 1: NER Extraction ────────────────────────────────────
    def run_ner_extraction(self):
        self._print_header(1, "NER EXTRACTION")

        all_records, all_mislabel, all_cast_rels = NER_Extraction.run(
            nlp=self.nlp,
            play_files=self.play_files
        )

        self.cast_rels = all_cast_rels

        print(f"\n{'═' * 60}")
        print(f"  COMPLETE")
        print(f"  Total entity records : {len(all_records):,}")
        print(f"  Total mislabelings   : {len(all_mislabel)}")
        print(f"  Total cast relations : {len(all_cast_rels)}")

    # ── STEP 2: Fine-Tuning ───────────────────────────────────────
    def run_fine_tuning(self, n_iter: int = 40):
        self._print_header(2, "FINE-TUNING")

        if not self.cast_rels:
            print("  WARNING: No cast relationships found.")
            print("  Run run_ner_extraction() first.")

        self.nlp_ft, all_ft_records = FineTuning.run(
            nlp_base           = self.nlp,
            play_files         = self.play_files,
            cast_relationships = self.cast_rels,
            n_iter             = n_iter
        )

        print(f"\n{'═' * 60}")
        print(f"  COMPLETE")
        print(f"  Fine-tuned entity records : {len(all_ft_records):,}")
        print(f"  Model saved to            : {Globals.MODEL_OUT}")

    # ── STEP 3: Coreference Resolution ───────────────────────────
    def run_coreference(self):
        self._print_header(3, "COREFERENCE RESOLUTION")

        # Use fine-tuned model if available, fall back to base model
        if self.nlp_ft is not None:
            print("  Using fine-tuned model for coreference resolution.")
            nlp_coref = self.nlp_ft
        else:
            print("  WARNING: Fine-tuned model not found.")
            print("  Run run_fine_tuning() first for best results.")
            print("  Falling back to base en_core_web_md model.")
            nlp_coref = self.nlp

        all_coref_records = Coreference.run(
            nlp        = nlp_coref,
            play_files = self.play_files
        )

        print(f"\n{'═' * 60}")
        print(f"  COMPLETE")
        print(f"  Total pronoun resolutions : {len(all_coref_records):,}")

    # ── RUN ALL ───────────────────────────────────────────────────
    def run_all(self, n_iter: int = 40):
        """Run the full pipeline in order."""
        self.run_ner_extraction()
        self.run_fine_tuning(n_iter=n_iter)
        self.run_coreference()