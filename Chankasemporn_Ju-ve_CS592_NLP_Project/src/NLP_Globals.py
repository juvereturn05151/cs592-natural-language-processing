"""
File Name:    NLP_Globals.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import nltk
import re
from pathlib import Path

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
VALID_TAGS = ['NN', 'NNP', 'NNPS', 'NNS', 'CD', 'FW', 'JJ', 'JJR', 'JJS']
REGEX_CLEANER = re.compile(r"['—_\“\”\"\”’‘\-)\:!\&]")


def get_default_data_dir() -> str:
    """Get default data directory."""
    # Get the project root by going up from the current file
    current_file = Path(__file__).resolve()

    # Go up from TF_IDF_Method.py: Project1 -> src -> project_root
    project_root = current_file.parent.parent.parent

    # Construct the correct data directory path
    data_dir = project_root / 'data' / 'train'

    # Debug: print the path for verification
    print(f"Looking for data in: {data_dir}")

    if not data_dir.exists():
        # Try alternative: look relative to current working directory
        alt_data_dir = Path.cwd() / 'data' / 'train'
        if alt_data_dir.exists():
            return str(alt_data_dir)

        # Create a list of possible locations to help debugging
        possible_locations = [
            project_root / 'data' / 'train',
            Path.cwd() / 'data' / 'train',
            Path.cwd().parent / 'data' / 'train',  # One level up
            Path.home() / 'data' / 'train',  # Home directory
        ]

        print("Tried these locations:")
        for loc in possible_locations:
            print(f"  - {loc} (exists: {loc.exists()})")

        # Return the expected path anyway (it will show a clear error on load)
        return str(data_dir)

    return str(data_dir)