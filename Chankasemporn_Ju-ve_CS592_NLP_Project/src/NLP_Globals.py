"""
File Name:    NLP_Globals.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import nltk
import re

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
VALID_TAGS = ['NN', 'NNP', 'NNPS', 'NNS', 'CD', 'FW', 'JJ', 'JJR', 'JJS']
REGEX_CLEANER = re.compile(r"['—_\“\”\"\”’‘\-)\:!\&]")