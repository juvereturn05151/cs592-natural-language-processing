"""
File Name:    TokenizerHelper.py
Author(s):    Ju-ve Chankasemporn
Description:  Text processing and tokenization utilities for NLP methods
"""

import nltk
import re
import string
from typing import List, Dict, Set
from dataclasses import dataclass
import src.NLP_Globals as Globals

@dataclass
class DocumentData:
    """Container for document data."""
    filename: str
    path: str
    raw_text: str
    # filtered tokens if tokens have valid tags and not a stopwords
    tokens: List[str]
    stemmed_tokens: List[str]
    raw_counts: Dict[str, int]
    term_frequencies: Dict[str, float]
    unique_terms: Set[str]

class TokenizerHelper:
    """Helper class for text processing and tokenization."""

    def __init__(self, use_stemming: bool = True, use_pos_tagging: bool = True, remove_stopwords: bool = True):
        #we define these attributes because some data sets might be better
        self.use_stemming = use_stemming
        self.use_pos_tagging = use_pos_tagging
        self.remove_stopwords = remove_stopwords

        self._ensure_nltk_data()
        self._initialize_stemmer()

    def _ensure_nltk_data(self) -> None:
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')

    def _initialize_stemmer(self) -> None:
        if self.use_stemming:
            """Initialize the Porter Stemmer."""
            try:
                # try to import PorterStemmer
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer() if self.use_stemming else None
            except ImportError:
                # Fallback if NLTK stemmer not available
                print("Warning: NLTK PorterStemmer not available. Stemming disabled.")
                self.stemmer = None
                self.use_stemming = False

    def tokenize_text(self, text: str) -> List[str]:
        """Modified to respect remove_stopwords flag."""
        if not text or not text.strip():
            return []

        raw_tokens = nltk.wordpunct_tokenize(text.lower())

        tokens = []
        for token in raw_tokens:
            #we remove the punctuation here
            cleaned = re.sub(Globals.REGEX_CLEANER, "", token)

            #only filter if remove_stopwords is True
            if cleaned and cleaned not in string.punctuation:
                if self.remove_stopwords:
                    if cleaned not in Globals.STOP_WORDS:
                        tokens.append(cleaned)
                else:
                    #keep ALL words, including stop words
                    tokens.append(cleaned)
        return tokens

    def pos_tag_and_filter(self, tokens: List[str]) -> List[str]:
        """Apply POS tagging and filter tokens based on valid tags."""
        if not tokens:
            return []

        # POS tagging
        pos_tags = nltk.pos_tag(tokens)

        # Filter by valid POS tags
        filtered_tokens = [
            token for token, tag in pos_tags
            if tag in Globals.VALID_TAGS and token not in Globals.STOP_WORDS
        ]

        return filtered_tokens

    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming to tokens."""
        if not tokens or not self.stemmer:
            return tokens

        return [self.stemmer.stem(token) for token in tokens]

    def process_text(self, text: str, file_path,use_stemming: bool = None) -> DocumentData:
        """Process text through full tokenization pipeline with optional stemming."""
        # determine if stemming should be used
        should_stem = use_stemming if use_stemming is not None else self.use_stemming

        #tokenize
        tokens = self.tokenize_text(text)

        #POS tag and filter
        filtered_tokens = self.pos_tag_and_filter(tokens)

        #apply stemming if requested
        stemmed_tokens = []
        #copy for frequency calculations
        working_tokens = filtered_tokens.copy()

        if should_stem and self.stemmer:
            stemmed_tokens = self.apply_stemming(filtered_tokens)
            #use stemmed tokens for frequency calculations
            working_tokens = stemmed_tokens

        #calculate frequencies based on stemmed or original tokens
        token_counts = {}
        for token in working_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        #calculate normalized frequencies
        total_tokens = len(working_tokens)
        term_frequencies = {}
        for token, count in token_counts.items():
            term_frequencies[token] = count / total_tokens if total_tokens > 0 else 0

        unique_terms = set(token_counts.keys())

        return DocumentData(
            filename=file_path.name,
            path = str(file_path),
            raw_text=text,
            # Original filtered tokens
            tokens=filtered_tokens,
            # Stemmed tokens if used
            stemmed_tokens=stemmed_tokens if should_stem else [],
            term_frequencies=term_frequencies,
            raw_counts=token_counts,
            unique_terms = unique_terms,
        )

    def process_query(self, query: str, use_stemming: bool = None) -> List[str]:
        """ Process a search query into relevant terms with optional stemming."""
        if not query or not query.strip():
            return []

        # Determine if stemming should be used
        should_stem = use_stemming if use_stemming is not None else self.use_stemming

        # Tokenize the query
        tokens = self.tokenize_text(query)

        # Filter using POS tags
        filtered_tokens = self.pos_tag_and_filter(tokens)

        # Apply stemming if requested
        if should_stem and self.stemmer:
            filtered_tokens = self.apply_stemming(filtered_tokens)

        return filtered_tokens