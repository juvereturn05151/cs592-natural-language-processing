"""
File Name:    TokenizerHelper.py
Author(s):    Ju-ve Chankasemporn
Description:  Text processing and tokenization utilities for NLP methods
"""

import nltk
import re
import string
from typing import List, Dict, Tuple
from dataclasses import dataclass
import src.NLP_Globals as Globals

@dataclass
class TokenizedDocument:
    """Container for tokenized document data."""
    name: str
    tokens: List[str]
    term_frequencies: Dict[str, float]
    raw_counts: Dict[str, int]
    text: str = ""


class TokenizerHelper:
    """Helper class for text processing and tokenization."""

    def __init__(self):
        # Ensure required NLTK data is available
        self._ensure_nltk_data()

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

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into cleaned, filtered tokens.

        Args:
            text: Input text string

        Returns:
            List of processed tokens
        """
        if not text or not text.strip():
            return []

        # Convert to lowercase and tokenize
        token_list = []
        for word in nltk.WordPunctTokenizer().tokenize(text.lower()):
            cleaned = re.sub(Globals.REGEX_CLEANER, "", word)
            if cleaned and cleaned not in Globals.STOP_WORDS and cleaned not in string.punctuation:
                token_list.append(cleaned)

        return token_list

    def pos_tag_and_filter(self, tokens: List[str]) -> List[str]:
        """
        Apply POS tagging and filter tokens based on valid tags.

        Args:
            tokens: List of token strings

        Returns:
            List of filtered tokens
        """
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

    def process_text(self, text: str, doc_name: str = "") -> TokenizedDocument:
        """
        Process text through full tokenization pipeline.

        Args:
            text: Input text
            doc_name: Optional document name

        Returns:
            TokenizedDocument object
        """
        # Tokenize
        tokens = self.tokenize_text(text)

        # POS tag and filter
        filtered_tokens = self.pos_tag_and_filter(tokens)

        # Calculate frequencies
        token_counts = {}
        for token in filtered_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Calculate normalized frequencies
        total_tokens = len(filtered_tokens)
        term_frequencies = {}
        for token, count in token_counts.items():
            term_frequencies[token] = count / total_tokens if total_tokens > 0 else 0

        return TokenizedDocument(
            name=doc_name,
            tokens=filtered_tokens,
            term_frequencies=term_frequencies,
            raw_counts=token_counts,
            text=text
        )

    def process_query(self, query: str) -> List[str]:
        """
        Process a search query into relevant terms.

        Args:
            query: Search query string

        Returns:
            List of processed query terms
        """
        if not query or not query.strip():
            return []

        # Tokenize the query
        tokens = self.tokenize_text(query)

        # Filter using POS tags
        filtered_tokens = self.pos_tag_and_filter(tokens)

        return filtered_tokens

    def calculate_tfidf_scores(
        self,
        doc_frequencies: Dict[str, float],
        idf_values: Dict[str, float],
        scale_factor: float = 100.0
    ) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for document terms.

        Args:
            doc_frequencies: Term frequencies in document
            idf_values: IDF values for terms
            scale_factor: Scaling factor for scores

        Returns:
            Dictionary of term -> TF-IDF score
        """
        scores = {}
        for term, tf in doc_frequencies.items():
            idf = idf_values.get(term, 0)
            scores[term] = tf * idf * scale_factor

        return scores