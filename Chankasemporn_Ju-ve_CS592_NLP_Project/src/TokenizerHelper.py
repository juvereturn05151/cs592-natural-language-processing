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
    stemmed_tokens: List[str]  # Added stemmed tokens
    term_frequencies: Dict[str, float]
    raw_counts: Dict[str, int]
    text: str = ""
    use_stemming: bool = False  # Track if stemming was used


class TokenizerHelper:
    """Helper class for text processing and tokenization."""

    def __init__(self, use_stemming: bool = True):
        """
        Initialize TokenizerHelper.

        Args:
            use_stemming: Whether to apply stemming (default: True)
        """
        self.use_stemming = use_stemming
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
        """Initialize the Porter Stemmer."""
        try:
            # Try to import PorterStemmer
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer() if self.use_stemming else None
        except ImportError:
            # Fallback if NLTK stemmer not available
            print("Warning: NLTK PorterStemmer not available. Stemming disabled.")
            self.stemmer = None
            self.use_stemming = False

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

    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter stemming to tokens.

        Args:
            tokens: List of tokens to stem

        Returns:
            List of stemmed tokens
        """
        if not tokens or not self.stemmer:
            return tokens

        return [self.stemmer.stem(token) for token in tokens]

    def process_text(self, text: str, doc_name: str = "", use_stemming: bool = None) -> TokenizedDocument:
        """
        Process text through full tokenization pipeline with optional stemming.

        Args:
            text: Input text
            doc_name: Optional document name
            use_stemming: Override instance setting for stemming (default: None = use instance setting)

        Returns:
            TokenizedDocument object
        """
        # Determine if stemming should be used
        should_stem = use_stemming if use_stemming is not None else self.use_stemming

        # Tokenize
        tokens = self.tokenize_text(text)

        # POS tag and filter
        filtered_tokens = self.pos_tag_and_filter(tokens)

        # Apply stemming if requested
        stemmed_tokens = []
        working_tokens = filtered_tokens.copy()  # Copy for frequency calculations

        if should_stem and self.stemmer:
            stemmed_tokens = self.apply_stemming(filtered_tokens)
            working_tokens = stemmed_tokens  # Use stemmed tokens for frequency calculations

        # Calculate frequencies based on stemmed or original tokens
        token_counts = {}
        for token in working_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Calculate normalized frequencies
        total_tokens = len(working_tokens)
        term_frequencies = {}
        for token, count in token_counts.items():
            term_frequencies[token] = count / total_tokens if total_tokens > 0 else 0

        return TokenizedDocument(
            name=doc_name,
            tokens=filtered_tokens,  # Original filtered tokens
            stemmed_tokens=stemmed_tokens if should_stem else [],  # Stemmed tokens if used
            term_frequencies=term_frequencies,
            raw_counts=token_counts,
            text=text,
            use_stemming=should_stem
        )

    def process_query(self, query: str, use_stemming: bool = None) -> List[str]:
        """
        Process a search query into relevant terms with optional stemming.

        Args:
            query: Search query string
            use_stemming: Override instance setting for stemming

        Returns:
            List of processed query terms
        """
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

    def get_original_from_stemmed(self, stemmed_token: str, original_tokens: List[str]) -> List[str]:
        """
        Find original token(s) that map to a stemmed token.

        Args:
            stemmed_token: A stemmed token
            original_tokens: List of original tokens

        Returns:
            List of original tokens that stem to the given stemmed token
        """
        if not self.stemmer or not original_tokens:
            return []

        matching_tokens = []
        for token in original_tokens:
            if self.stemmer.stem(token) == stemmed_token:
                matching_tokens.append(token)

        return matching_tokens

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