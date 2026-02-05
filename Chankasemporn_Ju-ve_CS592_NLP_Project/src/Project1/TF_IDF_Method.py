"""
File Name:    TF_IDF_Method.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import nltk
import re
import string
import xml.etree.ElementTree as ET
from collections import defaultdict
import math
from pathlib import Path
import src.NLP_Globals as Globals
from .KeywordMethod import SearchResult, KeywordMethod

@dataclass
class DocumentData:
    """Stores processed document data for TF-IDF."""
    path: str
    term_frequencies: Dict[str, float]
    raw_counts: Dict[str, int]


class TfIdfMethod:
    """
    TF-IDF implementation for keyword extraction and document search.
    """

    def __init__(self, name: str = "TF-IDF", data_dir: str = None):
        self.name = name
        self.data_dir = data_dir or Globals.get_default_data_dir()
        self.documents: List[DocumentData] = []
        self.corpus_data: Dict[str, Dict[str, float]] = {}  # doc_name -> {term: tf}
        self.idf_cache: Dict[str, float] = {}  # term -> IDF value
        self.doc_names: List[str] = []
        self._is_loaded = False



    def load_documents(self, file_pattern: str = "*.txt") -> None:
        """
        Load and process all documents in the data directory.
        """
        if self._is_loaded:
            return

        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Find all text files
        text_files = list(data_path.glob(file_pattern))
        if not text_files:
            raise FileNotFoundError(f"No text files found in {self.data_dir}")

        print(f"Loading {len(text_files)} documents...")

        for file_path in text_files:
            try:
                # Extract normalized frequencies
                tf_data = self._extract_normalized_frequency(str(file_path))

                # Convert to DocumentData
                term_freq = {}
                raw_counts = {}
                for term, (count, freq) in tf_data.items():
                    term_freq[term] = freq
                    raw_counts[term] = count

                doc_data = DocumentData(
                    path=str(file_path),
                    term_frequencies=term_freq,
                    raw_counts=raw_counts
                )

                self.documents.append(doc_data)
                self.corpus_data[file_path.name] = term_freq
                self.doc_names.append(file_path.name)

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

        # Pre-calculate IDF values for all terms
        self._calculate_idf()
        self._is_loaded = True
        print(f"Loaded {len(self.documents)} documents successfully.")

    def _extract_normalized_frequency(self, file_path: str) -> Dict[str, Tuple[int, float]]:
        """
        Extract and normalize term frequencies from a single document.
        Returns dict: term -> (count, normalized_frequency)
        """
        try:
            tree = ET.parse(file_path)
        except ET.ParseError:
            # If it's a plain text file, not XML
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self._process_text(text, Path(file_path).name)

        root = tree.getroot()
        big_string = ""

        body_node = root.find('Body')
        if body_node is not None:
            item_list = body_node.findall('.//Item')
            for item in item_list:
                text = item.text
                if text is None:
                    continue
                big_string += text + " "
        else:
            # Fallback: use all text content
            big_string = ET.tostring(root, method='text', encoding='unicode')

        return self._process_text(big_string, Path(file_path).name)

    def _process_text(self, text: str, doc_name: str) -> Dict[str, Tuple[int, float]]:
        """Process raw text into token frequencies."""
        if not text:
            return {}

        # Tokenize
        token_list = []
        for word in nltk.WordPunctTokenizer().tokenize(text.lower()):
            cleaned = re.sub(Globals.REGEX_CLEANER, "", word)
            if cleaned and cleaned not in Globals.STOP_WORDS and cleaned not in string.punctuation:
                token_list.append(cleaned)

        # POS Tagging and filtering
        if token_list:
            pos_tags = nltk.pos_tag(token_list)
            filtered_tokens = [
                token for token, tag in pos_tags
                if tag in Globals.VALID_TAGS and token not in Globals.STOP_WORDS
            ]
        else:
            filtered_tokens = []

        # Count frequencies
        token_counts = {}
        for token in filtered_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Normalize
        total_tokens = len(filtered_tokens)
        normalized = {}
        for token, count in token_counts.items():
            normalized[token] = (count, count / total_tokens if total_tokens > 0 else 0)

        return normalized

    def _calculate_idf(self) -> None:
        """Calculate IDF for all terms in the corpus."""
        N = len(self.documents)

        # Count documents containing each term
        doc_freq = defaultdict(int)  # term -> number of documents containing it

        for doc in self.documents:
            unique_terms = set(doc.term_frequencies.keys())
            for term in unique_terms:
                doc_freq[term] += 1

        # Calculate IDF for each term
        for term, df in doc_freq.items():
            # Using smoothed IDF: log((N + 1) / (df + 1)) + 1
            self.idf_cache[term] = math.log((N + 1) / (df + 1)) + 1

    def _calculate_tfidf(self, term: str, doc_tf: float) -> float:
        """Calculate TF-IDF score for a term in a document."""
        idf = self.idf_cache.get(term, 0)
        return doc_tf * idf * 100  # Scale factor from the lecture

    def search_documents(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search documents using TF-IDF similarity to query.
        Returns list of (document_name, score) sorted by relevance.
        """
        if not self._is_loaded:
            self.load_documents()

        # Process query into terms
        query_terms = self._process_query(query)
        if not query_terms:
            return []

        # Calculate scores for each document
        scores = []
        for i, doc in enumerate(self.documents):
            score = 0.0

            # For each query term, add its TF-IDF in this document
            for term in query_terms:
                if term in doc.term_frequencies:
                    tf = doc.term_frequencies[term]
                    score += self._calculate_tfidf(term, tf)

            # Normalize by query length (optional)
            if query_terms:
                score /= len(query_terms)

            if score > 0:
                scores.append((self.doc_names[i], score))

        # Sort by score (highest first) and return top K
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _process_query(self, query: str) -> List[str]:
        """Process query string into relevant terms."""
        if not query.strip():
            return []

        # Same processing as documents
        token_list = []
        for word in nltk.WordPunctTokenizer().tokenize(query.lower()):
            cleaned = re.sub(Globals.REGEX_CLEANER, "", word)
            if cleaned and cleaned not in Globals.STOP_WORDS and cleaned not in string.punctuation:
                token_list.append(cleaned)

        if token_list:
            pos_tags = nltk.pos_tag(token_list)
            filtered_tokens = [
                token for token, tag in pos_tags
                if tag in Globals.VALID_TAGS and token not in Globals.STOP_WORDS
            ]
            return filtered_tokens
        return []

    def extract_keywords(self, doc_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract top keywords from a specific document using TF-IDF.
        Returns list of (term, tfidf_score) sorted by score.
        """
        if not self._is_loaded:
            self.load_documents()

        # Find the document
        doc_index = None
        for i, name in enumerate(self.doc_names):
            if name == doc_name:
                doc_index = i
                break

        if doc_index is None:
            return []

        doc = self.documents[doc_index]

        # Calculate TF-IDF for all terms in this document
        term_scores = []
        for term, tf in doc.term_frequencies.items():
            tfidf = self._calculate_tfidf(term, tf)
            term_scores.append((term, tfidf))

        # Sort by TF-IDF score (highest first)
        term_scores.sort(key=lambda x: x[1], reverse=True)
        return term_scores[:top_k]

    def run(self, query: str) -> List[SearchResult]:
        """
        Implementation of the KeywordMethod protocol.
        Searches documents and returns ranked results.
        """
        # Load documents if not already loaded
        if not self._is_loaded:
            try:
                self.load_documents()
            except Exception as e:
                return [SearchResult(
                    title=f"Error loading documents: {e}",
                    score=0.0,
                    details="Please check data directory configuration."
                )]

        # Search for documents relevant to query
        search_results = self.search_documents(query, top_k=10)

        # Convert to SearchResult objects
        results = []
        for i, (doc_name, score) in enumerate(search_results, 1):
            # Extract top keywords for this document
            keywords = self.extract_keywords(doc_name, top_k=5)
            keyword_str = ", ".join([term for term, _ in keywords])

            result = SearchResult(
                title=f"{doc_name} (Score: {score:.4f})",
                score=score,
                details=f"Top keywords: {keyword_str}\nPath: {self.documents[i - 1].path}"
            )
            results.append(result)

        # If no results, provide some feedback
        if not results and query.strip():
            # Try keyword extraction mode instead
            return self._keyword_extraction_mode(query)

        return results

    def _keyword_extraction_mode(self, query: str) -> List[SearchResult]:
        """
        Alternative mode: treat query as document and extract keywords from it.
        """
        # Process the query as if it were a document
        query_terms = self._process_query(query)
        if not query_terms:
            return []

        # Simple TF calculation for query
        term_counts = {}
        for term in query_terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        total_terms = len(query_terms)
        results = []
        for term, count in term_counts.items():
            tf = count / total_terms if total_terms > 0 else 0
            idf = self.idf_cache.get(term, 1.0)  # Default IDF if term not in corpus
            tfidf = tf * idf * 100

            results.append(SearchResult(
                title=f"Keyword: '{term}'",
                score=tfidf,
                details=f"TF: {tf:.4f}, IDF: {idf:.4f}, TF-IDF: {tfidf:.4f}\n"
                        f"Appears in {count} of {total_terms} query terms."
            ))

        # Sort by TF-IDF score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:10]  # Return top 10 keywords


def create_tfidf_method() -> KeywordMethod:
    """Factory function to create and initialize TF-IDF method."""
    # Use the exact path
    data_dir = r"E:\DigiPenMasterCourse\Semester4_2026\cs592_NLP\cs592-natural-language-processing\Chankasemporn_Ju-ve_CS592_NLP_Project\data\train"

    tfidf = TfIdfMethod(
        name="TF-IDF (Document Search)",
        data_dir=data_dir  # Pass the exact path
    )

    # Pre-load documents
    try:
        tfidf.load_documents()
    except Exception as e:
        print(f"Warning: Could not load documents: {e}")
        print(f"Data directory being used: {data_dir}")
        print("TF-IDF will load documents on first search.")

    return tfidf