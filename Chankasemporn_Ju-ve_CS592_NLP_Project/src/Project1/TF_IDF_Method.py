"""
File Name:    TF_IDF_Method.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from typing import Dict, List, Tuple
import math
from .KeywordMethod import SearchResult, KeywordMethod
import src.NLP_Globals as Globals
from .DocumentProcessor import DocumentProcessor, DocumentData


def create_tfidf_method() -> KeywordMethod:
    """Factory function to create and initialize TF-IDF method."""
    data_dir = Globals.get_default_data_dir()

    tfidf = TfIdfMethod(
        name="TF-IDF (Document Search)",
        data_dir=data_dir
    )

    try:
        tfidf.load_documents()
    except Exception as e:
        print(f"Warning: Could not load documents: {e}")
        print(f"Data directory being used: {data_dir}")
        print("TF-IDF will load documents on first search.")

    return tfidf


class TfIdfMethod:
    """
    TF-IDF implementation for keyword extraction and document search.
    Uses DocumentProcessor for document handling.
    """

    def __init__(self, name: str = "TF-IDF", data_dir: str = None):
        self.name = name
        self.data_dir = data_dir or Globals.get_default_data_dir()
        self.processor = DocumentProcessor()
        self.idf_cache: Dict[str, float] = {}
        self._is_loaded = False

    def load_documents(self, file_pattern: str = "*.txt") -> None:
        """
        Load and process all documents using DocumentProcessor.
        """
        if self._is_loaded:
            return

        self.processor.load_from_directory(self.data_dir, file_pattern)
        self._calculate_idf()
        self._is_loaded = True

    def _calculate_idf(self) -> None:
        """Calculate IDF for all terms in the corpus."""
        N = self.processor.document_count

        if N == 0:
            return

        # Get document frequencies from processor
        doc_freq = self.processor.get_document_frequencies()

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

        # Process query using tokenizer
        query_terms = self.processor.tokenizer.process_query(query)
        if not query_terms:
            return []

        # Calculate scores for each document
        scores = []
        for doc in self.processor.documents:
            score = 0.0

            # For each query term, add its TF-IDF in this document
            for term in query_terms:
                if term in doc.term_frequencies:
                    tf = doc.term_frequencies[term]
                    score += self._calculate_tfidf(term, tf)

            # Normalize by query length
            if query_terms:
                score /= len(query_terms)

            if score > 0:
                scores.append((doc.filename, score))

        # Sort by score (highest first) and return top K
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def extract_keywords(self, doc_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract top keywords from a specific document using TF-IDF.
        Returns list of (term, tfidf_score) sorted by score.
        """
        if not self._is_loaded:
            self.load_documents()

        try:
            doc = self.processor.get_document_by_name(doc_name)
        except ValueError:
            return []

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

            # Find document for path
            doc = self.processor.get_document_by_name(doc_name)

            result = SearchResult(
                title=f"{doc_name} (Score: {score:.4f})",
                score=score,
                details=f"Top keywords: {keyword_str}\nPath: {doc.path}"
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
        tokenized_query = self.processor.tokenizer.process_text(query, "query")

        if not tokenized_query.tokens:
            return []

        # Calculate TF-IDF scores for query terms
        scores = {}
        total_terms = len(tokenized_query.tokens)

        for term, count in tokenized_query.raw_counts.items():
            tf = count / total_terms if total_terms > 0 else 0
            idf = self.idf_cache.get(term, 1.0)  # Default IDF if term not in corpus
            tfidf = tf * idf * 100
            scores[term] = tfidf

        # Convert to SearchResult objects
        results = []
        for term, tfidf in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]:
            count = tokenized_query.raw_counts[term]
            tf = count / total_terms

            results.append(SearchResult(
                title=f"Keyword: '{term}'",
                score=tfidf,
                details=f"TF: {tf:.4f}, IDF: {self.idf_cache.get(term, 1.0):.4f}, "
                        f"TF-IDF: {tfidf:.4f}\n"
                        f"Appears {count} times in query."
            ))
        return results

    def add_document(self, file_path: str) -> None:
        """
        Add a single document to the corpus and update IDF.
        """
        doc = self.processor.load_single_document(file_path)
        self._recalculate_idf()

    def _recalculate_idf(self) -> None:
        """Recalculate IDF after corpus changes."""
        self._calculate_idf()