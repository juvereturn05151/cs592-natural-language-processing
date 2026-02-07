"""
File Name:    TF_IDF_Method.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from typing import Dict, List, Tuple
import math
from .KeywordMethod import SearchResult, KeywordMethod
from .DocumentProcessor import DocumentProcessor


def create_tfidf_method(processor: DocumentProcessor = None) -> KeywordMethod:
    """Factory function to create and initialize TF-IDF method."""
    tfidf = TfIdfMethod(
        name="TF-IDF (Document Search)",
        processor=processor
    )

    tfidf.preprocess()

    return tfidf


class TfIdfMethod:
    """
    TF-IDF implementation for keyword extraction and document search. Uses DocumentProcessor for document handling.
    """

    def __init__(self, name: str = "TF-IDF", processor: DocumentProcessor = None):
        self.name = name
        self.processor = processor
        self.idf_cache: Dict[str, float] = {}

    def run(self, query: str) -> List[SearchResult]:
        """
        Implementation of the KeywordMethod protocol. Searches documents and returns ranked results.
        """

        #search for documents relevant to query
        tf_idf_per_document_results = self.get_tf_idf_per_document(query, top_k=10)

        #convert to SearchResult objects
        results = []
        for i, (doc_name, score) in enumerate(tf_idf_per_document_results, 1):
            #extract top keywords for this document
            keywords = self.extract_keywords(doc_name, top_k=5)
            keyword_str = ", ".join([term for term, _ in keywords])

            #find document for path
            doc = self.processor.get_document_by_name(doc_name)

            result = SearchResult(
                title=f"{doc_name} (Score: {score:.4f})",
                score=score,
                details=f"Top keywords: {keyword_str}\nPath: {doc.path}"
            )
            results.append(result)

        # If no results, provide some feedback
        if not results and query.strip():
            print("No results found.")

        return results

    def preprocess(self) -> None:
        """Load and process all documents using DocumentProcessor."""
        self._calculate_idf()

    def _calculate_idf(self) -> None:
        """Calculate IDF for all terms in the corpus."""
        N = self.processor.document_count

        if N == 0:
            return

        doc_freq = self.processor.get_document_frequencies()

        #get document containing term t
        #calculate IDF for each term
        for term, df in doc_freq.items():
            self.idf_cache[term] = math.log((N + 1) / (df + 1)) + 1

    def _calculate_tfidf(self, term: str, doc_tf: float) -> float:
        """Calculate TF-IDF score for a term in a document."""
        idf = self.idf_cache.get(term, 0)
        return doc_tf * idf * 100  # Scale factor from the lecture

    def get_tf_idf_per_document(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search documents using TF-IDF similarity to query.
        Returns list of (document_name, score) sorted by relevance.
        """

        #process query using tokenizer
        query_terms = self.processor.tokenizer.process_query(query)
        if not query_terms:
            return []

        #calculate scores for each document
        scores = []
        for doc in self.processor.documents:
            score = 0.0

            #for each query term, add its TF-IDF in this document
            for term in query_terms:
                if term in doc.term_frequencies:
                    tf = doc.term_frequencies[term]
                    score += self._calculate_tfidf(term, tf)

            # Normalize by query length
            if query_terms:
                score /= len(query_terms)

            if score > 0:
                scores.append((doc.filename, score))

        #sort by score (highest first) and return top K
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def extract_keywords(self, doc_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract top keywords from a specific document using TF-IDF.
        Returns list of (term, tfidf_score) sorted by score.
        """

        try:
            doc = self.processor.get_document_by_name(doc_name)
        except ValueError:
            return []

        #calculate TF-IDF for all terms in this document
        term_scores = []
        for term, tf in doc.term_frequencies.items():
            tfidf = self._calculate_tfidf(term, tf)
            term_scores.append((term, tfidf))

        #sort by TF-IDF score (highest first)
        term_scores.sort(key=lambda x: x[1], reverse=True)
        return term_scores[:top_k]

