"""
File Name:    BM25_Method.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from typing import Dict, List, Tuple, Optional
import math

from .KeywordMethod import SearchResult, KeywordMethod
from .DocumentProcessor import DocumentProcessor


def create_bm25_method(processor: DocumentProcessor = None) -> KeywordMethod:
    """Factory function to create and initialize BM25 method."""
    bm25 = BM25Method(
        name="BM25 (Document Search)",
        processor=processor
    )
    bm25.preprocess()
    return bm25


class BM25Method(KeywordMethod):
    """
    BM25 implementation for document search + keyword extraction.
    - preprocess(): computes document frequencies, IDF, avg doc length
    - run(query): ranks documents by BM25 score
    - extract_keywords(doc_name): returns top terms by BM25-ish term contribution
    """

    def __init__(
        self,
        name: str = "BM25",
        processor: Optional[DocumentProcessor] = None,
        k1: float = 1.5,
        b: float = 0.75,
        use_plus_idf: bool = True,  # more stable if df ~ N
    ):
        super().__init__(name=name)
        self.processor = processor or DocumentProcessor()

        self.k1 = float(k1)
        self.b = float(b)
        self.use_plus_idf = bool(use_plus_idf)

        self.idf_cache: Dict[str, float] = {}
        self.avg_doc_len: float = 0.0
        self.doc_len: Dict[str, int] = {}  # doc filename -> length in tokens

    # ---------------- Public API ----------------

    def preprocess(self) -> None:
        """Compute IDF cache + average document length."""
        self._calculate_idf_and_lengths()

    def run(self, query: str) -> List[SearchResult]:
        """Rank documents for a query using BM25."""
        bm25_results = self.get_bm25_per_document(query, top_k=10)

        results: List[SearchResult] = []
        for doc_name, score in bm25_results:
            keywords = self.extract_keywords(doc_name, top_k=5)
            keyword_str = ", ".join([term for term, _ in keywords])

            doc = self.processor.get_document_by_name(doc_name)
            results.append(SearchResult(
                title=f"{doc_name} (Score: {score:.4f})",
                score=float(score),
                details=f"Top keywords: {keyword_str}\nPath: {doc.path}"
            ))

        if not results and query.strip():
            print("No results found.")

        return results

    def get_bm25_per_document(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns list of (document_name, bm25_score) sorted by relevance.
        """
        query_terms = self.processor.tokenizer.process_query(query)
        if not query_terms:
            return []

        scores: List[Tuple[str, float]] = []
        for doc in self.processor.documents:
            dl = self.doc_len.get(doc.filename, 0)
            if dl <= 0:
                continue

            score = 0.0
            for term in query_terms:
                tf = doc.term_frequencies.get(term, 0.0)
                if tf <= 0:
                    continue
                score += self._bm25_term_score(term=term, tf=tf, dl=dl)

            # Optional: normalize by query length (matches your TF-IDF style)
            if query_terms:
                score /= len(query_terms)

            if score > 0:
                scores.append((doc.filename, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def extract_keywords(self, doc_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract top keywords from a specific document using BM25-like term contributions.
        Returns list of (term, bm25_term_score) sorted by score.
        """
        try:
            doc = self.processor.get_document_by_name(doc_name)
        except ValueError:
            return []

        dl = self.doc_len.get(doc.filename, 0)
        if dl <= 0:
            return []

        term_scores: List[Tuple[str, float]] = []
        for term, tf in doc.term_frequencies.items():
            if tf <= 0:
                continue
            term_scores.append((term, self._bm25_term_score(term=term, tf=tf, dl=dl)))

        term_scores.sort(key=lambda x: x[1], reverse=True)
        return term_scores[:top_k]

    # ---------------- Internals ----------------

    def _calculate_idf_and_lengths(self) -> None:
        N = self.processor.document_count
        if N == 0:
            self.avg_doc_len = 0.0
            self.idf_cache = {}
            self.doc_len = {}
            return

        # Use the same DF function you already have for TF-IDF
        doc_freq = self.processor.get_document_frequencies()  # term -> df

        # doc lengths: sum of term frequencies (works if term_frequencies are counts)
        total_len = 0
        self.doc_len = {}
        for doc in self.processor.documents:
            dl = int(sum(doc.term_frequencies.values()))
            self.doc_len[doc.filename] = dl
            total_len += dl

        self.avg_doc_len = (total_len / max(N, 1)) if total_len > 0 else 0.0

        # BM25 IDF
        # Standard: log( (N - df + 0.5) / (df + 0.5) )
        # "plus" variant: log(1 + (N - df + 0.5)/(df + 0.5)) keeps it positive
        self.idf_cache = {}
        for term, df in doc_freq.items():
            if self.use_plus_idf:
                self.idf_cache[term] = math.log(
                    1.0 + (N - df + 0.5) / (df + 0.5)
                )
            else:
                self.idf_cache[term] = math.log(
                    (N - df + 0.5) / (df + 0.5)
                )

    def _bm25_term_score(self, term: str, tf: float, dl: int) -> float:
        """
        BM25 term contribution:
          idf(t) * ( tf*(k1+1) / ( tf + k1*(1 - b + b*(dl/avgdl)) ) )
        """
        idf = self.idf_cache.get(term, 0.0)
        if idf <= 0.0:
            return 0.0

        if self.avg_doc_len <= 0.0:
            return idf  # degenerate fallback

        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
        if denom <= 0:
            return 0.0

        return idf * (tf * (self.k1 + 1.0) / denom)
