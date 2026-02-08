"""
File Name:    BM25_Method.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
import time
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

    def __init__(self, name: str = "BM25", processor: Optional[DocumentProcessor] = None,
                 k1: float = 1.5, b: float = 0.75):
        self.name = name
        self.processor = processor or DocumentProcessor()

        self.k1 = float(k1)
        self.b = float(b)

        #BM25 state
        self.idf: Dict[str, float] = {}

        # incremental bookkeeping
        # document frequency per term
        self._df: Dict[str, int] = {}
        self._doc_terms: Dict[str, set] = {}

        self._N: int = 0
        self._total_len: int = 0
        self.avg_doc_len: float = 0.0

        self.doc_len: Dict[str, int] = {}

        if self.processor is not None:
            self.processor.add_document_added_listener(self._on_document_changed)

    def preprocess(self) -> None:
        """Compute IDF + average document length."""
        start = time.perf_counter()
        self._calculate_idf_and_lengths()
        elapsed = time.perf_counter() - start
        print(f"[BM25] Preprocess completed in {elapsed:.4f} seconds")

    def _calculate_idf_and_lengths(self) -> None:
        self.idf.clear()
        self._df.clear()
        self._doc_terms.clear()
        self.doc_len.clear()

        self._N = self.processor.document_count
        if self._N == 0:
            self.avg_doc_len = 0.0
            self._total_len = 0
            return

        doc_freq = self.processor.get_document_frequencies()
        self._df.update(doc_freq)

        self._total_len = 0
        for doc in self.processor.documents:
            dl = int(sum(doc.term_frequencies.values()))
            self.doc_len[doc.filename] = dl
            self._total_len += dl
            self._doc_terms[doc.filename] = set(doc.unique_terms)

        self.avg_doc_len = self._total_len / max(self._N, 1)

        for term, df in self._df.items():
            self.idf[term] = math.log(
                1.0 + (self._N - df + 0.5) / (df + 0.5)
            )

    def run(self, query: str) -> List[SearchResult]:
        """Rank documents for a query using BM25."""
        start = time.perf_counter()
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
        elapsed = time.perf_counter() - start
        print(
            f"[BM25] Run | query_len={len(query.split())} | "
            f"time={elapsed:.4f}s"
        )

        return results

    def get_bm25_per_document(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Returns list of (document_name, bm25_score) sorted by relevance."""
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

            if score > 0:
                scores.append((doc.filename, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _bm25_term_score(self, term: str, tf: float, dl: int) -> float:
        """BM25 term contribution: idf(t) * ( tf*(k1+1) / ( tf + k1*(1 - b + b*(dl/avgdl)) ) )"""
        idf = self.idf.get(term, 0.0)
        if idf <= 0.0:
            return 0.0

        if self.avg_doc_len <= 0.0:
            return idf

        denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
        if denominator <= 0:
            return 0.0

        return idf * (tf * (self.k1 + 1.0) / denominator)

    def _on_document_changed(self, doc_data, action: str) -> None:
        start = time.perf_counter()

        if action == "added":
            self._incremental_add(doc_data)
        elif action == "replaced":
            self._incremental_replace(doc_data)
        else:
            self.preprocess()
        elapsed = time.perf_counter() - start
        print(
            f"[BM25] Document {action}: {doc_data.filename} | "
            f"time = {elapsed:.4f}s"
        )

    def _incremental_add(self, doc_data) -> None:
        filename = doc_data.filename

        if filename in self._doc_terms:
            self._incremental_replace(doc_data)
            return

        new_terms = set(doc_data.unique_terms)
        dl = int(sum(doc_data.term_frequencies.values()))

        self._N += 1
        self._total_len += dl

        self.doc_len[filename] = dl
        self._doc_terms[filename] = new_terms

        for term in new_terms:
            self._df[term] = self._df.get(term, 0) + 1

        self.avg_doc_len = self._total_len / max(self._N, 1)

        self._recompute_idf_for_terms(new_terms)

    def _incremental_replace(self, doc_data) -> None:
        filename = doc_data.filename

        old_terms = self._doc_terms.get(filename)
        if old_terms is None:
            self.preprocess()
            return

        old_dl = self.doc_len.get(filename, 0)

        new_terms = set(doc_data.unique_terms)
        new_dl = int(sum(doc_data.term_frequencies.values()))

        # update total length
        self._total_len += (new_dl - old_dl)

        self.doc_len[filename] = new_dl
        self._doc_terms[filename] = new_terms

        for term in old_terms:
            self._df[term] -= 1

        for term in new_terms:
            self._df[term] = self._df.get(term, 0) + 1

        self.avg_doc_len = self._total_len / max(self._N, 1)

        changed_terms = old_terms.union(new_terms)
        self._recompute_idf_for_terms(changed_terms)

    def _recompute_idf_for_terms(self, terms: set) -> None:
        for term in terms:
            df = self._df.get(term, 0)
            if df <= 0:
                self._df.pop(term, None)
                self.idf.pop(term, None)
            else:
                self.idf[term] = math.log(
                    1.0 + (self._N - df + 0.5) / (df + 0.5)
                )

    def extract_keywords(self, doc_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract top keywords from a specific document using BM25-like term contributions. Returns list of (term, bm25_term_score) sorted by score."""
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