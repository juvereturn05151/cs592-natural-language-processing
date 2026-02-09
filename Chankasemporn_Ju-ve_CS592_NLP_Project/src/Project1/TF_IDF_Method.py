"""
File Name:    TF_IDF_Method.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
import time
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
    """TF-IDF implementation for keyword extraction and document search. Uses DocumentProcessor for document handling."""

    def __init__(self, name: str = "TF-IDF", processor: DocumentProcessor = None):
        self.name = name
        self.processor = processor

        # calculate inverse document frequency for each term
        self.idf: Dict[str, float] = {}

        # incremental bookkeeping
        # document frequency per term
        self._df: Dict[str, int] = {}
        # number of docs
        self._N: int = 0
        # filename -> unique_terms snapshot
        self._doc_terms: Dict[str, set] = {}

        if self.processor is not None:
            self.processor.add_document_added_listener(self._on_document_changed)

    def preprocess(self) -> None:
        """Load and process all documents using DocumentProcessor."""
        start = time.perf_counter()
        self._calculate_idf()
        elapsed = time.perf_counter() - start
        print(f"[TF-IDF] Preprocess completed in {elapsed:.4f} seconds")

    def _calculate_idf(self) -> None:
        self.idf.clear()
        self._df.clear()
        self._doc_terms.clear()

        self._N = self.processor.document_count
        if self._N == 0:
            return

        # build DF from processor
        doc_freq = self.processor.get_document_frequencies()  # term -> df
        self._df.update(doc_freq)

        # snapshot each document's unique terms (needed for replace deltas)
        for doc in self.processor.documents:
            self._doc_terms[doc.filename] = set(doc.unique_terms)

        # compute IDF from DF
        for term, df in self._df.items():
            self.idf[term] = math.log((self._N + 1) / (df + 1)) + 1

    def _calculate_tfidf(self, term: str, doc_tf: float) -> float:
        """Calculate TF-IDF score for a term in a document."""
        idf = self.idf.get(term, 0)
        return doc_tf * idf * 100

    def run(self, query: str) -> List[SearchResult]:
        """Implementation of the KeywordMethod protocol. Searches documents and returns ranked results."""
        start = time.perf_counter()

        if not query or not query.strip():
            return [SearchResult(title="Empty query", score=0.0, details="Please type a query.")]

        # Tokenize query ONCE (this must match what scoring uses)
        query_terms = self.processor.tokenizer.process_query(query)
        if not query_terms:
            return [SearchResult(title="Empty query", score=0.0, details="No valid query terms after tokenization.")]

        # Search for documents relevant to query
        tf_idf_per_document_results = self.get_tf_idf_per_document(query, top_k=10)

        results: List[SearchResult] = []
        for i, (doc_name, score) in enumerate(tf_idf_per_document_results, 1):
            # Extract top keywords for this document
            keywords = self.extract_keywords(doc_name, top_k=5)
            keyword_str = ", ".join([term for term, _ in keywords])

            # Find document for path + length + counts
            doc = self.processor.get_document_by_name(doc_name)

            # Document length (prefer tokens; fallback to text split)
            if hasattr(doc, "tokens") and doc.tokens is not None:
                doc_len = len(doc.tokens)
            elif hasattr(doc, "stemmed_tokens") and doc.stemmed_tokens is not None:
                doc_len = len(doc.stemmed_tokens)
            else:
                doc_len = len(doc.text.split()) if getattr(doc, "text", "") else 0

            # Query word COUNTS in this document (e.g., "holm × 37")
            # Prefer raw_counts if available; else approximate from term_frequencies * doc_len
            counts_parts = []
            for term in query_terms:
                if hasattr(doc, "raw_counts") and isinstance(doc.raw_counts, dict):
                    c = int(doc.raw_counts.get(term, 0))
                else:
                    # term_frequencies looks like normalized TF; approximate count using doc_len
                    c = int(round(float(doc.term_frequencies.get(term, 0.0)) * float(doc_len)))

                if c > 0:
                    counts_parts.append(f"{term} × {c}")

            query_counts_str = ", ".join(counts_parts) if counts_parts else "None"

            result = SearchResult(
                title=f"{doc_name} (Score: {score:.4f})",
                score=score,
                details=(
                    f"Top keywords: {keyword_str}\n"
                    f"Document length: {doc_len} tokens\n"
                    f"Query word counts: {query_counts_str}\n"
                    f"Path: {doc.path}"
                )
            )
            results.append(result)

        if not results and query.strip():
            print("No results found.")

        elapsed = time.perf_counter() - start
        print(
            f"[TF-IDF] Run | query_len={len(query_terms)} | "
            f"time={elapsed:.4f}s"
        )

        return results

    def refresh(self) -> None:
        """Recompute IDF after the processor changes (documents added/removed)."""
        self.idf.clear()
        self._calculate_idf()

    def get_tf_idf_per_document(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search documents using TF-IDF similarity to query. Returns list of (document_name, score) sorted by relevance."""

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
        """Extract top keywords from a specific document using TF-IDF.Returns list of (term, tfidf_score) sorted by score."""

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

    def _on_document_changed(self, doc_data, action: str) -> None:
        """Called automatically when DocumentProcessor.load_one_file emits. action is 'added' or 'replaced'"""
        start = time.perf_counter()

        if action == "added":
            self._incremental_add(doc_data)
        elif action == "replaced":
            self._incremental_replace(doc_data)
        else:
            self.refresh()

        elapsed = time.perf_counter() - start
        print(
            f"[TF-IDF] Document {action}: {doc_data.filename} | "
            f"time = {elapsed:.4f}s"
        )

    def _recompute_idf_for_terms(self, terms: set) -> None:
        """Recompute IDF only for the given terms."""
        for term in terms:
            df = self._df.get(term, 0)
            if df <= 0:
                # term no longer exists in any doc
                self.idf.pop(term, None)
                self._df.pop(term, None)
            else:
                self.idf[term] = math.log((self._N + 1) / (df + 1)) + 1

    def _incremental_add(self, doc_data) -> None:
        filename = doc_data.filename

        #if the same file is "added" again, treat as replace
        if filename in self._doc_terms:
            self._incremental_replace(doc_data)
            return

        self._N += 1
        new_terms = set(doc_data.unique_terms)
        self._doc_terms[filename] = new_terms

        #update DF counts for terms in the new doc
        for term in new_terms:
            self._df[term] = self._df.get(term, 0) + 1

        self._recompute_idf_for_terms(new_terms)

    def _incremental_replace(self, doc_data) -> None:
        filename = doc_data.filename

        old_terms = self._doc_terms.get(filename)
        new_terms = set(doc_data.unique_terms)

        # If we don't know old terms, safest fallback
        if old_terms is None:
            self.refresh()
            return

        self._doc_terms[filename] = new_terms

        # decrement DF for terms that were in old doc
        for term in old_terms:
            self._df[term] = self._df.get(term, 0) - 1

        # increment DF for terms that are in new doc
        for term in new_terms:
            self._df[term] = self._df.get(term, 0) + 1

        changed_terms = old_terms.union(new_terms)

        # N didn't change on replace, so we only need to recompute changed terms
        self._recompute_idf_for_terms(changed_terms)
