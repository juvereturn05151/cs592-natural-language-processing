"""
File Name:    Yake_Method.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
import math
import re
import os
import time

from .KeywordMethod import SearchResult, KeywordMethod
from .DocumentProcessor import DocumentProcessor
import src.NLP_Globals as Globals


def create_yake_method(processor: DocumentProcessor = None) -> KeywordMethod:
    """Factory function to create and initialize YAKE method."""
    yake = YakeMethod(
        name="YAKE (Keyword Extraction / Document Search)",
        processor=processor
    )
    yake.preprocess()
    return yake


class YakeMethod(KeywordMethod):
    def __init__(self, name: str = "YAKE", processor: Optional[DocumentProcessor] = None):
        super().__init__(name=name)
        self.processor = processor or DocumentProcessor()

        self.stopwords: Set[str] = Globals.STOP_WORDS
        self._regex_nonword = Globals.REGEX_NONWORD

        # doc -> top-k keyword -> score (for extraction)
        self._doc_kw_scores: Dict[str, Dict[str, float]] = {}
        # doc -> ALL candidate ngram -> score (for search)
        self._doc_all_phrase_scores: Dict[str, Dict[str, float]] = {}
        # doc -> set(ALL candidate ngrams)
        self._doc_phrase_set: Dict[str, Set[str]] = {}
        # doc -> set(words appearing in candidate ngrams)
        self._doc_word_set: Dict[str, Set[str]] = {}

        if self.processor is not None and hasattr(self.processor, "add_document_added_listener"):
            self.processor.add_document_added_listener(self._on_document_changed)  # type: ignore

    def preprocess(self) -> None:
        start = time.perf_counter()

        # Clear caches so re-preprocess doesn't keep stale docs
        self._doc_kw_scores.clear()
        self._doc_all_phrase_scores.clear()
        self._doc_phrase_set.clear()
        self._doc_word_set.clear()

        docs_processed = 0
        total_unique_candidates = 0  # debug metric: unique ngrams across docs

        for doc in self.processor.documents:
            self._process_one_doc(doc)
            docs_processed += 1
            total_unique_candidates += len(self._doc_all_phrase_scores.get(doc.filename, {}))

        elapsed = time.perf_counter() - start
        print(
            f"[YAKE] Preprocess | docs={docs_processed} | "
            f"total_unique_candidates={total_unique_candidates} | time={elapsed:.4f}s"
        )

    def run(self, query: str) -> List[SearchResult]:
        start = time.perf_counter()

        if not query or not query.strip():
            return [SearchResult(title="Empty query", score=0.0, details="Please type a query.")]

        # For SEARCH matching, use ALL query candidates (not only top-k).
        _, qdbg = self._extract_yake_keywords(query, top_k=200, max_n=3)
        query_phrase_scores_all: Dict[str, float] = qdbg["phrase_scores"]
        query_phrase_set = set(query_phrase_scores_all.keys())

        # Build query word set (words from the extracted query phrases)
        query_word_set: Set[str] = set()
        for p in query_phrase_set:
            query_word_set.update(p.split())

        # Also include raw query tokens so single-word queries still work well.
        # (Keeps behavior consistent with your RAKE change request.)
        raw_q_words = []
        for w in query.lower().split():
            w = self._regex_nonword.sub("", w)
            if w and (w not in self.stopwords):
                raw_q_words.append(w)
        query_word_set.update(raw_q_words)

        # Store extra info so we can print "phrases containing any query word"
        # doc_name, total_score, matched_exact_str, contains_word_phrases, word_overlap
        results: List[Tuple[str, float, str, List[Tuple[str, float]], Set[str]]] = []
        docs_scored = 0

        for doc in self.processor.documents:
            doc_name = doc.filename

            doc_phrase_scores = self._doc_all_phrase_scores.get(doc_name, {})
            doc_phrases = self._doc_phrase_set.get(doc_name, set())
            doc_words = self._doc_word_set.get(doc_name, set())

            if not doc_phrase_scores:
                continue

            docs_scored += 1
            overlap = query_phrase_set.intersection(doc_phrases)

            # scoring
            # YAKE lower-is-better. Convert to stable higher-is-better.
            kw_score = 0.0
            for kw in overlap:
                s = doc_phrase_scores.get(kw, 1.0)
                kw_score += -math.log(s + 1e-12)

            # Word overlap bonus (helps when phrases differ but single words match)
            word_overlap = query_word_set.intersection(doc_words)
            word_bonus = len(word_overlap) * 0.10

            total = kw_score + word_bonus
            if total <= 0:
                continue

            matched_sorted = sorted(
                [(kw, doc_phrase_scores.get(kw, 1.0)) for kw in overlap],
                # lower YAKE score first (better)
                key=lambda x: x[1]
            )[:5]
            matched_str = ", ".join([f"'{kw}' (yake={s:.3e})" for kw, s in matched_sorted]) or "No exact keyword match"

            # NEW: phrases that contain ANY query word (phrase != query phrase is ok)
            # We'll show best phrases by YAKE score (lower is better).
            contains_word_phrases: List[Tuple[str, float]] = []
            if query_word_set:
                for p, s in doc_phrase_scores.items():
                    if set(p.split()).intersection(query_word_set):
                        contains_word_phrases.append((p, s))
                contains_word_phrases.sort(key=lambda x: x[1])  # best first (lowest YAKE)
                contains_word_phrases = contains_word_phrases[:10]

            results.append((doc_name, total, matched_str, contains_word_phrases, word_overlap))

        results.sort(key=lambda x: x[1], reverse=True)

        def _doc_token_len(doc_obj) -> int:
            """Best-effort document length in tokens."""
            if doc_obj is None:
                return 0
            if hasattr(doc_obj, "tokens") and doc_obj.tokens is not None:
                return len(doc_obj.tokens)
            if hasattr(doc_obj, "stemmed_tokens") and doc_obj.stemmed_tokens is not None:
                return len(doc_obj.stemmed_tokens)
            text = getattr(doc_obj, "text", "") or ""
            return len(text.split()) if text else 0

        def _query_word_counts_in_doc(doc_obj, words: Set[str]) -> Dict[str, int]:
            """Best-effort raw counts of each query word in a document."""
            if doc_obj is None or not words:
                return {}

            # Best case: raw_counts already exists
            raw_counts = getattr(doc_obj, "raw_counts", None)
            if isinstance(raw_counts, dict):
                return {w: int(raw_counts.get(w, 0)) for w in words}

            # Fallback: count from tokens/stemmed_tokens
            token_list = None
            if hasattr(doc_obj, "tokens") and doc_obj.tokens is not None:
                token_list = doc_obj.tokens
            elif hasattr(doc_obj, "stemmed_tokens") and doc_obj.stemmed_tokens is not None:
                token_list = doc_obj.stemmed_tokens

            if token_list is None:
                text = getattr(doc_obj, "text", "") or ""
                token_list = text.split() if text else []

            counts: Dict[str, int] = {w: 0 for w in words}
            for t in token_list:
                if t in counts:
                    counts[t] += 1
            return counts

        out: List[SearchResult] = []
        for doc_name, score, matched_str, contains_word_phrases, word_overlap in results[:10]:
            doc_obj = self.processor.get_document_by_name(doc_name)

            # Top 10 phrases overall in this document (best = lowest YAKE score)
            doc_phrase_scores = self._doc_all_phrase_scores.get(doc_name, {})
            top_phrases = sorted(doc_phrase_scores.items(), key=lambda x: x[1])[:10]
            top_phrases_str = ", ".join([f"'{p}' ({s:.3e})" for p, s in top_phrases]) or "N/A"

            contains_word_str = (
                ", ".join([f"'{p}' ({s:.3e})" for p, s in contains_word_phrases])
                if contains_word_phrases else "No phrases containing query words"
            )

            word_overlap_str = ", ".join(sorted(word_overlap)) if word_overlap else "None"

            # Document length
            doc_len = _doc_token_len(doc_obj)

            # Query word counts in this doc (e.g., "holm × 37")
            q_counts = _query_word_counts_in_doc(doc_obj, query_word_set)
            # Keep only words that actually appear, and show up to 10 most frequent
            present = [(w, c) for w, c in q_counts.items() if c > 0]
            present.sort(key=lambda x: x[1], reverse=True)
            query_counts_str = ", ".join([f"{w} × {c}" for w, c in present[:10]]) or "None"

            out.append(SearchResult(
                title=doc_name,
                score=float(score),
                details=(
                    f"Matched: {matched_str}\n"
                    f"Phrases containing any query word: {contains_word_str}\n"
                    f"Matched words: {word_overlap_str}\n"
                    f"Top phrases: {top_phrases_str}\n"
                    f"Document length: {doc_len} tokens\n"
                    f"Query word counts: {query_counts_str}\n"
                    f"Path: {doc_obj.path if doc_obj else 'N/A'}\n"
                    f"Score: sum(-log(doc_phrase_yake_score)) over matched phrases + 0.10 * |word_overlap|\n"
                    f"Note: higher score is better (we convert YAKE lower-is-better using -log)."
                )
            ))

        if not out:
            # keep UI behavior consistent: show docs even with no overlap
            for doc in self.processor.documents:
                doc_len = _doc_token_len(doc)
                q_counts = _query_word_counts_in_doc(doc, query_word_set)
                present = [(w, c) for w, c in q_counts.items() if c > 0]
                present.sort(key=lambda x: x[1], reverse=True)
                query_counts_str = ", ".join([f"{w} × {c}" for w, c in present[:10]]) or "None"

                # Also show phrases containing query words even if score==0
                doc_phrase_scores = self._doc_all_phrase_scores.get(doc.filename, {})
                contains_word_phrases: List[Tuple[str, float]] = []
                if query_word_set and doc_phrase_scores:
                    for p, s in doc_phrase_scores.items():
                        if set(p.split()).intersection(query_word_set):
                            contains_word_phrases.append((p, s))
                    contains_word_phrases.sort(key=lambda x: x[1])
                    contains_word_phrases = contains_word_phrases[:10]
                contains_word_str = (
                    ", ".join([f"'{p}' ({s:.3e})" for p, s in contains_word_phrases])
                    if contains_word_phrases else "No phrases containing query words"
                )

                out.append(SearchResult(
                    title=doc.filename,
                    score=0.0,
                    details=(
                        f"No YAKE keyword overlap with query.\n"
                        f"Phrases containing any query word: {contains_word_str}\n"
                        f"Document length: {doc_len} tokens\n"
                        f"Query word counts: {query_counts_str}\n"
                        f"Path: {doc.path}"
                    )
                ))

        elapsed = time.perf_counter() - start
        print(
            f"[YAKE] Run | query_len={len(query.split())} | "
            f"time={elapsed:.4f}s"
        )

        return out

    def extract_keywords(self, doc_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Returns (keyword/ngram, yake_score) sorted by YAKE score ascending (lower is better). This uses the cached top-k extraction list (not all candidates)."""
        scores = self._doc_kw_scores.get(doc_name, {})
        items = sorted(scores.items(), key=lambda x: x[1])
        return items[:top_k]

    def _split_into_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in Globals.SENT_SPLIT.split(text) if s.strip()]

    def _tokenize(self, text: str) -> List[str]:
        """
        Fast tokenizer:
          - lower
          - split
          - REGEX_NONWORD to strip junk
          - remove stopwords
        """
        out: List[str] = []
        for w in text.lower().split():
            w = self._regex_nonword.sub("", w)
            if not w or w in self.stopwords:
                continue
            out.append(w)
        return out

    def _count_ngrams(self, tokens: List[str], max_n: int = 3) -> Dict[str, int]:
        """Count ngrams directly (avoids building a huge candidate list). Returns: ngram -> frequency"""
        ng_freq: Dict[str, int] = defaultdict(int)
        L = len(tokens)
        for n in range(1, max_n + 1):
            for i in range(0, L - n + 1):
                ng = " ".join(tokens[i:i + n])
                ng_freq[ng] += 1
        return ng_freq

    def _extract_yake_keywords(self, text: str, top_k: int = 20, max_n: int = 3) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Simplified YAKE-like scorer using:
          - term frequency
          - positional bias (earlier is better)
          - context diversity (unique neighbors)
        Score: lower is better.
        """
        sentences = self._split_into_sentences(text)

        # build a flat token list, plus first positions and frequency in one pass.
        tokens_all: List[str] = []
        word_frequency: Dict[str, int] = defaultdict(int)
        first_pos: Dict[str, int] = {}

        pos = 0
        for s in sentences:
            toks = self._tokenize(s)
            for t in toks:
                tokens_all.append(t)
                word_frequency[t] += 1
                if t not in first_pos:
                    first_pos[t] = pos
                pos += 1

        if not tokens_all:
            return {}, {"phrase_scores": {}, "word_scores": {}, "candidate_ngrams": []}

        # Neighbor diversity with ONE set per word (undirected adjacency)
        neighbors: Dict[str, Set[str]] = defaultdict(set)
        for i in range(1, len(tokens_all)):
            a = tokens_all[i - 1]
            b = tokens_all[i]
            neighbors[a].add(b)
            neighbors[b].add(a)

        # Word scores
        word_score: Dict[str, float] = {}
        total_len = max(len(tokens_all), 1)

        for w, f in word_frequency.items():
            fp = first_pos.get(w, total_len)
            pos_norm = (fp + 1) / total_len  # smaller = earlier

            div = len(neighbors.get(w, set()))
            freq_term = 1.0 / (1.0 + math.log(1.0 + f))
            div_term = 1.0 / (1.0 + div)

            word_score[w] = pos_norm * freq_term * div_term

        # Ngram frequencies (fast, no big list)
        ng_freq = self._count_ngrams(tokens_all, max_n=max_n)

        # Phrase scores (lower is better)
        phrase_scores: Dict[str, float] = {}
        for ng, f in ng_freq.items():
            parts = ng.split()
            if not parts:
                continue

            # mean word score across words in the ngram
            ws = sum(word_score.get(w, 1.0) for w in parts) / len(parts)

            # mild preference for shorter phrases and rare phrases
            length_bonus = 1.0 / (1.0 + (len(parts) - 1) * 0.5)
            freq_bonus = 1.0 / (1.0 + math.log(1.0 + f))

            yake_like = ws * length_bonus * freq_bonus

            # keep best (lowest) if recomputed
            if (ng not in phrase_scores) or (yake_like < phrase_scores[ng]):
                phrase_scores[ng] = yake_like

        best = sorted(phrase_scores.items(), key=lambda x: x[1])[:top_k]
        kw_scores = {k: v for k, v in best}

        debug = {
            # Unique ngrams only (much smaller than “all occurrences” list)
            "candidate_ngrams": list(ng_freq.keys()),
            "phrase_scores": phrase_scores,
            "word_scores": word_score,
        }
        return kw_scores, debug

    # ---------------- Incremental updates (drop-in) ----------------

    def _on_document_changed(self, doc_data, action: str) -> None:
        start = time.perf_counter()

        if action == "added":
            self._incremental_add(doc_data)
        else:
            # If you later add "replaced" incremental, you can do it here.
            self.preprocess()

        elapsed = time.perf_counter() - start
        print(
            f"[YAKE] Document {action}: {getattr(doc_data, 'filename', '(unknown)')} | "
            f"time={elapsed:.4f}s"
        )

    def _process_one_doc(self, doc) -> None:
        """Compute and cache YAKE data for one document."""
        doc_name = doc.filename
        text = getattr(doc, "raw_text", "") or ""

        # kw_scores: top-k best ngrams for keyword extraction
        # dbg["phrase_scores"]: ALL candidate ngrams for search
        kw_scores, dbg = self._extract_yake_keywords(text, top_k=200, max_n=3)

        self._doc_kw_scores[doc_name] = kw_scores

        all_phrase_scores = dbg["phrase_scores"]
        self._doc_all_phrase_scores[doc_name] = all_phrase_scores
        self._doc_phrase_set[doc_name] = set(all_phrase_scores.keys())

        ws: Set[str] = set()
        for ng in all_phrase_scores.keys():
            ws.update(ng.split())
        self._doc_word_set[doc_name] = ws

    def _incremental_add(self, doc) -> None:
        """Incremental add: compute YAKE caches only for the newly added doc."""
        self._process_one_doc(doc)

    def _write_yake_debug(
        self,
        out_path: str,
        candidate_ngrams: List[str],
        phrase_scores: Dict[str, float],
        word_scores: Dict[str, float],
        title: str = "YAKE Debug Output",
        max_items: int = 5000,
    ) -> None:
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        def _best_sorted(d: Dict[str, float]) -> List[Tuple[str, float]]:
            return sorted(d.items(), key=lambda x: x[1])[:max_items]  # lower is better

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n")
            f.write(f"Candidate ngrams (unique): {len(candidate_ngrams)}\n")
            f.write(f"Unique words: {len(word_scores)}\n")
            f.write(f"Unique phrases/ngrams: {len(phrase_scores)}\n")
            f.write("NOTE: YAKE scores are lower-is-better.\n")
            f.write("\n" + "=" * 80 + "\n\n")

            f.write("[CANDIDATE_NGRAMS] (unique)\n")
            for i, ng in enumerate(candidate_ngrams[:max_items], start=1):
                f.write(f"{i:05d}. {ng}\n")
            if len(candidate_ngrams) > max_items:
                f.write(f"... truncated (showing first {max_items})\n")

            f.write("\n" + "=" * 80 + "\n\n")

            f.write("[WORD_SCORES] (sorted low -> high, best -> worst)\n")
            for w, s in _best_sorted(word_scores):
                f.write(f"{s:12.6e}\t{w}\n")
            if len(word_scores) > max_items:
                f.write(f"... truncated (showing best {max_items})\n")

            f.write("\n" + "=" * 80 + "\n\n")

            f.write("[PHRASE_SCORES] (sorted low -> high, best -> worst)\n")
            for p, s in _best_sorted(phrase_scores):
                f.write(f"{s:12.6e}\t{p}\n")
            if len(phrase_scores) > max_items:
                f.write(f"... truncated (showing best {max_items})\n")
