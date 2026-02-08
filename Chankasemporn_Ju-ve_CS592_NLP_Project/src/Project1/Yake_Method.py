"""
File Name:    Yake_Method.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
import math
import re
import string
import os

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
    """
    YAKE-ish implementation:
      - preprocess(): extract YAKE phrases for each document and cache them
      - run(query): rank documents by overlap with YAKE phrases from query
    Notes:
      - YAKE scores are lower-is-better.
      - For SEARCH, we must match against ALL candidate phrases, not only top_k.
    """

    def __init__(self, name: str = "YAKE", processor: Optional[DocumentProcessor] = None):
        super().__init__(name=name)
        self.processor = processor or DocumentProcessor()

        self.stopwords: Set[str] = set(w.lower() for w in Globals.STOP_WORDS)
        self.punct: Set[str] = set(string.punctuation)

        # caches
        self._doc_kw_scores: Dict[str, Dict[str, float]] = {}         # doc -> top-k keyword -> score (for extraction)
        self._doc_all_phrase_scores: Dict[str, Dict[str, float]] = {} # doc -> ALL candidate ngram -> score (for search)
        self._doc_phrase_set: Dict[str, Set[str]] = {}                # doc -> set(ALL candidate ngrams)
        self._doc_word_set: Dict[str, Set[str]] = {}                  # doc -> set(words appearing in candidate ngrams)

    # ---------------- Public API ----------------

    def preprocess(self) -> None:
        for doc in self.processor.documents:
            text = doc.text

            # kw_scores: top-k best ngrams for keyword extraction
            # dbg["phrase_scores"]: ALL candidate ngrams for search
            kw_scores, dbg = self._extract_yake_keywords(text, top_k=200)

            # Keep top-k only for extract_keywords()
            self._doc_kw_scores[doc.filename] = kw_scores

            # For SEARCH, cache ALL candidate phrase scores + sets
            all_phrase_scores = dbg["phrase_scores"]
            self._doc_all_phrase_scores[doc.filename] = all_phrase_scores
            self._doc_phrase_set[doc.filename] = set(all_phrase_scores.keys())

            ws: Set[str] = set()
            for ng in all_phrase_scores.keys():
                ws.update(ng.split())
            self._doc_word_set[doc.filename] = ws

            # Debug file (optional)
            self._write_yake_debug(
                out_path=f"output/yake/yake_debug_{doc.filename}.txt",
                candidate_ngrams=dbg["candidate_ngrams"],
                phrase_scores=dbg["phrase_scores"],
                word_scores=dbg["word_scores"],
                title=f"YAKE Debug Output (doc={doc.filename})",
            )

    def run(self, query: str) -> List[SearchResult]:
        if not query or not query.strip():
            return [SearchResult(title="Empty query", score=0.0, details="Please type a query.")]

        # IMPORTANT: _extract_yake_keywords returns (kw_scores, debug_dict)
        # For SEARCH matching, use ALL query candidates, not only top-k.
        _, qdbg = self._extract_yake_keywords(query, top_k=200)
        query_phrase_scores_all: Dict[str, float] = qdbg["phrase_scores"]
        query_phrase_set = set(query_phrase_scores_all.keys())

        # Also build query word set
        query_word_set: Set[str] = set()
        for p in query_phrase_set:
            query_word_set.update(p.split())

        results: List[Tuple[str, float, str]] = []

        for doc in self.processor.documents:
            doc_name = doc.filename

            doc_phrase_scores = self._doc_all_phrase_scores.get(doc_name, {})
            doc_phrases = self._doc_phrase_set.get(doc_name, set())
            doc_words = self._doc_word_set.get(doc_name, set())

            if not doc_phrase_scores:
                continue

            overlap = query_phrase_set.intersection(doc_phrases)

            # ---- scoring ----
            # YAKE lower-is-better. Convert to a stable higher-is-better score.
            # Use -log(score) instead of 1/score to avoid huge explosions.
            kw_score = 0.0
            for kw in overlap:
                s = doc_phrase_scores.get(kw, 1.0)
                kw_score += -math.log(s + 1e-12)  # stable, higher is better

            # Word overlap bonus (helps when phrases differ but single words match)
            word_overlap = query_word_set.intersection(doc_words)
            word_bonus = len(word_overlap) * 0.10

            total = kw_score + word_bonus
            if total <= 0:
                continue

            matched_sorted = sorted(
                [(kw, doc_phrase_scores.get(kw, 1.0)) for kw in overlap],
                key=lambda x: x[1]  # lower YAKE score first (better)
            )[:5]

            matched_str = ", ".join([f"'{kw}' (yake={s:.3e})" for kw, s in matched_sorted]) or "No exact keyword match"
            results.append((doc_name, total, matched_str))

        results.sort(key=lambda x: x[1], reverse=True)

        out: List[SearchResult] = []
        for doc_name, score, matched_str in results[:10]:
            doc_obj = self.processor.get_document_by_name(doc_name)
            out.append(SearchResult(
                title=doc_name,
                score=float(score),
                details=f"Matched: {matched_str}\nPath: {doc_obj.path if doc_obj else 'N/A'}"
            ))

        if not out:
            # Fallback: show all docs (like your other methods)
            for doc in self.processor.documents:
                out.append(SearchResult(
                    title=doc.filename,
                    score=0.0,
                    details=f"No YAKE keyword overlap with query.\nPath: {doc.path}"
                ))

        return out

    def extract_keywords(self, doc_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns (keyword/ngram, yake_score) sorted by YAKE score ascending (lower is better).
        This uses the cached top-k extraction list (not all candidates).
        """
        scores = self._doc_kw_scores.get(doc_name, {})
        items = sorted(scores.items(), key=lambda x: x[1])
        return items[:top_k]

    # ---------------- YAKE-ish internals ----------------

    def _split_into_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'[\n.!?]+', text) if s.strip()]

    def _tokenize(self, text: str) -> List[str]:
        words = []
        for w in text.lower().split():
            w = w.strip(string.punctuation)
            w = re.sub(r"[^\w]+", "", w)
            if not w:
                continue
            if w in self.stopwords:
                continue
            words.append(w)
        return words

    def _generate_ngrams(self, tokens: List[str], max_n: int = 3) -> List[str]:
        out: List[str] = []
        L = len(tokens)
        for n in range(1, max_n + 1):
            for i in range(0, L - n + 1):
                out.append(" ".join(tokens[i:i + n]))
        return out

    def _extract_yake_keywords(
        self,
        text: str,
        top_k: int = 20,
        max_n: int = 3
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Simplified YAKE-like scorer using:
          - term frequency
          - positional bias (earlier is better)
          - context diversity (how many different neighbors it sees)
        Score: lower is better.
        Returns:
          - kw_scores: top_k best ngrams (dict)
          - debug: includes ALL phrase_scores (for search) and candidate_ngrams list
        """
        sentences = self._split_into_sentences(text)
        tokens_all: List[str] = []
        token_positions: Dict[str, List[int]] = defaultdict(list)

        pos = 0
        for s in sentences:
            toks = self._tokenize(s)
            for t in toks:
                tokens_all.append(t)
                token_positions[t].append(pos)
                pos += 1

        if not tokens_all:
            return {}, {"candidate_ngrams": [], "phrase_scores": {}, "word_scores": {}}

        wf: Dict[str, int] = defaultdict(int)
        for t in tokens_all:
            wf[t] += 1

        left_neighbors: Dict[str, Set[str]] = defaultdict(set)
        right_neighbors: Dict[str, Set[str]] = defaultdict(set)
        for i, t in enumerate(tokens_all):
            if i > 0:
                left_neighbors[t].add(tokens_all[i - 1])
            if i < len(tokens_all) - 1:
                right_neighbors[t].add(tokens_all[i + 1])

        word_score: Dict[str, float] = {}
        total_len = max(len(tokens_all), 1)

        for w, f in wf.items():
            first_pos = token_positions[w][0] if token_positions[w] else total_len
            pos_norm = (first_pos + 1) / total_len  # [0..1], smaller = earlier

            div = len(left_neighbors[w]) + len(right_neighbors[w])
            freq_term = 1.0 / (1.0 + math.log(1.0 + f))
            div_term = 1.0 / (1.0 + div)

            word_score[w] = pos_norm * freq_term * div_term

        candidate_ngrams = self._generate_ngrams(tokens_all, max_n=max_n)

        ng_freq: Dict[str, int] = defaultdict(int)
        for ng in candidate_ngrams:
            ng_freq[ng] += 1

        phrase_scores: Dict[str, float] = {}
        for ng, f in ng_freq.items():
            parts = ng.split()
            if not parts:
                continue

            ws = sum(word_score.get(w, 1.0) for w in parts) / len(parts)

            length_bonus = 1.0 / (1.0 + (len(parts) - 1) * 0.5)
            freq_bonus = 1.0 / (1.0 + math.log(1.0 + f))

            yake_like = ws * length_bonus * freq_bonus

            if (ng not in phrase_scores) or (yake_like < phrase_scores[ng]):
                phrase_scores[ng] = yake_like

        best = sorted(phrase_scores.items(), key=lambda x: x[1])[:top_k]
        kw_scores = {k: v for k, v in best}

        debug = {
            "candidate_ngrams": candidate_ngrams,
            "phrase_scores": phrase_scores,  # ALL candidates (use this for search)
            "word_scores": word_score,
        }
        return kw_scores, debug

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
            f.write(f"Candidate ngrams: {len(candidate_ngrams)}\n")
            f.write(f"Unique words: {len(word_scores)}\n")
            f.write(f"Unique phrases/ngrams: {len(phrase_scores)}\n")
            f.write("NOTE: YAKE scores are lower-is-better.\n")
            f.write("\n" + "=" * 80 + "\n\n")

            f.write("[CANDIDATE_NGRAMS]\n")
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
