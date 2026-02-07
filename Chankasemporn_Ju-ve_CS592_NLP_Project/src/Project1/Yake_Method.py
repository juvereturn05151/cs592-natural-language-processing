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
      - preprocess(): extract YAKE keywords for each document and cache them
      - run(query): rank documents by overlap with YAKE keywords from query
    """

    def __init__(self, name: str = "YAKE", processor: Optional[DocumentProcessor] = None):
        super().__init__(name=name)
        self.processor = processor or DocumentProcessor()

        self.stopwords: Set[str] = set(w.lower() for w in Globals.STOP_WORDS)
        self.punct: Set[str] = set(string.punctuation)

        # caches
        self._doc_kw_scores: Dict[str, Dict[str, float]] = {}   # doc -> keyword -> score (lower is better in YAKE)
        self._doc_kw_set: Dict[str, Set[str]] = {}              # doc -> set(keywords)
        self._doc_word_set: Dict[str, Set[str]] = {}            # doc -> set(words appearing in keywords)

    # ---------------- Public API ----------------

    def preprocess(self) -> None:
        for doc in self.processor.documents:
            text = doc.text
            kw_scores, dbg = self._extract_yake_keywords(text, top_k=200)
            self._doc_kw_scores[doc.filename] = kw_scores
            self._doc_kw_set[doc.filename] = set(kw_scores.keys())

            ws = set()
            for kw in kw_scores.keys():
                ws.update(kw.split())
            self._doc_word_set[doc.filename] = ws

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

        # YAKE keywords from the query
        query_kw_scores = self._extract_yake_keywords(query, top_k=30)
        query_kws = list(query_kw_scores.keys())
        query_kw_set = set(query_kws)

        query_word_set = set()
        for kw in query_kws:
            query_word_set.update(kw.split())

        results: List[Tuple[str, float, str]] = []

        for doc in self.processor.documents:
            doc_name = doc.filename
            doc_kw_scores = self._doc_kw_scores.get(doc_name, {})
            doc_kw_set = self._doc_kw_set.get(doc_name, set())
            doc_word_set = self._doc_word_set.get(doc_name, set())

            if not doc_kw_scores:
                continue

            overlap = query_kw_set.intersection(doc_kw_set)

            # YAKE scores are "lower is better". Convert to "higher is better" contribution.
            # contribution(kw) = 1 / (epsilon + yake_score)
            eps = 1e-9
            kw_score = 0.0
            for kw in overlap:
                kw_score += 1.0 / (eps + doc_kw_scores.get(kw, 1.0))

            # Word overlap bonus (small)
            word_overlap = query_word_set.intersection(doc_word_set)
            word_bonus = len(word_overlap) * 0.10

            total = kw_score + word_bonus
            if total <= 0:
                continue

            # explanation: top matched keywords by contribution
            matched_sorted = sorted(
                [(kw, doc_kw_scores.get(kw, 1.0)) for kw in overlap],
                key=lambda x: x[1]  # lower YAKE score first (better)
            )[:5]

            matched_str = ", ".join([f"'{kw}' (yake={s:.9f})" for kw, s in matched_sorted]) or "No exact keyword match"
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
            for doc in self.processor.documents:
                out.append(SearchResult(
                    title=doc.filename,
                    score=0.0,
                    details=f"No YAKE keyword overlap with query.\nPath: {doc.path}"
                ))

        return out

    def extract_keywords(self, doc_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns (keyword, yake_score) sorted by YAKE score ascending (lower is better).
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
            # filter out pure numbers (optional)
            # if w.isdigit(): continue
            words.append(w)
        return words

    def _generate_ngrams(self, tokens: List[str], max_n: int = 3) -> List[str]:
        out: List[str] = []
        L = len(tokens)
        for n in range(1, max_n + 1):
            for i in range(0, L - n + 1):
                ng = " ".join(tokens[i:i+n])
                out.append(ng)
        return out

    def _extract_yake_keywords(
            self,
            text: str,
            top_k: int = 20,
            max_n: int = 3
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        A simplified YAKE-like scorer using:
          - term frequency
          - positional bias (earlier is better)
          - context diversity (how many different neighbors it sees)

        Score: lower is better.
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

        # word frequency
        wf: Dict[str, int] = defaultdict(int)
        for t in tokens_all:
            wf[t] += 1

        # neighbor diversity (unique left/right neighbors)
        left_neighbors: Dict[str, Set[str]] = defaultdict(set)
        right_neighbors: Dict[str, Set[str]] = defaultdict(set)
        for i, t in enumerate(tokens_all):
            if i > 0:
                left_neighbors[t].add(tokens_all[i - 1])
            if i < len(tokens_all) - 1:
                right_neighbors[t].add(tokens_all[i + 1])

        # word-level features -> lower is better
        word_score: Dict[str, float] = {}
        total_len = max(len(tokens_all), 1)

        for w, f in wf.items():
            first_pos = token_positions[w][0] if token_positions[w] else total_len
            pos_norm = (first_pos + 1) / total_len  # [0..1], smaller = earlier

            div = len(left_neighbors[w]) + len(right_neighbors[w])
            freq_term = 1.0 / (1.0 + math.log(1.0 + f))
            div_term = 1.0 / (1.0 + div)

            word_score[w] = pos_norm * freq_term * div_term

        # candidate ngrams + scoring
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
            "phrase_scores": phrase_scores,
            "word_scores": word_score,
        }
        return kw_scores, debug

    def _write_yake_debug(
            self,
            out_path: str,
            candidate_ngrams: List[str],
            phrase_scores: Dict[str, float],  # keyword/ngram -> yake score (lower is better)
            word_scores: Dict[str, float],  # word -> yake-ish score (lower is better)
            title: str = "YAKE Debug Output",
            max_items: int = 5000,
    ) -> None:
        """
        Writes a YAKE debug text report:
          - candidate_ngrams (in original order)
          - word_scores (sorted ASC: best -> worst) because YAKE is lower-is-better
          - phrase_scores (sorted ASC: best -> worst)
        """
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        def _best_sorted(d: Dict[str, float]) -> List[Tuple[str, float]]:
            # YAKE: lower score = better
            return sorted(d.items(), key=lambda x: x[1])[:max_items]

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n")
            f.write(f"Candidate ngrams: {len(candidate_ngrams)}\n")
            f.write(f"Unique words: {len(word_scores)}\n")
            f.write(f"Unique phrases/ngrams: {len(phrase_scores)}\n")
            f.write("NOTE: YAKE scores are lower-is-better.\n")
            f.write("\n" + "=" * 80 + "\n\n")

            # 1) Candidate n-grams (original order)
            f.write("[CANDIDATE_NGRAMS]\n")
            for i, ng in enumerate(candidate_ngrams[:max_items], start=1):
                f.write(f"{i:05d}. {ng}\n")
            if len(candidate_ngrams) > max_items:
                f.write(f"... truncated (showing first {max_items})\n")

            f.write("\n" + "=" * 80 + "\n\n")

            # 2) Word scores (sorted best -> worst)
            f.write("[WORD_SCORES] (sorted low -> high, best -> worst)\n")
            for w, s in _best_sorted(word_scores):
                f.write(f"{s:10.6f}\t{w}\n")
            if len(word_scores) > max_items:
                f.write(f"... truncated (showing best {max_items})\n")

            f.write("\n" + "=" * 80 + "\n\n")

            # 3) Phrase/ngram scores (sorted best -> worst)
            f.write("[PHRASE_SCORES] (sorted low -> high, best -> worst)\n")
            for p, s in _best_sorted(phrase_scores):
                f.write(f"{s:10.6f}\t{p}\n")
            if len(phrase_scores) > max_items:
                f.write(f"... truncated (showing best {max_items})\n")