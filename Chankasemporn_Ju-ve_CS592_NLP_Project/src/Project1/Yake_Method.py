"""
File Name:    Yake_Method.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import math
import re
import string

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
            kw_scores = self._extract_yake_keywords(text, top_k=200)  # keep more for searching
            self._doc_kw_scores[doc.filename] = kw_scores
            self._doc_kw_set[doc.filename] = set(kw_scores.keys())

            ws = set()
            for kw in kw_scores.keys():
                ws.update(kw.split())
            self._doc_word_set[doc.filename] = ws

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

            matched_str = ", ".join([f"'{kw}' (yake={s:.4f})" for kw, s in matched_sorted]) or "No exact keyword match"
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

    def _extract_yake_keywords(self, text: str, top_k: int = 20, max_n: int = 3) -> Dict[str, float]:
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
            return {}

        # word frequency
        wf: Dict[str, int] = defaultdict(int)
        for t in tokens_all:
            wf[t] += 1

        # neighbor diversity (unique left/right neighbors)
        left_neighbors: Dict[str, Set[str]] = defaultdict(set)
        right_neighbors: Dict[str, Set[str]] = defaultdict(set)
        for i, t in enumerate(tokens_all):
            if i > 0:
                left_neighbors[t].add(tokens_all[i-1])
            if i < len(tokens_all) - 1:
                right_neighbors[t].add(tokens_all[i+1])

        # word-level features
        # - freq term: higher freq => lower score
        # - pos term: earlier occurrence => lower score
        # - diversity term: more unique neighbors => lower score
        word_score: Dict[str, float] = {}

        total_len = max(len(tokens_all), 1)
        for w, f in wf.items():
            first_pos = token_positions[w][0] if token_positions[w] else total_len
            pos_norm = (first_pos + 1) / total_len  # [0..1], smaller = earlier

            div = len(left_neighbors[w]) + len(right_neighbors[w])  # 0..large
            # Convert to a score where lower is better:
            #   base = (pos_norm) * (1 / (1+log(1+f))) * (1 / (1+div))
            freq_term = 1.0 / (1.0 + math.log(1.0 + f))
            div_term = 1.0 / (1.0 + div)

            word_score[w] = pos_norm * freq_term * div_term

        # candidate ngrams + scoring
        ngrams = self._generate_ngrams(tokens_all, max_n=max_n)

        # keep unique, but track minimal score if duplicates
        cand_scores: Dict[str, float] = {}

        for ng in ngrams:
            parts = ng.split()
            if not parts:
                continue

            # ngram frequency
            # (cheap approximate count using string map would be expensive;
            #  we do a small rolling count using a dict)
            # Instead: accumulate counts in a first pass:
            # We'll do it properly with a separate pass below.
            pass

        # Count ngram frequency efficiently
        ng_freq: Dict[str, int] = defaultdict(int)
        for ng in ngrams:
            ng_freq[ng] += 1

        for ng, f in ng_freq.items():
            parts = ng.split()
            if not parts:
                continue

            # combine word scores (lower better) + ngram length bonus + freq bonus
            # lower score => better keyword
            ws = sum(word_score.get(w, 1.0) for w in parts) / len(parts)

            # favor longer phrases a bit (YAKE often returns multiword)
            length_bonus = 1.0 / (1.0 + (len(parts) - 1) * 0.5)

            # favor repeated ngrams
            freq_bonus = 1.0 / (1.0 + math.log(1.0 + f))

            yake_like = ws * length_bonus * freq_bonus

            # keep best (lowest)
            if (ng not in cand_scores) or (yake_like < cand_scores[ng]):
                cand_scores[ng] = yake_like

        # return top_k best (lowest) as dict
        best = sorted(cand_scores.items(), key=lambda x: x[1])[:top_k]
        return {k: v for k, v in best}