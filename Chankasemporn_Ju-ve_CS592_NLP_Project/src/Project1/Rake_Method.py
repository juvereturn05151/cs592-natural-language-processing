"""
File Name:    Rake_Method.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
import re
import string
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

from .KeywordMethod import SearchResult, KeywordMethod
from .DocumentProcessor import DocumentProcessor
import src.NLP_Globals as Globals


def create_rake_method(processor: DocumentProcessor = None) -> KeywordMethod:
    rake = RakeMethod(
        name="RAKE (Keyword Extraction / Document Search)",
        processor=processor
    )

    rake.preprocess()
    return rake


class RakeMethod(KeywordMethod):
    def __init__(self, name: str = "RAKE", processor: Optional[DocumentProcessor] = None):
        super().__init__(name=name)
        self.processor = processor or DocumentProcessor()

        # STOP_WORDS from NLTK are already lowercase, so no need to lower() again
        self.stopwords: Set[str] = Globals.STOP_WORDS

        # Kept for compatibility, but we won't rely on per-token strip anymore
        self.punct: Set[str] = set(string.punctuation)

        # Use precompiled regex from Globals where possible
        self._regex_cleaner = Globals.REGEX_CLEANER

        # Extra compiled regex (fallback) for removing remaining non-word chars
        # If you add REGEX_NONWORD in NLP_Globals.py, we'll automatically use it.
        self._regex_nonword = getattr(Globals, "REGEX_NONWORD", re.compile(r"[^\w]+"))

        # Caches
        self._doc_phrase_scores: Dict[str, Dict[str, float]] = {}  # doc -> phrase->score
        self._doc_word_scores: Dict[str, Dict[str, float]] = {}    # doc -> word->score (optional)
        self._doc_phrase_set: Dict[str, Set[str]] = {}             # doc -> set(phrases)
        self._doc_word_set: Dict[str, Set[str]] = {}               # doc -> set(words in phrases)

    # ---------- Public API ----------

    def preprocess(self) -> None:
        start = time.perf_counter()

        # Clear caches so re-preprocess doesn't keep stale docs
        self._doc_phrase_scores.clear()
        self._doc_word_scores.clear()
        self._doc_phrase_set.clear()
        self._doc_word_set.clear()

        docs_processed = 0
        total_phrases = 0

        for doc in self.processor.documents:
            raw_text = doc.raw_text
            phrases = self._generate_candidate_phrases(raw_text)
            phrase_scores, word_scores = self._calculate_rake_scores(phrases)

            self._doc_phrase_scores[doc.filename] = phrase_scores
            self._doc_word_scores[doc.filename] = word_scores
            self._doc_phrase_set[doc.filename] = set(phrase_scores.keys())

            # Build doc word set from unique phrases
            word_set: Set[str] = set()
            for p in phrase_scores.keys():
                word_set.update(p.split())
            self._doc_word_set[doc.filename] = word_set

            docs_processed += 1
            total_phrases += len(phrases)

        elapsed = time.perf_counter() - start
        print(
            f"[RAKE] Preprocess | docs={docs_processed} | "
            f"total_phrases={total_phrases} | time={elapsed:.4f}s"
        )

    def extract_keywords(self, doc_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        scores = self._doc_phrase_scores.get(doc_name, {})
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def extract_keywords_from_text(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        phrases = self._generate_candidate_phrases(text)
        phrase_scores, _ = self._calculate_rake_scores(phrases)
        return sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def run(self, query: str) -> List[SearchResult]:
        """Rank documents for a query using RAKE phrase/word overlap."""
        start = time.perf_counter()

        if not query or not query.strip():
            return [SearchResult(title="Empty query", score=0.0, details="Please type a query.")]

        # Extract RAKE phrases from the query itself
        query_phrases_scored = self.extract_keywords_from_text(query, top_k=30)
        query_phrases = [p for p, _ in query_phrases_scored]
        query_phrase_set = set(query_phrases)

        # Also build a query word set for partial matching
        query_word_set: Set[str] = set()
        for p in query_phrases:
            query_word_set.update(p.split())

        results: List[Tuple[str, float, str]] = []
        docs_scored = 0

        for doc in self.processor.documents:
            doc_name = doc.filename
            doc_phrase_scores = self._doc_phrase_scores.get(doc_name, {})
            doc_phrases = self._doc_phrase_set.get(doc_name, set())
            doc_words = self._doc_word_set.get(doc_name, set())

            if not doc_phrase_scores:
                continue

            docs_scored += 1

            # Phrase overlap weighted by document phrase scores
            phrase_overlap = query_phrase_set.intersection(doc_phrases)
            phrase_score = sum(doc_phrase_scores[p] for p in phrase_overlap)

            # Word overlap bonus (helps when phrases differ but words match)
            word_overlap = query_word_set.intersection(doc_words)
            word_bonus = len(word_overlap) * 0.25

            total = phrase_score + word_bonus
            if total <= 0:
                continue

            # Make a short explanation (top matched phrases)
            matched_phrases_sorted = sorted(
                [(p, doc_phrase_scores[p]) for p in phrase_overlap],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            matched_str = ", ".join([f"'{p}' ({s:.2f})" for p, s in matched_phrases_sorted]) or "No exact phrase match"

            results.append((doc_name, total, matched_str))

        # Sort and format SearchResult
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
            # Fallback: show all available documents
            for doc in self.processor.documents:
                out.append(SearchResult(
                    title=doc.filename,
                    score=0.0,
                    details=(
                        "No RAKE keyword overlap with query.\n"
                        f"Path: {doc.path}"
                    )
                ))

        elapsed = time.perf_counter() - start
        print(
            f"[RAKE] Run | query_len={len(query.split())} | "
            f"docs_scored={docs_scored} | hits={len(results)} | "
            f"time={elapsed:.4f}s"
        )

        return out

    # ---------- RAKE internals ----------

    def _split_into_sentences(self, text: str) -> List[str]:
        # Slightly more robust than splitting on .!? only
        return [s.strip() for s in re.split(r'[\n.!?]+', text) if s.strip()]

    def _generate_candidate_phrases(self, text: str) -> List[str]:
        """
        Faster candidate phrase generation:
        - Do punctuation cleaning ONCE per sentence using Globals.REGEX_CLEANER (compiled)
        - Avoid per-token strip() + re.sub() work where possible
        """
        sentences = self._split_into_sentences(text)
        candidates: List[str] = []

        for sentence in sentences:
            # One regex pass per sentence instead of per token
            cleaned = self._regex_cleaner.sub(" ", sentence.lower())
            words = cleaned.split()

            phrase: List[str] = []
            for w in words:
                # Remove any remaining non-word chars quickly (compiled)
                w = self._regex_nonword.sub("", w)

                if (not w) or (w in self.stopwords):
                    if phrase:
                        candidates.append(" ".join(phrase))
                        phrase = []
                else:
                    phrase.append(w)

            if phrase:
                candidates.append(" ".join(phrase))

        # Filter candidates
        filtered: List[str] = []
        for p in candidates:
            toks = p.split()
            if 1 <= len(toks) <= 4 and len(p) > 1:
                filtered.append(p)

        return filtered

    def _calculate_rake_scores(self, candidate_phrases: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Standard-ish RAKE:
          - word_freq[word] += 1
          - word_degree[word] += (len(phrase)-1)
          - then word_degree[word] += word_freq[word]
          - word_score = word_degree / word_freq
          - phrase_score = sum(word_score[word] for word in phrase)
        """
        word_freq = defaultdict(int)
        word_degree = defaultdict(int)

        for phrase in candidate_phrases:
            words = phrase.split()
            if not words:
                continue
            L = len(words)
            for w in words:
                word_freq[w] += 1
                word_degree[w] += (L - 1)

        for w in word_freq:
            word_degree[w] += word_freq[w]

        word_scores: Dict[str, float] = {}
        for w in word_freq:
            word_scores[w] = word_degree[w] / max(word_freq[w], 1)

        phrase_scores: Dict[str, float] = {}
        for phrase in candidate_phrases:
            score = 0.0
            for w in phrase.split():
                score += word_scores.get(w, 0.0)
            # keep the max score if phrase repeats
            if phrase not in phrase_scores or score > phrase_scores[phrase]:
                phrase_scores[phrase] = score

        return phrase_scores, word_scores

    def _write_rake_debug(
            self,
            out_path: str,
            candidate_phrases: List[str],
            phrase_scores: Dict[str, float],
            word_scores: Dict[str, float],
            title: str = "RAKE Debug Output",
            max_items: int = 5000,  # safety cap for huge docs
    ) -> None:
        """
        Writes a text report:
          - candidate_phrases (in original order)
          - word_scores (sorted desc)
          - phrase_scores (sorted desc)
        """
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        def _top_sorted(d: Dict[str, float]) -> List[Tuple[str, float]]:
            return sorted(d.items(), key=lambda x: x[1], reverse=True)[:max_items]

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n")
            f.write(f"Candidate phrases: {len(candidate_phrases)}\n")
            f.write(f"Unique words: {len(word_scores)}\n")
            f.write(f"Unique phrases: {len(phrase_scores)}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            f.write("[CANDIDATE_PHRASES]\n")
            for i, p in enumerate(candidate_phrases[:max_items], start=1):
                f.write(f"{i:05d}. {p}\n")
            if len(candidate_phrases) > max_items:
                f.write(f"... truncated (showing first {max_items})\n")

            f.write("\n" + "=" * 80 + "\n\n")

            f.write("[WORD_SCORES] (sorted high -> low)\n")
            for w, s in _top_sorted(word_scores):
                f.write(f"{s:10.6f}\t{w}\n")
            if len(word_scores) > max_items:
                f.write(f"... truncated (showing top {max_items})\n")

            f.write("\n" + "=" * 80 + "\n\n")

            f.write("[PHRASE_SCORES] (sorted high -> low)\n")
            for p, s in _top_sorted(phrase_scores):
                f.write(f"{s:10.6f}\t{p}\n")
            if len(phrase_scores) > max_items:
                f.write(f"... truncated (showing top {max_items})\n")
