import re
import string
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

from .KeywordMethod import SearchResult, KeywordMethod
from .DocumentProcessor import DocumentProcessor
import src.NLP_Globals as Globals
import src.TokenizerHelper as TokenizerHelper

def create_rake_method(processor: DocumentProcessor = None) -> KeywordMethod:
    """
    Factory function to create and initialize RAKE method.

    Important:
    - RAKE needs stopwords to split phrases => tokenizer should NOT remove stopwords.
    - We should reuse the same DocumentProcessor that main.py already loaded,
      so RAKE and TF-IDF operate on the same document corpus.
    """

    rake = RakeMethod(
        name="RAKE (Keyword Extraction / Document Search)",
        processor=processor
    )

    rake.preprocess()
    return rake

class RakeMethod(KeywordMethod):
    """
    RAKE-based document ranking.
    - preprocess(): extracts RAKE phrases for each document and caches them
    - run(query): ranks documents for the query (NOT keyword extraction mode)
    """

    def __init__(self, name: str = "RAKE", processor: Optional[DocumentProcessor] = None):
        super().__init__(name=name)
        self.processor = processor or DocumentProcessor()
        self._is_loaded = False

        self.stopwords: Set[str] = set(w.lower() for w in Globals.STOP_WORDS)
        self.punct: Set[str] = set(string.punctuation)

        # Caches
        self._doc_phrase_scores: Dict[str, Dict[str, float]] = {}  # doc -> phrase->score
        self._doc_word_scores: Dict[str, Dict[str, float]] = {}    # doc -> word->score (optional)
        self._doc_phrase_set: Dict[str, Set[str]] = {}             # doc -> set(phrases)
        self._doc_word_set: Dict[str, Set[str]] = {}               # doc -> set(words in phrases)

    # ---------- Public API ----------

    def preprocess(self) -> None:
        if self._is_loaded:
            return
        if not self.processor.documents:
            self._is_loaded = True
            return

        for doc in self.processor.documents:
            raw_text = doc.text
            phrases = self._generate_candidate_phrases(raw_text)
            phrase_scores, word_scores = self._calculate_rake_scores(phrases)

            self._doc_phrase_scores[doc.filename] = phrase_scores
            self._doc_word_scores[doc.filename] = word_scores
            self._doc_phrase_set[doc.filename] = set(phrase_scores.keys())

            word_set = set()
            for p in phrase_scores.keys():
                word_set.update(p.split())
            self._doc_word_set[doc.filename] = word_set

        self._is_loaded = True

    def extract_keywords(self, doc_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self._is_loaded:
            self.preprocess()

        scores = self._doc_phrase_scores.get(doc_name, {})
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def extract_keywords_from_text(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        phrases = self._generate_candidate_phrases(text)
        phrase_scores, _ = self._calculate_rake_scores(phrases)
        return sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def run(self, query: str) -> List[SearchResult]:
        """
        Rank documents for a query using RAKE phrase/word overlap.
        """
        if not self._is_loaded:
            try:
                self.preprocess()
            except Exception as e:
                return [SearchResult(
                    title=f"Error initializing RAKE: {e}",
                    score=0.0,
                    details="Please check DocumentProcessor configuration."
                )]

        if not query or not query.strip():
            return [SearchResult(title="Empty query", score=0.0, details="Please type a query.")]

        # Extract RAKE phrases from the query itself
        query_phrases_scored = self.extract_keywords_from_text(query, top_k=30)
        query_phrases = [p for p, _ in query_phrases_scored]
        query_phrase_set = set(query_phrases)

        # Also build a query word set for partial matching
        query_word_set = set()
        for p in query_phrases:
            query_word_set.update(p.split())

        results: List[Tuple[str, float, str]] = []

        for doc in self.processor.documents:
            doc_name = doc.filename
            doc_phrase_scores = self._doc_phrase_scores.get(doc_name, {})
            doc_phrases = self._doc_phrase_set.get(doc_name, set())
            doc_words = self._doc_word_set.get(doc_name, set())

            if not doc_phrase_scores:
                continue

            # Phrase overlap weighted by document phrase scores
            phrase_overlap = query_phrase_set.intersection(doc_phrases)
            phrase_score = sum(doc_phrase_scores[p] for p in phrase_overlap)

            # Word overlap bonus (helps when phrases differ but words match)
            word_overlap = query_word_set.intersection(doc_words)
            word_bonus = len(word_overlap) * 0.25  # small, to avoid overpowering phrase matches

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

        return out

    # ---------- RAKE internals ----------

    def _split_into_sentences(self, text: str) -> List[str]:
        # Slightly more robust than splitting on .!? only
        return [s.strip() for s in re.split(r'[\n.!?]+', text) if s.strip()]

    def _generate_candidate_phrases(self, text: str) -> List[str]:
        sentences = self._split_into_sentences(text)
        candidates: List[str] = []

        for sentence in sentences:
            words = sentence.lower().split()
            phrase: List[str] = []

            for w in words:
                # Remove punctuation around the word
                w = w.strip(string.punctuation)
                # Remove non-word characters inside (keeps letters/numbers/_)
                w = re.sub(r"[^\w]+", "", w)

                if (not w) or (w in self.stopwords):
                    if phrase:
                        candidates.append(" ".join(phrase))
                        phrase = []
                else:
                    phrase.append(w)

            if phrase:
                candidates.append(" ".join(phrase))

        # Filter candidates
        filtered = []
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
