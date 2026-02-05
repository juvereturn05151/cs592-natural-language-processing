"""
File Name:    TokenizerHelper.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import nltk
import re
import string
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer


class TokenizerHelper:
    # Regex: remove common punctuation-like characters (kept close to your original)
    _CLEAN_RE = re.compile(r"[\'—_“”\"\,\?.’‘\-\)\(:!\&]")

    def __init__(self):
        # (Optional) If you run this on a machine that might not have NLTK data,
        # uncomment these downloads (or run them once elsewhere).
        # nltk.download("punkt")
        # nltk.download("stopwords")
        # nltk.download("averaged_perceptron_tagger")
        # nltk.download("wordnet")

        self.wp = WordPunctTokenizer()

        # stemmers / lemmatizer
        self.ps = PorterStemmer()
        self.ls = LancasterStemmer()
        self.sb = SnowballStemmer("english")
        self.wl = WordNetLemmatizer()

        # stopwords
        self.stopWordList = set(stopwords.words("english"))

    def _clean_token(self, word: str) -> str:
        """Lowercase + strip punctuation artifacts used by this assignment."""
        return self._CLEAN_RE.sub("", word).strip()

    def getTokensSplit(self, item):
        textBlock = item.text
        return textBlock.split()

    def getTokens_NLTK_Tokenize(self, item):
        textBlock = item.text.lower()
        return nltk.tokenize.word_tokenize(textBlock)

    def getTokens_NLTK_PunktTokenize(self, item):
        textBlock = item.text.lower()

        tokenlist = []
        for word in self.wp.tokenize(textBlock):
            if word in self.stopWordList:
                continue
            if word in string.punctuation:
                continue

            cleaned = self._clean_token(word)
            if cleaned == "":
                continue

            tokenlist.append(cleaned)

        return tokenlist

    def _penn_to_wordnet_pos(self, penn_tag: str):
        """Map Penn Treebank POS tags to WordNet POS tags for better lemmatization."""
        if penn_tag.startswith("J"):
            return "a"  # adjective
        if penn_tag.startswith("V"):
            return "v"  # verb
        if penn_tag.startswith("N"):
            return "n"  # noun
        if penn_tag.startswith("R"):
            return "r"  # adverb
        return "n"      # default to noun

    def getTokens_NLTK_Stemmer(self, item, choice):
        """
        choice:
        0 = Porter Stemmer
        1 = Lancaster Stemmer
        2 = WordNet Lemmatizer (with POS mapping)
        3 = Snowball Stemmer
        """
        # Tokenize first (already lowercased + stopwords removed)
        tokens = self.getTokens_NLTK_PunktTokenize(item)

        # POS tag tokens
        tagged = pos_tag(tokens)

        stemWordList = []
        validTagList = {"NN", "NNP", "NNPS", "NNS", "CD", "FW", "JJ", "JJR", "JJS"}

        for token, tag in tagged:
            # token should already be cleaned, but keep it safe
            if token == "" or token in string.punctuation or token in self.stopWordList:
                continue

            if tag not in validTagList:
                continue

            if choice == 0:
                out = self.ps.stem(token)
            elif choice == 1:
                out = self.ls.stem(token)
            elif choice == 2:
                wn_pos = self._penn_to_wordnet_pos(tag)
                out = self.wl.lemmatize(token, pos=wn_pos)
            elif choice == 3:
                out = self.sb.stem(token)
            else:
                raise ValueError("choice must be 0 (Porter), 1 (Lancaster), 2 (Lemma), or 3 (Snowball)")

            if out != "":
                stemWordList.append(out)

        return stemWordList

    def buildWordMap(self, documentTokens):
        wordMap = {}
        for word in documentTokens:
            wordMap[word] = wordMap.get(word, 0) + 1
        return wordMap

    def printSortedMap(self, inputMap):
        dummyList = [[value, key] for key, value in inputMap.items()]
        dummyList.sort(reverse=True)
        print(dummyList)
        return dummyList
