"""
File Name:    KeywordMethod.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol, Tuple

# data model
@dataclass(frozen=True)
class SearchResult:
    """One ranked item shown in the UI."""
    title: str
    score: float
    details: str = ""

# extractor interface (plug-in later)
class KeywordMethod(Protocol):
    """
    A method takes a query string and returns ranked results.
    TODO: adapt this to call your TFIDF/RAKE/other code.
    """
    def run(self, query: str) -> List[SearchResult]:
        ...

