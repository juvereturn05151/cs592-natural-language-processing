"""
File Name:    keyword_ui.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
from __future__ import annotations
"""
Keyword Extraction UI Template (Tkinter)

Features required by the assignment:
(a) User types a phrase with one or more keywords
(b) User selects 1 of 4 methods
(c) User clicks a button
(d) App displays ranked results (highest score first)

The 4 methods are placeholders; plug in real implementations later.
"""


from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol, Tuple
import tkinter as tk
from tkinter import ttk


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class SearchResult:
    """One ranked item shown in the UI."""
    title: str
    score: float
    details: str = ""  # optional: snippet, matched keywords, etc.


# -----------------------------
# Extractor interface (plug-in later)
# -----------------------------
class KeywordMethod(Protocol):
    """
    A method takes a query string and returns ranked results.
    You can later adapt this to call your TFIDF/RAKE/other code.
    """
    def run(self, query: str) -> List[SearchResult]:
        ...


class PlaceholderMethod:
    """
    Dummy method for now.
    Replace the run() body later with your real keyword extraction + search.
    """
    def __init__(self, name: str):
        self.name = name

    def run(self, query: str) -> List[SearchResult]:
        q = (query or "").strip()

        # Keep this simple: deterministic “fake” ranking so the UI works now.
        if not q:
            return []

        tokens = [t for t in q.split() if t]
        # Example placeholder results; replace with real results later.
        results = [
            SearchResult(title=f"[{self.name}] Result A for: {q}", score=1.0 + 0.10 * len(tokens),
                         details=f"Matched tokens: {', '.join(tokens[:5])}"),
            SearchResult(title=f"[{self.name}] Result B for: {q}", score=0.8 + 0.05 * len(tokens),
                         details="(placeholder details)"),
            SearchResult(title=f"[{self.name}] Result C for: {q}", score=0.6 + 0.02 * len(tokens),
                         details="(placeholder details)"),
        ]

        # Ensure sorted high → low (requirement d)
        results.sort(key=lambda r: r.score, reverse=True)
        return results


# -----------------------------
# Controller / App class (the “class that supports the UI”)
# -----------------------------
class KeywordSearchApp:
    def __init__(self, methods: Dict[str, KeywordMethod]):
        if not methods:
            raise ValueError("methods must not be empty")

        self.methods = methods

        # Tk setup
        self.root = tk.Tk()
        self.root.title("Keyword Search (Template)")

        # Variables
        self.query_var = tk.StringVar(value="")
        self.method_var = tk.StringVar(value=list(methods.keys())[0])

        # Build UI
        self._build_layout()

    def _build_layout(self) -> None:
        # Main frame
        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        # Query
        ttk.Label(container, text="Enter a phrase / keywords:").grid(
            row=0, column=0, sticky="w"
        )
        query_entry = ttk.Entry(container, textvariable=self.query_var)
        query_entry.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        query_entry.bind("<Return>", lambda e: self.on_search())  # press Enter to search

        # Method selector
        ttk.Label(container, text="Choose a method:").grid(row=2, column=0, sticky="w")
        method_combo = ttk.Combobox(
            container,
            textvariable=self.method_var,
            values=list(self.methods.keys()),
            state="readonly",
        )
        method_combo.grid(row=3, column=0, sticky="ew", pady=(4, 10))

        # Button
        btn = ttk.Button(container, text="Show me the results", command=self.on_search)
        btn.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        # Results list
        ttk.Label(container, text="Ranked results:").grid(row=5, column=0, sticky="w")

        # Use a Listbox for simplicity (you can swap to Treeview later)
        self.results_list = tk.Listbox(container, height=10)
        self.results_list.grid(row=6, column=0, sticky="nsew")
        container.rowconfigure(6, weight=1)

        # Optional details panel
        ttk.Label(container, text="Details:").grid(row=7, column=0, sticky="w", pady=(10, 0))
        self.details_text = tk.Text(container, height=6, wrap="word")
        self.details_text.grid(row=8, column=0, sticky="nsew", pady=(4, 0))
        container.rowconfigure(8, weight=1)

        # When selecting an item, show details
        self.results_list.bind("<<ListboxSelect>>", self.on_select_result)

        # Keep last results
        self._last_results: List[SearchResult] = []

    def on_search(self) -> None:
        query = self.query_var.get().strip()
        method_name = self.method_var.get()

        method = self.methods.get(method_name)
        if method is None:
            self._set_details(f"Unknown method: {method_name}")
            return

        results = method.run(query)
        # Already sorted high → low by the method; but safe to enforce here too.
        results.sort(key=lambda r: r.score, reverse=True)

        self._last_results = results
        self._populate_results(results)

        if not results:
            self._set_details("No results. Try typing a phrase (e.g., 'Best breakfast in London').")

    def _populate_results(self, results: List[SearchResult]) -> None:
        self.results_list.delete(0, tk.END)
        for i, r in enumerate(results, start=1):
            self.results_list.insert(tk.END, f"{i}. ({r.score:.4f}) {r.title}")
        self._set_details("")  # clear details on new search

    def on_select_result(self, _event=None) -> None:
        sel = self.results_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < 0 or idx >= len(self._last_results):
            return
        r = self._last_results[idx]
        self._set_details(f"Title: {r.title}\nScore: {r.score:.6f}\n\n{r.details}")

    def _set_details(self, text: str) -> None:
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert(tk.END, text)

    def run(self) -> None:
        self.root.mainloop()