"""
File Name:    keyword_ui.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
from __future__ import annotations
"""
Keyword Extraction UI Template (Tkinter)

Required (assignment):
(a) User types a phrase with one or more keywords
(b) User selects 1 of 4 methods
(c) User clicks a button
(d) App displays ranked results (highest score first)

Added:
- RIGHT PANEL:
  - Shows CURRENT LOADED FILES from your shared DocumentProcessor
  - Button: "Add files..." to browse and load new .txt files into the processor
  - File preview panel
- Assumption:
  - You pass the SAME DocumentProcessor instance to all methods
  - Each method has a .processor attribute pointing to that shared instance
    (true for your TFIDF/BM25/RAKE/YAKE code)
"""

from dataclasses import dataclass
from typing import Dict, List, Protocol, Optional
import os
import tkinter as tk
from tkinter import ttk, filedialog


# ---------------- Data model ----------------

@dataclass(frozen=True)
class SearchResult:
    """One ranked item shown in the UI."""
    title: str
    score: float
    details: str = ""


# ---------------- Method interface ----------------

class KeywordMethod(Protocol):
    def run(self, query: str) -> List[SearchResult]:
        ...


# ---------------- App ----------------

class KeywordSearchApp:
    def __init__(self, methods: Dict[str, KeywordMethod]):
        if not methods:
            raise ValueError("methods must not be empty")

        self.methods = methods

        # Best-effort: grab the shared processor from the first method
        self.processor = self._try_get_shared_processor()

        # Tk setup
        self.root = tk.Tk()
        self.root.title("Keyword Search")

        # Vars
        self.query_var = tk.StringVar(value="")
        self.method_var = tk.StringVar(value=list(methods.keys())[0])

        # Keep last results
        self._last_results: List[SearchResult] = []

        # Build UI
        self._build_layout()

        # Load file list at start
        self._refresh_loaded_files()

    # ---------------- UI Layout ----------------

    def _build_layout(self) -> None:
        # Root: LEFT (search) + RIGHT (files)
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(6, weight=1)
        left.rowconfigure(8, weight=1)

        right = ttk.Frame(self.root, padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)  # list grows
        right.rowconfigure(4, weight=1)  # preview grows

        # ---------- LEFT ----------
        ttk.Label(left, text="Enter a phrase / keywords:").grid(row=0, column=0, sticky="w")
        query_entry = ttk.Entry(left, textvariable=self.query_var)
        query_entry.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        query_entry.bind("<Return>", lambda e: self.on_search())

        ttk.Label(left, text="Choose a method:").grid(row=2, column=0, sticky="w")
        method_combo = ttk.Combobox(
            left,
            textvariable=self.method_var,
            values=list(self.methods.keys()),
            state="readonly",
        )
        method_combo.grid(row=3, column=0, sticky="ew", pady=(4, 10))

        ttk.Button(left, text="Show me the results", command=self.on_search).grid(
            row=4, column=0, sticky="ew", pady=(0, 10)
        )

        ttk.Label(left, text="Ranked results:").grid(row=5, column=0, sticky="w")
        self.results_list = tk.Listbox(left, height=10)
        self.results_list.grid(row=6, column=0, sticky="nsew")
        self.results_list.bind("<<ListboxSelect>>", self.on_select_result)

        ttk.Label(left, text="Details:").grid(row=7, column=0, sticky="w", pady=(10, 0))
        self.details_text = tk.Text(left, height=6, wrap="word")
        self.details_text.grid(row=8, column=0, sticky="nsew", pady=(4, 0))

        # ---------- RIGHT ----------
        header = ttk.Frame(right)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Loaded files:").grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Refresh", command=self._refresh_loaded_files).grid(row=0, column=1, sticky="e")

        btn_row = ttk.Frame(right)
        btn_row.grid(row=1, column=0, sticky="ew", pady=(6, 8))
        btn_row.columnconfigure(0, weight=1)
        ttk.Button(btn_row, text="Add files...", command=self._add_files_dialog).grid(row=0, column=0, sticky="ew")

        # File list + scrollbar
        list_frame = ttk.Frame(right)
        list_frame.grid(row=2, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.files_list = tk.Listbox(list_frame, height=12)
        self.files_list.grid(row=0, column=0, sticky="nsew")
        sbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.files_list.yview)
        sbar.grid(row=0, column=1, sticky="ns")
        self.files_list.configure(yscrollcommand=sbar.set)
        self.files_list.bind("<<ListboxSelect>>", self.on_select_file)

        # Preview
        ttk.Label(right, text="File preview:").grid(row=3, column=0, sticky="w", pady=(10, 0))

        preview_frame = ttk.Frame(right)
        preview_frame.grid(row=4, column=0, sticky="nsew")
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.preview_text = tk.Text(preview_frame, wrap="word")
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        psbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_text.yview)
        psbar.grid(row=0, column=1, sticky="ns")
        self.preview_text.configure(yscrollcommand=psbar.set)

        # Status line
        self.status_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.status_var, foreground="gray").grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )

    # ---------------- Search ----------------

    def on_search(self) -> None:
        query = self.query_var.get().strip()
        method_name = self.method_var.get()

        method = self.methods.get(method_name)
        if method is None:
            self._set_details(f"Unknown method: {method_name}")
            return

        results = method.run(query)
        results.sort(key=lambda r: r.score, reverse=True)

        self._last_results = results
        self._populate_results(results)

        if not results:
            self._set_details("No results. Try typing a phrase (e.g., 'frankenstein monster').")

    def _populate_results(self, results: List[SearchResult]) -> None:
        self.results_list.delete(0, tk.END)
        for i, r in enumerate(results, start=1):
            self.results_list.insert(tk.END, f"{i}. ({r.score:.4f}) {r.title}")
        self._set_details("")

    def on_select_result(self, _event=None) -> None:
        sel = self.results_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < 0 or idx >= len(self._last_results):
            return

        r = self._last_results[idx]
        self._set_details(f"Title: {r.title}\nScore: {r.score:.6f}\n\n{r.details}")

        # Auto-select matching filename on the right if present
        # (your methods typically use doc filename as SearchResult.title)
        self._try_select_loaded_file_by_name(r.title)

    def _set_details(self, text: str) -> None:
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert(tk.END, text)

    # ---------------- Processor discovery ----------------

    def _try_get_shared_processor(self):
        # Most of your methods are classes with .processor attribute
        for m in self.methods.values():
            p = getattr(m, "processor", None)
            if p is not None:
                return p
        return None

    # ---------------- Loaded files panel ----------------

    def _refresh_loaded_files(self) -> None:
        self.files_list.delete(0, tk.END)
        self.preview_text.delete("1.0", tk.END)

        if self.processor is None:
            self.status_var.set("No shared DocumentProcessor found on methods (expected .processor).")
            return

        docs = getattr(self.processor, "documents", None)
        if not docs:
            self.status_var.set("No documents currently loaded.")
            return

        for d in docs:
            # Document object assumed to have filename and path (your code does)
            self.files_list.insert(tk.END, d.filename)

        self.status_var.set(f"{len(docs)} documents loaded.")

    def _try_select_loaded_file_by_name(self, name: str) -> None:
        if self.processor is None:
            return
        docs = getattr(self.processor, "documents", None) or []
        names = [d.filename for d in docs]
        if name not in names:
            return
        idx = names.index(name)
        self.files_list.selection_clear(0, tk.END)
        self.files_list.selection_set(idx)
        self.files_list.see(idx)
        self._load_preview_by_index(idx)

    def on_select_file(self, _event=None) -> None:
        sel = self.files_list.curselection()
        if not sel:
            return
        idx = sel[0]
        self._load_preview_by_index(idx)

    def _load_preview_by_index(self, idx: int, max_chars: int = 12000) -> None:
        self.preview_text.delete("1.0", tk.END)

        if self.processor is None:
            self.preview_text.insert(tk.END, "(No DocumentProcessor.)")
            return

        docs = getattr(self.processor, "documents", None) or []
        if idx < 0 or idx >= len(docs):
            return

        doc = docs[idx]
        path = getattr(doc, "path", "")
        text = getattr(doc, "text", "")

        # Prefer showing file content from doc.text (already loaded)
        content = text or ""
        if not content and path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(max_chars + 1)
            except Exception as e:
                content = f"(Could not read file)\n{path}\n\nError: {e}"

        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n... (truncated preview) ..."

        self.preview_text.insert(
            tk.END,
            f"Filename: {getattr(doc, 'filename', '(unknown)')}\n"
            f"Path: {path}\n\n"
            f"{content}"
        )

    # ---------------- Add files ----------------

    def _add_files_dialog(self) -> None:
        """
        Browse and add new files to the shared DocumentProcessor.

        Works with the drop-in DocumentProcessor incremental APIs:
          - processor.load_files(file_paths, parallel=False/True)  [preferred]
          - processor.load_one_file(file_path)                    [fallback]

        If neither exists, falls back to load_from_directory.
        """
        if self.processor is None:
            self.status_var.set("Cannot add files: no shared DocumentProcessor found.")
            return

        paths = filedialog.askopenfilenames(
            title="Select text files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not paths:
            return

        added = 0
        errors: List[str] = []

        # 1) Preferred: incremental batch API (from the drop-in)
        if hasattr(self.processor, "load_files"):
            try:
                # DocumentProcessor.load_files returns count (in our drop-in)
                added = int(self.processor.load_files(list(paths), parallel=False))  # type: ignore
            except Exception as e:
                errors.append(f"load_files failed: {type(e).__name__}: {e}")

        # 2) Fallback: incremental single-file API (from the drop-in)
        elif hasattr(self.processor, "load_one_file"):
            for p in paths:
                try:
                    self.processor.load_one_file(p)  # type: ignore
                    added += 1
                except Exception as e:
                    errors.append(f"{os.path.basename(p)}: {type(e).__name__}: {e}")

        # 3) Back-compat: your older APIs (if they exist in some version)
        elif hasattr(self.processor, "load_from_files"):
            try:
                self.processor.load_from_files(list(paths), parallel=False)  # type: ignore
                added = len(paths)
            except Exception as e:
                errors.append(f"load_from_files failed: {type(e).__name__}: {e}")

        elif hasattr(self.processor, "add_file"):
            for p in paths:
                try:
                    self.processor.add_file(p)  # type: ignore
                    added += 1
                except Exception as e:
                    errors.append(f"{os.path.basename(p)}: {type(e).__name__}: {e}")

        # 4) Last resort: directory load (may load extra txt files)
        else:
            try:
                dirs = sorted(set(os.path.dirname(p) for p in paths))
                for d in dirs:
                    try:
                        self.processor.load_from_directory(d, "*.txt", parallel=True)  # type: ignore
                    except Exception as e:
                        errors.append(f"load_from_directory({d}) failed: {type(e).__name__}: {e}")
                added = len(paths)
            except Exception as e:
                errors.append(f"fallback failed: {type(e).__name__}: {e}")

        # Refresh right panel list
        self._refresh_loaded_files()

        # Status
        msg = f"Added/updated {added} file(s)."
        if errors:
            msg += f" Errors: {len(errors)} (see console)."
            for e in errors[:20]:
                print("[Add files error]", e)

        self.status_var.set(msg)

    # ---------------- Run ----------------

    def run(self) -> None:
        self.root.mainloop()
