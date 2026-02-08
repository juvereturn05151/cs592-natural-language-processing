"""
File Name:    DocumentProcessor.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from typing import Dict, List, Set, Optional, Tuple
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import time

from src.TokenizerHelper import TokenizerHelper, DocumentData

# ---------------------------
# Multiprocessing helpers
# ---------------------------
# NOTE: These must be TOP-LEVEL functions (not methods) so Pool can pickle them.

_WORKER_TOKENIZER: Optional[TokenizerHelper] = None


def _init_worker_tokenizer(use_stemming: bool, use_pos_tagging: bool, remove_stopwords: bool) -> None:
    """
    Initializer runs once per worker process.
    Creates a TokenizerHelper inside that process (prevents pickling issues).
    """
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = TokenizerHelper(
        use_stemming=use_stemming,
        use_pos_tagging=use_pos_tagging,
        remove_stopwords=remove_stopwords
    )


def _extract_text_from_file_worker(file_path: str) -> str:
    """Same logic as DocumentProcessor._extract_text_from_file but top-level."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        body_node = root.find('Body')
        if body_node is not None:
            text_parts = []
            item_list = body_node.findall('.//Item')
            for item in item_list:
                if item.text:
                    text_parts.append(item.text)
            return " ".join(text_parts)
        else:
            return ET.tostring(root, method='text', encoding='unicode')

    except ET.ParseError:
        #plain text fallback
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


def _process_one_file(file_path_str: str) -> Tuple[Optional[DocumentData], Optional[str]]:
    """
    Worker job:
      - extract raw text
      - tokenize/process into DocumentData
      - return (doc_data, error_message)
    """
    global _WORKER_TOKENIZER
    try:
        if _WORKER_TOKENIZER is None:
            return None, "Worker tokenizer not initialized. Did Pool initializer run?"

        p = Path(file_path_str)
        text = _extract_text_from_file_worker(file_path_str)

        doc_data = _WORKER_TOKENIZER.process_text(text, p)
        return doc_data, None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


class DocumentProcessor:
    """Handles document loading, processing, and text extraction."""

    def __init__(self, tokenizer: TokenizerHelper = None):
        self.tokenizer = tokenizer or TokenizerHelper()
        self.documents: List[DocumentData] = []
        self.doc_names: List[str] = []
        self.corpus_terms: Set[str] = set()
        self._is_loaded = False

    def _extract_text_from_file(self, file_path: str) -> str:
        """ Return raw text from file. Tries XML <Body><Item> first; else uses XML text flattening; else plain text."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            body_node = root.find('Body')
            if body_node is not None:
                text_parts = []
                item_list = body_node.findall('.//Item')
                for item in item_list:
                    if item.text:
                        text_parts.append(item.text)
                return " ".join(text_parts)
            else:
                return ET.tostring(root, method='text', encoding='unicode')

        except ET.ParseError:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

    def load_from_directory(self,data_dir: str,file_pattern: str = "*.txt",parallel: bool = False,num_workers: Optional[int] = None, chunksize: int = 4) -> None:
        """Load and process all documents in the data directory."""
        if self._is_loaded:
            return

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        text_files = list(data_path.glob(file_pattern))
        if not text_files:
            raise FileNotFoundError(f"No files found in {data_dir} matching '{file_pattern}'")

        print(f"Loading {len(text_files)} documents... (parallel={parallel})")

        start_time = time.perf_counter()  # <-- ADDED

        # Clear state (in case this object is reused)
        self.documents.clear()
        self.doc_names.clear()
        self.corpus_terms.clear()

        if parallel:
            from multiprocessing import Pool, cpu_count

            file_paths = [str(p) for p in text_files]

            if num_workers is None:
                num_workers = min(cpu_count(), max(1, len(file_paths)))

            use_stemming = bool(getattr(self.tokenizer, "use_stemming", True))
            use_pos_tagging = bool(getattr(self.tokenizer, "use_pos_tagging", True))
            remove_stopwords = bool(getattr(self.tokenizer, "remove_stopwords", True))

            with Pool(
                processes=num_workers,
                initializer=_init_worker_tokenizer,
                initargs=(use_stemming, use_pos_tagging, remove_stopwords),
            ) as pool:
                results = pool.map(_process_one_file, file_paths, chunksize=chunksize)

            error_count = 0
            for (doc_data, err), file_path in zip(results, text_files):
                if err is not None or doc_data is None:
                    error_count += 1
                    print(f"Error processing {file_path.name}: {err}")
                    continue

                self.documents.append(doc_data)
                self.doc_names.append(doc_data.filename)
                self.corpus_terms.update(doc_data.unique_terms)

        else:
            error_count = 0
            for file_path in text_files:
                try:
                    text = self._extract_text_from_file(str(file_path))
                    doc_data = self.tokenizer.process_text(text, file_path)

                    self.documents.append(doc_data)
                    self.doc_names.append(file_path.name)
                    self.corpus_terms.update(doc_data.unique_terms)

                except Exception as e:
                    error_count += 1
                    print(f"Error processing {file_path.name}: {type(e).__name__}: {e}")

        self._is_loaded = True

        elapsed = time.perf_counter() - start_time  # <-- ADDED
        print(
            f"Loaded {len(self.documents)} documents successfully "
            f"(parallel={parallel}) | Errors: {error_count} | "
            f"Time: {elapsed:.3f}s"
        )

    # ---------------------------
    # Incremental loading (DROP-IN)
    # ---------------------------

    def load_one_file(self, file_path: str) -> DocumentData:
        """
        Load and process ONE file and append it to the current corpus.

        - Does NOT require calling load_from_directory again.
        - Updates: documents, doc_names, corpus_terms
        - If a document with the same filename already exists, it replaces it.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract + tokenize/process
        text = self._extract_text_from_file(str(p))
        doc_data = self.tokenizer.process_text(text, p)

        # Replace if already present (same filename)
        existing_idx = None
        for i, d in enumerate(self.documents):
            if d.filename == doc_data.filename:
                existing_idx = i
                break

        if existing_idx is not None:
            self.documents[existing_idx] = doc_data
            self.doc_names[existing_idx] = doc_data.filename
        else:
            self.documents.append(doc_data)
            self.doc_names.append(doc_data.filename)

        # Rebuild corpus_terms safely (simple + correct)
        self.corpus_terms.clear()
        for d in self.documents:
            self.corpus_terms.update(d.unique_terms)

        # Mark as loaded (meaning: we have some docs)
        self._is_loaded = True
        return doc_data

    def load_files(self, file_paths: List[str], parallel: bool = False) -> int:
        """
        Load and process MANY files incrementally.

        Returns number of successfully added/updated docs.

        Note: parallel=True is supported but optional; for small batches, parallel=False is fine.
        """
        if not file_paths:
            return 0

        # Simple sequential (most robust)
        if not parallel:
            ok = 0
            for fp in file_paths:
                try:
                    self.load_one_file(fp)
                    ok += 1
                except Exception as e:
                    print(f"[load_files] Error processing {fp}: {type(e).__name__}: {e}")
            return ok

        # Optional parallel version (uses your existing worker helpers)
        from multiprocessing import Pool, cpu_count

        paths = [str(Path(p)) for p in file_paths]
        use_stemming = bool(getattr(self.tokenizer, "use_stemming", True))
        use_pos_tagging = bool(getattr(self.tokenizer, "use_pos_tagging", True))
        remove_stopwords = bool(getattr(self.tokenizer, "remove_stopwords", True))

        num_workers = min(cpu_count(), max(1, len(paths)))
        with Pool(
            processes=num_workers,
            initializer=_init_worker_tokenizer,
            initargs=(use_stemming, use_pos_tagging, remove_stopwords),
        ) as pool:
            results = pool.map(_process_one_file, paths, chunksize=4)

        ok = 0
        for (doc_data, err), fp in zip(results, paths):
            if err is not None or doc_data is None:
                print(f"[load_files] Error processing {fp}: {err}")
                continue

            # Insert/replace by filename
            existing_idx = None
            for i, d in enumerate(self.documents):
                if d.filename == doc_data.filename:
                    existing_idx = i
                    break

            if existing_idx is not None:
                self.documents[existing_idx] = doc_data
                self.doc_names[existing_idx] = doc_data.filename
            else:
                self.documents.append(doc_data)
                self.doc_names.append(doc_data.filename)

            ok += 1

        # Rebuild corpus_terms safely
        self.corpus_terms.clear()
        for d in self.documents:
            self.corpus_terms.update(d.unique_terms)

        self._is_loaded = True
        return ok


    def get_document_term_frequencies(self) -> Dict[str, Dict[str, float]]:
        return {doc.filename: doc.term_frequencies for doc in self.documents}

    def get_document_frequencies(self) -> Dict[str, int]:
        doc_freq = defaultdict(int)
        for doc in self.documents:
            for term in doc.unique_terms:
                doc_freq[term] += 1
        return dict(doc_freq)

    def get_document_by_name(self, doc_name: str) -> DocumentData:
        for doc in self.documents:
            if doc.filename == doc_name:
                return doc
        raise ValueError(f"Document '{doc_name}' not found")

    @property
    def document_count(self) -> int:
        return len(self.documents)
