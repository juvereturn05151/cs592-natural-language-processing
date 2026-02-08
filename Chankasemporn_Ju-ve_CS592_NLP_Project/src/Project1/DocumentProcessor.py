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

# Multiprocessing helpers

_WORKER_TOKENIZER: Optional[TokenizerHelper] = None

def _init_worker_tokenizer(use_stemming: bool, use_pos_tagging: bool, remove_stopwords: bool) -> None:
    """initializer runs once per worker process.creates a TokenizerHelper inside that process (prevents pickling issues)."""
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = TokenizerHelper(
        use_stemming=use_stemming,
        use_pos_tagging=use_pos_tagging,
        remove_stopwords=remove_stopwords
    )


def _extract_text_from_file(file_path: str) -> str:
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
        #plain text fallback
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


def _process_single_file(file_path_str: str) -> Tuple[Optional[DocumentData], Optional[str]]:
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
        text = _extract_text_from_file(file_path_str)
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

    def load_all_files(self,data_dir: str,file_pattern: str = "*.txt",parallel: bool = False,num_workers: Optional[int] = None, chunk_size: int = 4) -> None:
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

        start_time = time.perf_counter()

        #clear state
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
                results = pool.map(_process_single_file, file_paths, chunksize=chunk_size)

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
                    text = _extract_text_from_file(str(file_path))
                    doc_data = self.tokenizer.process_text(text, file_path)

                    self.documents.append(doc_data)
                    self.doc_names.append(file_path.name)
                    self.corpus_terms.update(doc_data.unique_terms)

                except Exception as e:
                    error_count += 1
                    print(f"Error processing {file_path.name}: {type(e).__name__}: {e}")

        self._is_loaded = True

        elapsed = time.perf_counter() - start_time
        print(
            f"Loaded {len(self.documents)} documents successfully "
            f"(parallel={parallel}) | Errors: {error_count} | "
            f"Time: {elapsed:.3f}s"
        )

    #use at keyword ui when loading a single file
    def load_one_file(self, file_path: str) -> DocumentData:
        """load and process ONE file and append it to the current corpus.
        - does NOT require calling load_from_directory again.
        - Updates: documents, doc_names, corpus_terms
        - if a document with the same filename already exists, it replaces it.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        #extract + tokenize/process
        text = _extract_text_from_file(str(p))
        doc_data = self.tokenizer.process_text(text, p)

        #replace if already present (same filename)
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

        #rebuild corpus_terms safely (simple + correct)
        self.corpus_terms.clear()
        for d in self.documents:
            self.corpus_terms.update(d.unique_terms)

        #mark as loaded (meaning: we have some docs)
        self._is_loaded = True
        return doc_data

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
