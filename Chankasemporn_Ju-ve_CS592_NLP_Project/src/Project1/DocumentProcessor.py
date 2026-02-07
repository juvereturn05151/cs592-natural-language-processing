"""
File Name:    DocumentProcessor.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from dataclasses import dataclass
from typing import Dict, List, Set
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from src.TokenizerHelper import TokenizerHelper, DocumentData



class DocumentProcessor:
    """
    Handles document loading, processing, and text extraction.
    """

    def __init__(self, tokenizer: TokenizerHelper = None):
        self.tokenizer = tokenizer or TokenizerHelper()
        self.documents: List[DocumentData] = []
        self.doc_names: List[str] = []
        self.corpus_terms: Set[str] = set()
        self._is_loaded = False

    def load_from_directory(self, data_dir: str, file_pattern: str = "*.txt") -> None:
        """
        Load and process all documents in the data directory.
        """
        if self._is_loaded:
            return

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        text_files = list(data_path.glob(file_pattern))
        if not text_files:
            raise FileNotFoundError(f"No text files found in {data_dir}")

        print(f"Loading {len(text_files)} documents...")

        for file_path in text_files:
            try:
                # Extract text
                text = self._extract_text_from_file(str(file_path))

                # Process text
                tokenized_doc = self.tokenizer.process_text(text, file_path)

                self.documents.append(tokenized_doc)
                self.doc_names.append(file_path.name)
                self.corpus_terms.update(tokenized_doc.unique_terms)

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

        self._is_loaded = True
        print(f"Loaded {len(self.documents)} documents successfully.")

    def load_single_document(self, file_path: str) -> DocumentData:
        """
        Load and process a single document.
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = self._extract_text_from_file(file_path)
        tokenized_doc = self.tokenizer.process_text(text, path_obj.name)
        unique_terms = set(tokenized_doc.raw_counts.keys())

        doc_data = DocumentData(
            path=file_path,
            filename=path_obj.name,
            term_frequencies=tokenized_doc.term_frequencies,
            raw_counts=tokenized_doc.raw_counts,
            tokenized_doc=tokenized_doc,
            unique_terms=unique_terms
        )

        # Update corpus if maintaining collection
        if self._is_loaded:
            self.documents.append(doc_data)
            self.doc_names.append(path_obj.name)
            self.corpus_terms.update(unique_terms)

        return doc_data

    def _extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text content from file (XML or plain text).
        """
        try:
            # Try parsing as XML first
            tree = ET.parse(file_path)
            root = tree.getroot()

            body_node = root.find('Body')
            if body_node is not None:
                # Extract text from XML structure
                text_parts = []
                item_list = body_node.findall('.//Item')
                for item in item_list:
                    if item.text:
                        text_parts.append(item.text)
                return " ".join(text_parts)
            else:
                # Fallback: extract all text content
                return ET.tostring(root, method='text', encoding='unicode')

        except ET.ParseError:
            # If it's a plain text file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def get_document_term_frequencies(self) -> Dict[str, Dict[str, float]]:
        """
        Return dictionary of {doc_name: {term: tf}} for all loaded documents.
        """
        return {doc.filename: doc.term_frequencies for doc in self.documents}

    def get_document_frequencies(self) -> Dict[str, int]:
        """
        Calculate document frequency (DF) for all terms in corpus.
        Returns: term -> number of documents containing the term
        """
        doc_freq = defaultdict(int)

        for doc in self.documents:
            for term in doc.unique_terms:
                doc_freq[term] += 1

        return dict(doc_freq)

    def get_document_by_name(self, doc_name: str) -> DocumentData:
        """
        Retrieve a document by its filename.
        """
        for doc in self.documents:
            if doc.filename == doc_name:
                return doc
        raise ValueError(f"Document '{doc_name}' not found")

    @property
    def document_count(self) -> int:
        return len(self.documents)