"""
File Name:    main.py.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import keyword_ui
from src.Project1.TF_IDF_Method import create_tfidf_method
from src.Project1.Rake_Method import create_rake_method
from src.Project1.Yake_Method import create_yake_method
from src.Project1.BM25_Method import create_bm25_method
from src.Project1.DocumentProcessor import DocumentProcessor
import src.NLP_Globals as Globals

def main():
    data_dir = Globals.get_default_data_dir()
    processor = DocumentProcessor()

    try:
        processor.load_all_files(data_dir, "*.txt", parallel=True)
    except Exception as e:
        print(f"Warning: Could not load documents: {e}")
        print(f"Data directory being used: {data_dir}")
        print("TF-IDF will load documents on first search.")

    methods: keyword_ui.Dict[str, keyword_ui.KeywordMethod] = {
        "TFIDF": create_tfidf_method(processor),
        "BM25": create_bm25_method(processor),
        "RAKE": create_rake_method(processor),
        "YAKE": create_yake_method(processor),
    }

    app = keyword_ui.KeywordSearchApp(methods)
    app.run()


if __name__ == "__main__":
    main()