"""
File Name:    main.py.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import keyword_ui
from src.Project1.TF_IDF_Method import create_tfidf_method

def main():
    # 4 placeholder methods (swap these out later with real TFIDF/RAKE/etc.)
    methods: keyword_ui.Dict[str, keyword_ui.KeywordMethod] = {
        "TFIDF": create_tfidf_method(),
        "RAKE (placeholder)": keyword_ui.PlaceholderMethod("RAKE"),
        "Team Method #1 (placeholder)": keyword_ui.PlaceholderMethod("Team1"),
        "Team Method #2 (placeholder)": keyword_ui.PlaceholderMethod("Team2"),
    }

    app = keyword_ui.KeywordSearchApp(methods)
    app.run()


if __name__ == "__main__":
    main()