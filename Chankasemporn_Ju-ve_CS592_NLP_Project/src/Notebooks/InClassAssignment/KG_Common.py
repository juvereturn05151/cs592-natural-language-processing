"""
File Name:    KG_Common.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
def displayEntities(doc):
    """
    Display named entities in a spaCy Doc.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        The processed spaCy document.
    """
    if doc is None:
        print("No document provided.")
        return

    if not doc.ents:
        print("No named entities found.")
        return

    print("Named Entities:")
    print("-" * 60)
    for ent in doc.ents:
        print(
            f"Text: {ent.text:<25} "
            f"Label: {ent.label_:<10} "
            f"Tokens: [{ent.start}, {ent.end}) "
            f"Chars: [{ent.start_char}, {ent.end_char})"
        )