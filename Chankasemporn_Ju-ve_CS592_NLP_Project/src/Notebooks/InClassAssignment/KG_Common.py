"""
File Name:    KG_Common.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import spacy
from spacy.tokens import Span

def merge_tokens_to_ner(doc, token_list, label="PERSON"):
    """
    Merge consecutive tokens into a single named entity span.

    Parameters
    ----------
    doc : spacy.tokens.Doc
    token_list : list[str]
        Tokens to merge (e.g., ["DON", "PEDRO"])
    label : str
        NER label to assign (default: PERSON)

    Returns
    -------
    doc : spacy.tokens.Doc
    """

    phrase = " ".join(token_list)
    phrase_len = len(token_list)

    new_ents = list(doc.ents)

    i = 0
    while i < len(doc):
        # Check if tokens match the phrase
        match = True
        for j in range(phrase_len):
            if i + j >= len(doc) or doc[i + j].text.upper() != token_list[j]:
                match = False
                break

        if match:
            span = Span(doc, i, i + phrase_len, label=label)

            # Remove overlapping entities
            new_ents = [ent for ent in new_ents if not (ent.start < span.end and span.start < ent.end)]

            new_ents.append(span)
            i += phrase_len
        else:
            i += 1

    doc.ents = new_ents
    return doc

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