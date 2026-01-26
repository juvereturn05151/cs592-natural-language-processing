"""
File Name:    TokenizerHelper.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import nltk

def getTokensSplit(item):
    textBlock = item.text
    # 1. Using the built-in tokenizer from Python string
    tokenlist = textBlock.split()
    return tokenlist

def buildWordMap(documentTokens):
    wordMap = dict()
    for word in documentTokens:
        if word in wordMap :
            wordMap[word] += 1
        else:
            wordMap[word] = 1
    return wordMap

# This is a hack to print the words from the dictionary into a sorted order
def printSortedMap(inputMap):
    dummyList = []
    for key, value in inputMap.items():
        dummyList.append([value, key])
    dummyList = sorted(dummyList, reverse=True)
    print( dummyList )
    return dummyList

def getTokens_NLTK_Tokenize(item):
    textBlock = item.text.lower()
        # 2. Now using nltk.tokenize to get keywords
    tokenlist = nltk.tokenize.word_tokenize(textBlock)
    return tokenlist