# misspelling generator - optimised version 
import random
import string
import numpy as np
from typing import List, Callable


def modifyCharacters(text: str, numModifications: int, maxModifications: int, probability=0.2):
    tempText = list(text)
    methods = ["remove", "substitute", "noise"]
    if numModifications <= maxModifications:
        for _ in range(len(tempText)):
            if random.random() <= probability:
                method = methods[random.randint(0, len(methods)-1)]
                temp_ind = random.choice(range(0, len(tempText)))
                if method == "remove" and numModifications <= maxModifications:
                    numModifications += 1
                    del tempText[temp_ind]
                elif method == "substitute" and numModifications <= maxModifications:
                    numModifications += 1
                    tempText[temp_ind] = random.choice(string.ascii_letters)
                elif method == "noise" and numModifications <= maxModifications:
                    numModifications += 1
                    tempText[temp_ind] = random.choice(string.punctuation)
    return "".join(tempText), numModifications

def shuffleWords(text: str, numModifications: int, maxModifications: int):
    if numModifications <= maxModifications:
        numModifications += 1
        words = text.split(" ")
        random.shuffle(words)
        return " ".join(words), numModifications
    else:
        return text, numModifications

def processMisspelledVersion(originalText: str, selectedMethods: List[Callable], numModifications: int, maxModifications: int):
    misspelledText = originalText
    while misspelledText == originalText and numModifications <= maxModifications and misspelledText != "" and originalText != "":
        for method in selectedMethods:
            if numModifications > maxModifications:
                break
            else:
                misspelledText, numModifications = method(misspelledText, numModifications, maxModifications)
    return misspelledText


def MisspelledVersion(originalText: str, maxModifications: int = 5):

    numModifications = 0

    selectedMethods = random.sample(
        [
            modifyCharacters,
            shuffleWords,
        ],
        random.randint(1, 2),
    )

    misspelledText = processMisspelledVersion(
        originalText, 
        selectedMethods, 
        numModifications, 
        maxModifications
    )

    return misspelledText


def mainProcess(args):
    text, maxModifications = args
    text = text.strip()
    misspelledText = MisspelledVersion(text, maxModifications)
    return misspelledText
