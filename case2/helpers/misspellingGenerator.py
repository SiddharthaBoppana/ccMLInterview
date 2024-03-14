import random
import numpy as np
import string
from typing import List, Callable


def modifyCharacters(
    text: str, numModifications: int, maxModifications: int, probability=0.2
):
    tempText = list(text)
    if numModifications <= maxModifications:
        for _ in range(len(tempText)):

            if random.random() <= probability:

                selectedMethods = random.sample(
                    [
                        "remove",
                        "substitute",
                        # 'noise'
                    ],
                    random.randint(0, 2),
                )

                if len(selectedMethods) > 0:
                    for method in selectedMethods:
                        if numModifications <= maxModifications:

                            temp_ind = random.choice(range(0, len(tempText)))

                            if (
                                method == "remove"
                                and numModifications <= maxModifications
                            ):
                                numModifications += 1
                                tempText.pop(temp_ind)
                            elif (
                                method == "substitute"
                                and numModifications <= maxModifications
                            ):
                                numModifications += 1
                                tempText[temp_ind] = random.choice(
                                    string.ascii_letters
                                )
                            elif (
                                method == "noise"
                                and numModifications <= maxModifications
                            ):
                                numModifications += 1
                                tempText[temp_ind] = random.choice(
                                    string.punctuation
                                )
                        else:
                            break
        return "".join(tempText), numModifications
    else:
        return text, numModifications


def shuffleWords(text: str, numModifications: int, maxModifications: int):
    if numModifications <= maxModifications:
        numModifications += 1
        words = text.split(" ")
        np.random.shuffle(words)
        return " ".join(words), numModifications
    else:
        return text, numModifications


def processMisspelledVersion(
    originalText: str,
    selectedMethods: List[Callable],
    numModifications: int,
    maxModifications: int,
):
    misspelledText = f"{originalText}"
    for method in selectedMethods:
        if numModifications > maxModifications:
            break
        else:
            misspelledText, numModifications = method(
                misspelledText, numModifications, maxModifications
            )

    if (
        misspelledText == originalText
        and numModifications <= maxModifications
        and misspelledText != ""
        and originalText != ""
    ):
        return processMisspelledVersion(
            originalText,
            selectedMethods,
            numModifications,
            maxModifications=maxModifications,
        )
    else:
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


def mainProcess(
    text: str,
    maxModifications: int = 5,
):

    text = text.strip()
    misspelledText = MisspelledVersion(text, maxModifications)

    return misspelledText
