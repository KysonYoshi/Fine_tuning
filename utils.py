import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import re
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

NEAREST_KEYS = {
    'a': ['q', 'w', 's', 'z'],
    'b': ['v', 'g', 'h', 'n'],
    'c': ['x', 'd', 'f', 'v'],
    'd': ['s', 'e', 'f', 'c'],
    'e': ['w', 'r', 'd', 's'],
    'f': ['d', 'r', 'g', 'v'],
    'g': ['f', 't', 'h', 'b'],
    'h': ['g', 'y', 'j', 'n'],
    'i': ['u', 'o', 'k', 'j'],
    'j': ['h', 'u', 'k', 'm'],
    'k': ['j', 'i', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k'],
    'n': ['b', 'h', 'j', 'm'],
    'o': ['i', 'p', 'l', 'k'],
    'p': ['o', 'l'],
    'q': ['w', 'a'],
    'r': ['e', 't', 'd', 'f'],
    's': ['a', 'd', 'w', 'x'],
    't': ['r', 'y', 'f', 'g'],
    'u': ['y', 'i', 'j', 'h'],
    'v': ['c', 'f', 'b'],
    'w': ['q', 'e', 'a', 's'],
    'x': ['z', 's', 'd', 'c'],
    'y': ['t', 'u', 'h', 'g'],
    'z': ['a', 's', 'x'],
}


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

def replace_with_synonym(word):
    # Find synsets (synonyms) for the word using wordnet
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())

    # Filter out duplicates and keep valid synonyms
    synonyms = list(set(synonyms))

    # If no synonyms, return original word
    if len(synonyms) == 0:
        return word

    # Randomly replace the word with one of its synonyms with 20% probability
    return random.choice(synonyms) if random.random() < 0.2 else word

def introduce_typo(word):
    # Only introduce typos with a 20% probability
    if random.random() < 0.2:
        # Select a random character position in the word
        idx = random.randint(0, len(word) - 1)
        original_char = word[idx]
        
        # Check if the character has nearby keys defined
        if original_char in NEAREST_KEYS:
            # Replace it with a random nearest key
            typo_char = random.choice(NEAREST_KEYS[original_char])
            # Create the typo by replacing the character at idx
            word = word[:idx] + typo_char + word[idx + 1:]
            
    return word


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    words = example["text"].split()
    
    # Apply the synonym replacement transformation
    transformed_words = [replace_with_synonym(word) for word in words]
    
    # Join transformed words back into a sentence
    example["text"] = " ".join(transformed_words)

    words = example["text"].split()
    
    # Apply the typo introduction transformation
    transformed_words = [introduce_typo(word) for word in words]
    
    # Join transformed words back into a sentence
    example["text"] = " ".join(transformed_words)
    return example
