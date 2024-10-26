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


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]

    # Find all adjectives using a simple regex or POS tagging if available
    words = text.split()
    
    # Using a simple approach to locate adjectives. In a more comprehensive solution, we would use POS tagging
    # For simplicity, let's assume adjectives are words that end in "ing", "ive", "al", etc.
    adjectives = [word for word in words if re.match(r'\b\w*(ing|ive|al|ous|ful|able)\b', word)]
    
    # Shuffle the adjectives
    shuffled_adjectives = adjectives[:]
    random.shuffle(shuffled_adjectives)
    
    # Replace the original adjectives in order
    transformed_text = text
    for orig_adj, shuffled_adj in zip(adjectives, shuffled_adjectives):
        transformed_text = transformed_text.replace(orig_adj, shuffled_adj, 1)

    # Update the example with transformed text
    example["text"] = transformed_text
    text = example["text"]
    
    # Split text at common phrase separators
    phrases = text.split(", ")
    
    # Shuffle phrases
    random.shuffle(phrases)
    
    # Reconstruct the sentence
    transformed_text = ", ".join(phrases)
    
    example["text"] = transformed_text

    formal_to_informal = {
        "extraordinary": "awesome",
        "remarkable": "cool",
        "fantastic": "great",
        "film": "movie",
        "perform": "do",
        "purchase": "buy",
        "encountered": "met",
        "pleased": "happy"
    }
    words = example["text"].split()
    transformed_words = []
    
    # Set a probability threshold for transformation (e.g., 20%)
    prob_threshold = 0.2
    
    for word in words:
        # Check if the word is in the formal_to_informal dictionary
        if word in formal_to_informal and random.random() < prob_threshold:
            # Replace with the informal version
            transformed_words.append(formal_to_informal[word])
        else:
            transformed_words.append(word)
    
    # Join the words back into a transformed sentence
    example["text"] = " ".join(transformed_words)
    return example
