import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# Download NLTK data (only needed once)
nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    """Splits a sentence into words (tokens)."""
    return nltk.word_tokenize(sentence)

def stem(word):
    """Stems a word to its root form."""
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """Creates a bag-of-words representation for a sentence."""
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
