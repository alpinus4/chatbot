import nltk
from nltk.stem.porter import PorterStemmer

class SentenceProcessor:
    def __init__(self):
        pass

    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def stem(self, word):
        stemmer = PorterStemmer()
        return stemmer.stem(word.lower())

    def bag_of_words(self):
        pass
