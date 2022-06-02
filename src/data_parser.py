import json
import numpy as np

from sentence_processor import SentenceProcessor

class DataParser():
    def __init__(self):
        self.sp = SentenceProcessor()
        self.tags = []
        self.xy = [] # hold both patterns and the text
        self.all_words = []

    def load(self, path):
        with open(path, 'r') as f:
            intents = json.load(f)

        all_words = []
        for intent in intents['intents']:
            tag = intent['tag']
            self.tags.append(tag)
            for pattern in intent['patterns']:
                w = self.sp.tokenize(pattern)
                all_words.extend(w)
                self.xy.append((w, tag))

        ignore_words = ['?', '!', '.', ',']
        all_words = [self.sp.stem(w) for w in all_words if w not in ignore_words]
        all_words = sorted(set(all_words))  # removing duplicates
        self.all_words.extend(all_words)
        self.tags.extend(sorted(set(self.tags)))

    def generate_training_data(self):
        # generates xtrain, ytrain to pass into dataset
        x_train = []
        y_train = []

        for(pattern_sentence, tag) in self.xy:
            bag = self.sp.bag_of_words(pattern_sentence, self.all_words)
            x_train.append(bag)
            label = self.tags.index(tag)
            y_train.append(label)

        return np.array(x_train), np.array(y_train)

