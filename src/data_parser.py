import json
import numpy as np
from torch.utils.data import DataLoader

from sentence_processor import SentenceProcessor
from chat_dataset import ChatDataset

class DataParser():
    def __init__(self):
        self.sp = SentenceProcessor()
        self.bag_of_words = []
        self.tags = []
        self.xy = [] # hold both patterns and the text
        self.dataset = ChatDataset()

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
        self.tags = sorted(set(self.tags))
        self.bag_of_words.append(all_words)

    def generate_training_data(self):
        # generates xtrain, ytrain to pass into dataset
        x_train = []
        y_train = []

        for(pattern_sentence, tag) in self.xy:
            bag = self.sp.bag_of_words(pattern_sentence, tag)
            x_train.append(bag)
            label = self.tags.index(tag)
            y_train.append(label)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        #hyperparameters
        batch_size = 8
        train_loader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffler=True, num_workers=2)

