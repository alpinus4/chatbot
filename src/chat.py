import json
import os

import torch
import random
from neural_net import NeuralNet
import config as c
from sentence_processor import SentenceProcessor


class Chat:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNet(1, 1, 1)
        self.sp = SentenceProcessor()
        self.all_words = []
        self.tags = []
        with open(c.INTENTS_DATA[0], "r") as f:
            self.intents = json.load(f)

    def load_data(self):
        data = torch.load(os.path.join(c.OUT_PATH, c.TRAINED_DATA_FILENAME))
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]
        model_state = data["model_state"]

        self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

    def get_response(self, msg):
        sentence = self.sp.tokenize(msg)
        X = self.sp.bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.55:
            for intent in self.intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        else:
            return "I do not understand..."


def main():
    chatting = Chat()
    chatting.load_data()
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        else:
            print(f"{c.BOT_NAME}: {chatting.get_response(sentence)}")


if __name__ == "__main__":
    main()
