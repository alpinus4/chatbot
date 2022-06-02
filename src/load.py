import random
import json
import torch
from neural_net import NeuralNet
from SentenceProcessor import bag_of_words, tokenize


class Chat:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("data/intents.json","r") as f:
    intents = json.load(f)

FILE = "out/trained_data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
model.load_state_dict(model_state)
model.eval()