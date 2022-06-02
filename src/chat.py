import random
import json
import torch
from neural_net import NeuralNet
from SentenceProcessor import bag_of_words, tokenize

bot_name = "Shrek"
print("Welcome to our grocery store! Type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy()

    output = model(X)
    _, predicted = torch.max(output,dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent["responses"])}")
            else:
                print(f"{bot_name}: I don't understand...")