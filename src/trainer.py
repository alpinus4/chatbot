import torch
from torch.utils.data import DataLoader

from neural_net import NeuralNet

class Trainer:
    def __init__(self, dataset, input_size, output_size, hidden_size, batch_size, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = []

    def train(self, num_epochs = 1000):
        for epoch in range(num_epochs):
            for (words, labels) in self.loader:
                words = words.to(self.device)
                labels = labels.to(dtype=torch.long).to(self.device)

                outputs = self.model(words)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            #if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            self.loss.append(loss.item())
        print(f'final loss: {loss.item():.4f}')

        return {
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size
        }