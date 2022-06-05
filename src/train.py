import matplotlib.pyplot as plt
import torch
import os

from data_parser import DataParser
from chat_dataset import ChatDataset
from trainer import Trainer
import config as c


def main():
    parser = DataParser()
    for data_path in c.INTENTS_DATA:
        parser.load(data_path)

    x_train, y_train = parser.generate_training_data()
    dataset = ChatDataset(x_train, y_train)
    trainer = Trainer(dataset, len(x_train[0]), len(parser.tags), c.HIDDEN_SIZE, c.BATCH_SIZE, c.LEARNING_RATE)

    trained_data = trainer.train()

    # save data
    trained_data["all_words"] = parser.all_words
    trained_data["tags"] = parser.tags
    if not os.path.exists(c.OUT_PATH):
        os.mkdir(c.OUT_PATH)
    torch.save(trained_data, os.path.join(c.OUT_PATH, c.TRAINED_DATA_FILENAME))
    print(f'training complete. File saved to {os.path.join(c.OUT_PATH, c.TRAINED_DATA_FILENAME)}')

    plt.style.use('seaborn-pastel')
    plt.rc('font', size=14)  # controls default text sizes
    plt.rc('axes', titlesize=14)  # fontsize of the axes title
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
    plt.rc('legend', fontsize=14)  # legend fontsize
    plt.rc('figure', titlesize=14)  # fontsize of the figure title
    plt.figure(figsize=(12, 9), dpi=40)
    plt.plot(list(range(1, len(trainer.loss) + 1)), trainer.loss, label="Loss", linewidth=2)
    plt.legend(loc='lower right')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    main()
