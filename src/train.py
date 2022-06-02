import torch

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
    torch.save(trained_data, c.TRAINED_DATA_FILEPATH)
    print(f'training complete. file saved to {c.TRAINED_DATA_FILEPATH}')

if __name__ == "__main__":
    main()