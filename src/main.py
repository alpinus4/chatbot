from data_parser import DataParser
from chat_dataset import ChatDataset

def main():
    parser = DataParser()
    parser.load('data/intents.json')

    x_train, y_train = parser.generate_training_data()
    dataset = ChatDataset(x_train, y_train)


if __name__ == "__main__":
    main()
