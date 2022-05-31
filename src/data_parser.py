from sentence_processor import SentenceProcessor


class DataParser():
    def __init__(self):
        self.sp = SentenceProcessor()
        self.bag_of_words = []

    def load(self, path):
        # load json
        # process it using sentence processor
        # append to bag of words, so that we can call load multiple times, on multiple times, and build our dataset from multiple files
        pass

    def generate_training_data(self):
        # generates xtrain, ytrain to pass into dataset
        pass
