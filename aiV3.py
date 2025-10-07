import random as r
import numpy as np
import getDataNP as getData


class myAI:
    def __init__(self, structure) -> None:
        self.layer_counts = []
        self.list_of_connections = []
        self.list_of_biases = []
        for num_nodes in structure:
            self.create_random_layer(num_nodes)

    def predict(self, input_data):
        pass

    def train(self, input_data, correct_output):
        pass

    def test_accuracy(self, test_data, answers):
        pass
  
    def create_random_layer(self, num_new_nodes):
        pass


number_of_input_node = 784 
output_nodes = 10
struture = [number_of_input_node, 50, 30, output_nodes]
learning_rate = 0.01


# training on all the data 
digit_images_file = "files/train-images-idx3-ubyte.gz"
digit_labels_file = "files/train-labels-idx1-ubyte.gz"
lables, images = getData.get_data(300, digit_images_file, digit_images_file)


my_ai = myAI(struture)
my_ai.train(images, lables)


# testing the accuracy
digit_images_file = "files/t10k-images-idx3-ubyte"
digit_labels_file = "files/t10k-labels-idx1-ubyte"
test_lables, test_images = getData.get_data(100, digit_images_file, digit_labels_file)
my_ai.test_accuracy(test_images, test_lables)
