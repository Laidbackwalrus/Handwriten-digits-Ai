import random as r
import numpy as np
import getDataNP as getData

number_of_input_node = 784 
struture = [50, 30, 10] # last layer is the class layer (has to be 10)
learning_rate = 0.01


# training on all the data 
digit_images_file = "files/train-images-idx3-ubyte.gz"
digit_labels_file = "files/train-labels-idx1-ubyte.gz"
im = getData.get_data(300, digit_images_file, digit_images_file)


class myAI:
    lables, inputs = getData.get_data(300)
    def __init__(self, structure) -> None:
        pass
  
    def initialize_network(self, structure):
        self.num_nodes = 
        self.layer_counts = []
        self.list_of_connections = []
        self.list_of_biases = []
        for num_nodes in structure:
        self.create_random_layer(num_nodes)

    def create_random_layer(self, num_new_nodes):


