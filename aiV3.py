import random as r
import numpy as np
import getDataNP as getData


class myAI:
    def __init__(self, structure, alpha) -> None:
        self.layers = []
        self.alpha = alpha
        self.create_network(structure)

    def predict(self, input_data):
        pass

    def train(self, input_data, correct_output):
        pass

    def test_accuracy(self, test_data, answers):
        pass
  
    def create_network(self, nodes_per_layer):
        for i, node_num in enumerate(nodes_per_layer):
            if i == 0:
                input_layer = layer(node_num, None)
                self.layers.append(input_layer)
            else:
                prev_num = nodes_per_layer[i - 1]
                self.layers.append(layer(prev_num, node_num))

    
            

class layer:
    def __init__(self, num_nodes, num_inputs) -> None:
        self.num_nodes = num_nodes
        self.values = np.zeros((num_nodes, 1))
        self.weights = np.zeros((num_inputs, num_nodes))
        self.biases = np.zeros((num_nodes, 1))

    


number_of_input_node = 784 
output_nodes = 10
struture = [number_of_input_node, 50, 30, output_nodes]
learning_rate = 0.01

# training on all the data 
digit_images_file = "files/train-images-idx3-ubyte.gz"
digit_labels_file = "files/train-labels-idx1-ubyte.gz"
lables, images = getData.get_data(300, digit_images_file, digit_images_file)

my_ai = myAI(struture, learning_rate)
my_ai.train(images, lables)


# testing the accuracy
digit_images_file = "files/t10k-images-idx3-ubyte"
digit_labels_file = "files/t10k-labels-idx1-ubyte"
test_lables, test_images = getData.get_data(100, digit_images_file, digit_labels_file)
my_ai.test_accuracy(test_images, test_lables)
