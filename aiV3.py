import random as r
import numpy as np
import getDataNP as getData

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class layer:
    def __init__(self, num_nodes, num_inputs) -> None:
        self.num_nodes = num_nodes
        self.values = np.zeros((num_nodes, 1))
        self.biases = np.random.uniform(-0.5, 0.5, (num_nodes, 1))

        if num_inputs is not None: # input layer has no inputs
            self.weights = np.random.uniform(-0.5, 0.5, (num_nodes, num_inputs))

class myAI:
    def __init__(self, structure, alpha) -> None:
        self.layers = []
        self.alpha = alpha
        self.create_network(structure)

    def predict(self, input_data):
        self.layers[0].values = input_data.reshape((784, 1))
        for i in range(1, len(self.layers)):
            prev_layer = self.layers[i - 1]
            curr_layer = self.layers[i]
            # print(curr_layer.weights.shape, prev_layer.values.shape, curr_layer.biases.shape)
            z = np.dot(curr_layer.weights, prev_layer.values) + curr_layer.biases
            curr_layer.values = sigmoid(z)
    
        return self.layers[-1].values

    def train(self, input_data, correct_output):
        pass

    def cost_function(self, predicted, actual):
        """Cost function using Mean Squared Error"""
        v = np.zeros((10, 1), dtype=np.float32)
        v[actual, 0] = 1.0


        return np.sum((predicted - v) ** 2) / 2

    def test_accuracy(self, prediction, actual):
        pass
  
    def create_network(self, nodes_per_layer):
        for i, node_num in enumerate(nodes_per_layer):
            if i == 0:
                input_layer = layer(node_num, None)
                self.layers.append(input_layer)
            else:
                prev_num = nodes_per_layer[i - 1]
                self.layers.append(layer(node_num, prev_num))

# parameters for the Ai
number_of_input_node = 784 
output_nodes = 10
struture = [number_of_input_node, 50, 30, output_nodes]
learning_rate = 0.01

# training on all the data 
digit_images_file = "files/train-images.idx3-ubyte.gz"
digit_labels_file = "files/train-labels.idx1-ubyte.gz"
lables, images = getData.get_data(300, digit_labels_file, digit_images_file)

my_ai = myAI(struture, learning_rate)
my_ai.train(images, lables)


# testing the accuracy
digit_images_file = "files/t10k-images.idx3-ubyte.gz"
digit_labels_file = "files/t10k-labels.idx1-ubyte.gz"
test_lables, test_images = getData.get_data(10, digit_labels_file, digit_images_file)

for (image, label) in zip(test_images, test_lables):
    prediction = my_ai.predict(image)
    cost = my_ai.cost_function(prediction, label)
    # print(f"Predicted: {prediction}, Actual: {label}")
    print(f"Cost: {cost}")