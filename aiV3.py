import random as r
import numpy as np
import getDataNP as getData
import time  # Add this import

def sigmoid(z):
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def batch_label_to_vector(labels):
    """
    Convert a batch of labels to one-hot encoded vectors.
    Expects labels in range 0-9.
    Output shape is (N, 10, 1) where N is the number of labels.
    """
    batch_size = labels.shape[0]
    one_hot = np.zeros((batch_size, 10), dtype=np.float32)
    for i in range(batch_size):
        one_hot[i, labels[i]] = 1.0
    return one_hot

def label_to_vector(label):
    """
    Convert a single label to a one-hot encoded vector.
    Expects label in range 0-9.
    Output shape is (10, 1).
    """
    vector = np.zeros((10,), dtype=np.float32)
    vector[label] = 1.0
    return vector

class layer:
    def __init__(self, num_nodes, num_inputs) -> None:
        self.num_nodes = num_nodes
        self.values = np.zeros(num_nodes)
        self.values = np.zeros(num_nodes)

        if num_inputs is not None: # input layer has no weights or biases
            self.biases = np.random.uniform(-0.5, 0.5, num_nodes)
        if num_inputs is not None: # input layer has no weights or biases
            self.biases = np.random.uniform(-0.5, 0.5, num_nodes)
            self.weights = np.random.uniform(-0.5, 0.5, (num_nodes, num_inputs))

class myAI:
    def __init__(self, structure, alpha) -> None:
        self.layers = []
        self.alpha = alpha 
        self.alpha = alpha 
        self.create_network(structure)

    def predict(self, input_data):
        self.layers[0].values = input_data.reshape((784,))
        self.layers[0].values = input_data.reshape((784,))
        for i in range(1, len(self.layers)):
            prev_layer = self.layers[i - 1]
            curr_layer = self.layers[i]
            # print(curr_layer.weights.shape, prev_layer.values.shape, curr_layer.biases.shape)
            z = np.dot(curr_layer.weights, prev_layer.values) + curr_layer.biases
            curr_layer.values = sigmoid(z)
    
        return self.layers[-1].values

    def train(self, image, label):
        prediction = self.predict(image)
        deltas = self.calculate_deltas(prediction, label)
        gradients_w, gradients_b = self.calculate_gradients(deltas)
        self.gradient_descent(gradients_w, gradients_b)

    def calculate_deltas(self, prediction, label):
        # Output layer delta
        deltas = np.array([None] * (len(self.layers) - 1))
        output_layer = self.layers[-1]
        deltas[-1] = (prediction - label) * sigmoid_prime(output_layer.values) 

        # Hidden layers delta
        for i in range(len(self.layers) - 2, 0, -1):
            # print(i)
            # print("shape1", np.transpose(self.layers[i+1].weights.shape))
            # print("shape2", (np.transpose(self.layers[i+1].weights) @ deltas[i]).shape)
            deltas[i-1] = (np.transpose(self.layers[i+1].weights) @ deltas[i]) * sigmoid_prime(self.layers[i].values)

        return deltas

    def calculate_gradients(self, deltas):
        gradients_w = [None] * (len(self.layers) - 1)
        gradients_b = [None] * (len(self.layers) - 1)

        for i in range(0, len(self.layers) - 1):
            layer = self.layers[i]
            gradients_w[i] = np.outer(deltas[i], layer.values)
            gradients_b[i] = deltas[i]

        return gradients_w, gradients_b
    
    def gradient_descent(self, gradients_w, gradients_b):
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            layer.weights -= self.alpha * gradients_w[i-1]
            layer.biases -= self.alpha * gradients_b[i-1]

    def cost_function(self, predicted, actual):
        """Cost function using Mean Squared Error"""
        return np.sum((predicted - actual) ** 2) / 2

    def test_accuracy(self, images, labels):
        correct = 0
        for image, actual in zip(images, labels):
            prediction = self.predict(image)
            guess = np.argmax(prediction)
            if guess == actual:
                correct += 1

        return correct / len(images)

    def all_layer_values(self):
        array = np.zeros((len(self.layers), 1))
        for i, layer in enumerate(self.layers):
            array[i] = layer.values
        return array

    def create_network(self, nodes_per_layer):
        for i, node_num in enumerate(nodes_per_layer):
            if i == 0:
                input_layer = layer(node_num, None)
                self.layers.append(input_layer)
            else:
                prev_num = nodes_per_layer[i - 1]
                self.layers.append(layer(node_num, prev_num))

    def weight_summary(self):
        for i, layer in enumerate(self.layers[1:], start=1):
            print(f"Layer {i}: Weights shape: {layer.weights.shape}, Biases shape: {layer.biases.shape}")

if __name__ == "__main__":
    start_time = time.time()  # Start timer
    
    # parameters for the Ai
    number_of_input_node = 784 
    output_nodes = 10
    struture = [number_of_input_node, 200, 100, output_nodes]
    learning_rate = 0.01
    training_size, test_size = 10000, 5000
    epochs = 25

    # training on all the data 
    digit_images_file = "files/train-images.idx3-ubyte.gz"
    digit_labels_file = "files/train-labels.idx1-ubyte.gz"
    lables, images = getData.get_data(training_size, digit_labels_file, digit_images_file)

    vectorised_lables = batch_label_to_vector(lables)
    my_ai = myAI(struture, learning_rate)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for (image, v_label) in zip(images, vectorised_lables):
            my_ai.train(image, v_label)               

    # testing the accuracy
    digit_images_file = "files/t10k-images.idx3-ubyte.gz"
    digit_labels_file = "files/t10k-labels.idx1-ubyte.gz"
    test_labels, test_images = getData.get_data(test_size, digit_labels_file, digit_images_file)
    vectorised_labels = batch_label_to_vector(test_labels)

    test_accuracy = my_ai.test_accuracy(test_images, test_labels)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    end_time = time.time()  # End timer
    total_time = end_time - start_time
    print(f"Total Time: {total_time:.2f} seconds")
    
    # Convert to minutes if longer than 60 seconds
    if total_time > 60:
        minutes = int(total_time // 60)
        seconds = total_time % 60
        print(f"Total Time: {minutes}m {seconds:.2f}s")

# best result so far using:
# 10000 training size, 5000 test size, 25 epochs, 0.01 learning rate, 200 and 100 hidden nodes
## Test Accuracy: 87.10%
## Total Time: 276.19 seconds
## Total Time: 4m 36.19s