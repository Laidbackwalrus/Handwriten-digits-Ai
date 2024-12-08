from netrc import NetrcParseError
import random as r
import numpy as np
import getData

num_nodes = [784] #what?

layers = 0
list_of_connections = []
list_of_biases = []

labels,inputs = getData.get_data(300)#data set size

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def create_random_layer(num_new_nodes):
	global layers