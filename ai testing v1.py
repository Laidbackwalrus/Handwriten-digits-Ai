import random as r
import numpy.matlib
import numpy as np
import math as m

input_nodes = []
layers = [input_nodes]
num_nodes = [9]
layer_count = 0

circle = [
    0,1,0,
    1,0,1,
    0,1,0]
box = [
    1,1,1,
    1,0,1,
    1,1,1]
dot = [
    0,0,0,
    0,1,0,
    0,0,0]
star = [
    0,1,0,
    1,1,1,
    0,1,0]

def sigmoid(x):
    return 1 / (1 + m.exp(-x))

def get_weights(n):
    weights = []
    for i in range(num_nodes[n]):
        weights.append(layers[n][i].weight)
    return weights

def get_connections(n):
    connections = []
    for i in range(0,num_nodes[n]):
        connections.append(layers[n][i].connections)
    return connections

def get_bias(n):
    bias = []
    for i in range(num_nodes[n]):
        bias.append(layers[n][i].bias)
    return bias

def calculate_cost(outcome,expected_outcome):
    cost = np.sqrt(sum(np.square(np.subtract(outcome, expected_outcome))))
    return cost


def input_nodes():
    random_inputs = r.choice([circle, box, dot, star])

    for i in range(0,9):
        layers[0].append(Node(random_inputs[i],[],""))

    if random_inputs == circle:
        expected_outcome = [1,0,0,0]
    if random_inputs == box:
        expected_outcome = [0,1,0,0]
    if random_inputs == dot:
        expected_outcome = [0,0,1,0]
    if random_inputs == star:
        expected_outcome = [0,0,0,1]

    return expected_outcome


def new_layer(n):
    global layer_count
    layer_count += 1
    global num_nodes
    num_nodes.append(n)
    #create new layer in layers
    layers.append([])

    #adds nodes to object
    for i in range(0,n):
        #random connections
        connections = []
        for j in range(0,num_nodes[layer_count-1]):
          connections.append(r.uniform(0, 1))
        #random bias
        bias = r.uniform(0, 1)
        layers[layer_count].append(Node("",connections,bias))







def fire(n):
    cost = 0
    for i in range(0,n):
        expected_outcome = input_nodes()
        for j in range(1,layer_count+1):
            weights = get_weights(j-1) 
            connections = get_connections(j)
            bias = get_bias(j)
            raw_weights = np.subtract(np.matmul(connections, weights), bias)
            weights = []
            for row in raw_weights:
                weights.append(sigmoid(row))
            for k in range(0, num_nodes[j]):
                layers[j][k].weight = weights[k]  
        cost += calculate_cost(weights, expected_outcome)
        layers[0] = []   
    cost = cost/n
    print (cost)


class Node:
    def __init__(self, weight, connections, bias):
        self.weight = weight
        self.connections = connections
        self.bias = bias
   




new_layer(4)

fire(100)