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
	connections = []
	for i in range(0,num_new_nodes):
		temp_list = []
		for j in range(0, num_nodes[layers]):
			temp_list.append(r.uniform(-1,1))
		temp_list = np.array(temp_list)
		connections.append(temp_list)

	biases = []
	for i in range(0, num_new_nodes):
		biases.append([r.uniform(-1,1)])

	num_nodes.append(num_new_nodes)
	list_of_connections.append(connections)
	list_of_biases.append(biases)

	layers += 1

def calculate_weight(z):
	#caluclates all the activations of the image passed
	z = [z]
	activations = []
	for i in range(layers):
		z.append(np.add(np.matmul(list_of_connections[i],z[-1]),list_of_biases[i]))
		for sublist in z[1:]:#skipping input layer, MIGHT CAUSE PROBLEMS!
			for item in sublist:  
				activations = sigmoid(item)
	return activations, z

def expected_outcome(labels):
	out = [0] * 10
	out[labels] = 1
	return out

def mini_batch(inputs, labels):
	x = len(labels)
	if (x/10) < 100:
		y = x // 10
	else:
		y = 100
	mini_i = []
	mini_l = []
	for i in range(y):
		z = r.randrange(0,x-i)
		#wait hold the phone am i poping the local or global variable
		#i hope its not the global one cus that would be crazy stupid
		mini_i.append(inputs.pop(z))
		mini_l.append(labels.pop(z))
	return mini_i, mini_l

def calculate_error(doh, z):
	doh = [[item]for item in doh]
	errors = []
	e = doh*sigmoid_prime(z[-1])
	#when doing hadamard product it makes every individual element be in its own array
	#for next version there may be a way to remove this problem by doing the whole thing differently idk
	#for now i will just make my own one that only works with 1 column matricies
	"""
	e = []
	for i in range(len(doh)):
		e.append([doh[i][0]*sigmoid_prime(z[-1][i][0])])
	"""
	errors.append(e)
	for i in reversed(range(0, layers)):
		wt = np.matrix.transpose(np.array(list_of_connections[i]))
		e = np.matmul(wt,e)
		z = [[sigmoid_prime(item)]for item in z[i-1]]
		e = np.multiply(e,z)
		errors.append(e)
	errors.reverse()
	return errors

#need a function that
#takes a image and its label
#calculates the activations for every layer
#calculates the error and changes the weights and biases
#calculates the cost to plot on a graph
#stores all of the weights and biases for every image in the mini batch
def network_1():
	#hidden layer 
	create_random_layer(10)
	create_random_layer(8)

	#output layer
	create_random_layer(10)

	for i in range(0,10):#how many mini batches are done
		mini_i, mini_l =  mini_batch(inputs, labels)
		#update mini batch
		errors = []
		cost = 0
		for j, image in enumerate(mini_i):#for every image in mini batch
			image = [[item / 255.0] for sublist in image for item in sublist]#flattens image
			a, z = calculate_weight(image)#a(L) ... a(L-n)
			y = expected_outcome(mini_l[j])
			e = np.subtract(a[-1], y)
			cost += np.sum(np.square(abs(e)))
			errors = calculate_error(e, z)

			temperrors = []
			for j in range(len(errors)-1):
				temperrorsj = []
				for k in range(len(errors[j])):
					temp = []
					for l in range(len(errors[j][k])):
						temp.append(errors[j][k][l][0])
					temperrorsj.append(np.array(temp))
				temperrors.append(temperrorsj)

			for j in range(len(errors)-1):
				list_of_connections[j] = np.subtract(list_of_connections[j], temperrors[j])

		
network_1()

#save()
#load()

#getData.display_images(inputs)