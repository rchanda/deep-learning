import os
import numpy as np

def load_data(filePath):
	X = []
	Y = []
	
	file = open(filePath, 'r')

	for line in file:
		line = [float(s) for s in line.split(',')]
		X.append(line[:-1])
		Y.append(int(line[-1]))

	X = np.array(X)
	Y = np.array(Y)

	file.close()

	return X, Y

def sigmoid(x):
	return (1.0 / (1.0 + np.exp(x)))

def sigmoid_deriv(x):
	y = sigmoid(x)
	return np.multiply(y, (1-y))

def tanh(x):
	return 2*sigmoid(2*x)-1

def tanh_deriv(x):
	return 4*sigmoid_deriv(2*x)

def relu(x):
	return np.maximum(0, x)

def relu_deriv(x):
	ind = x>0
	x[ind]=1
	ind = x==0
	x[ind]=0
	return x

def softmax(x):
	xshift = x-np.max(x)
	return np.exp(xshift) / np.sum(np.exp(xshift), axis=1, keepdims=True)


def onehot(y):
	T = []

	for t in y:
		row = [0]*10
		row[t] = 1
		T.append(row)

	T = np.array(T)
	return T


def gradient_descent(weights, dW, bias, dBias, n, lr):
	k = len(weights)

	for i in range(k):
		weights[i] = weights[i] - (lr/n)*dW[i]
		bias[i] = bias[i] - (lr/n)*dBias[i]
	return weights, bias


def gradient_descent_momentum(weights, dW, vWeights, bias, dBias, vBias, n, lr, momentum):
	k = len(weights)

	for i in range(k):
		vWeights[i] = momentum*vWeights[i] - (lr/n)*dW[i]
		vBias[i] = momentum*vBias[i] - (lr/n)*dBias[i]

		weights[i] = weights[i] + vWeights[i]
		bias[i] = bias[i] + vBias[i]

	return weights, vWeights, bias, vBias


def forward_prop(X, weights, bias, func):
	k = len(weights)
	Y = [X]
	Z = []

	for i in range(k):
		Z.append(np.dot(Y[i], weights[i]) + bias[i])
		if i is k-1:
			Y.append(softmax(Z[i]))
		else:
			Y.append(func(Z[i]))
	return Y, Z


def backward_prop(Y, Z, weights, T, func_deriv):
	k = len(weights)
	dZ = [ [] for i in range(k) ]
	dZ[k-1] = Y[k] - T

	dW = [ [] for i in range(k) ]
	dBias = [ [] for i in range(k) ]
	dY = [ [] for i in range(k) ]

	for i in reversed(range(k)):
		if i is not k-1:
			dF = func_deriv(Y[i+1])
			dZ[i] = np.multiply(dF, dY[i+1])

		dW[i] = np.dot(Y[i].T, dZ[i])
		dBias[i] = np.sum(dZ[i], axis=0, keepdims=True)
		dY[i] = np.dot(dZ[i], weights[i].T)

	return dW, dBias


def update_weights_double_layer_act(X, Y, weights, bias, lr, activation):
	func = {"sigmoid" : sigmoid, "tanh" : tanh, "relu" : relu}
	func_deriv = {"sigmoid" : sigmoid_deriv, "tanh" : tanh_deriv, "relu" : relu_deriv}

	T = onehot(Y)
	Y, Z = forward_prop(X, weights, bias, func[activation])
	dW, dBias = backward_prop(Y, Z, weights, T, func_deriv[activation])
	updated_weights, updated_bias = gradient_descent(weights, dW, bias, dBias, X.shape[0], lr)
	return updated_weights, updated_bias


def update_weights_double_layer(X, Y, weights, bias, lr):
	return update_weights_double_layer_act(X, Y, weights, bias, lr, 'sigmoid')

def update_weights_single_layer(X, Y, weights, bias, lr):
	return update_weights_double_layer_act(X, Y, weights, bias, lr, 'sigmoid')

def update_weights_perceptron(X, Y, weights, bias, lr):
	return update_weights_double_layer_act(X, Y, weights, bias, lr, 'sigmoid')


def update_weights_double_layer_act_mom(X, Y, weights, bias, lr, activation, momentum, epochs):
	func = {"sigmoid" : sigmoid, "tanh" : tanh, "relu" : relu}
	func_deriv = {"sigmoid" : sigmoid_deriv, "tanh" : tanh_deriv, "relu" : relu_deriv}

	T = onehot(Y)

	vWeights = [np.zeros_like(w) for w in weights]
	vBias = [np.zeros_like(b) for b in bias]

	for i in range(epochs):
		Y, Z = forward_prop(X, weights, bias, func[activation])
		dW, dBias = backward_prop(Y, Z, weights, T, func_deriv[activation])
		weights, vWeights, bias, vBias = gradient_descent_momentum(weights, dW, vWeights, bias, dBias, vBias, X.shape[0], lr, momentum)

	return weights, bias


