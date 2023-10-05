'''
n_layer_neural_network.py
ELEC 576 HW 1
Robert Heeter
4 October 2023
'''

from three_layer_neural_network import generate_data_make_moons, plot_decision_boundary, NeuralNetwork

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

def generate_data_make_blobs():
    '''
    param:
    return:
    - X: input data
    - y: given labels
    '''

    np.random.seed(0)
    X, y = datasets.make_blobs(n_samples=300, n_features=2, centers=5)

    return X, y

class DeepNeuralNetwork(NeuralNetwork):
    '''
    builds and trains an n_layer deep neural network
    '''
    def __init__(self, nn_layer_dims, act_fun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        param:
        - nn_layer_dims: the number of units in each layer (input + hidden + output)
        - act_fun_type: 'tanh', 'sigmoid', 'relu'
        - reg_lambda: regularization coefficient
        - seed: random seed
        return:
        '''

        self.nn_layer_dims = nn_layer_dims
        self.nn_num_layers = len(nn_layer_dims)
        self.act_fun_type = act_fun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = []
        self.b = []

        np.random.seed(seed)
        for i in range(self.nn_num_layers-1):
            self.W.append(np.random.randn(self.nn_layer_dims[i], self.nn_layer_dims[i+1]) / np.sqrt(self.nn_layer_dims[i]))
            self.b.append(np.zeros((1, self.nn_layer_dims[i+1])))

    def feedforward(self, X):
        '''
        param:
        - X: input data
        return:
        '''

        self.z = []
        self.a = []

        for i in range(self.nn_num_layers-1):
            if i == 0:
                self.z.append(np.dot(X, self.W[i]) + self.b[i])
            elif i > 0:
                self.z.append(np.dot(self.a[i-1], self.W[i]) + self.b[i])
            
            if i < self.nn_num_layers-2:
                self.a.append(self.act_fun(self.z[i], type=self.act_fun_type))

        exp_scores = np.exp(self.z[-1])
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def calculate_loss(self, X, y):
        '''
        param:
        - X: input data
        - y: given labels
        return:
        - loss for prediction
        '''

        num_examples = len(X)
        self.feedforward(X)

        y_one_hot_encoded = OneHotEncoder(sparse_output=False)
        y_one_hot_encoded = y_one_hot_encoded.fit_transform(y.reshape((-1, 1)))
        data_loss = - np.sum(np.log(self.probs) * y_one_hot_encoded) / num_examples

        # add regulatization term to loss (optional)
        for i in range(self.nn_num_layers-1):
            data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W[i])))
            
        return data_loss

    def backpropagation(self, X, y):
        '''
        param:
        - X: input data
        - y: given labels
        return:
        - dW: set of dL/dWi
        - db: set of dL/dbi
        '''

        num_examples = len(X)
        delta = self.probs
        delta[range(num_examples), y] -= 1

        self.dW = []
        self.db = []

        for i in range(self.nn_num_layers-1):
            j = self.nn_num_layers - 2 - i
            if j > 0: # hidden layers
                self.dW.insert(0, np.dot(self.a[j-1].T, delta))
                self.db.insert(0, np.sum(delta, axis=0))
                delta = np.dot(delta, self.W[j].T) * (self.diff_act_fun(self.z[j-1], type=self.act_fun_type))
            elif j == 0: # output layer (final layer)
                self.dW.insert(0, np.dot(X.T, delta))
                self.db.insert(0, np.sum(delta, axis=0))

        return self.dW, self.db

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        param:
        - X: input data
        - y: given labels
        - epsilon: learning rate
        - num_passes: the number of times that the algorithm runs through the whole dataset
        - print_loss: print the loss or not
        return:
        '''

        # gradient descent
        for i in range(0, num_passes):

            # feedforward (forward propoagation)
            self.feedforward(X)

            # backpropagation
            dW, db = self.backpropagation(X, y)

            # add regularization terms (b1 and b2 don't have regularization terms)
            for i in range(self.nn_num_layers-1):
                dW[i] += self.reg_lambda * self.W[i]

            # gradient descent parameter update
            for i in range(self.nn_num_layers-1):
                self.W[i] += -epsilon * dW[i]
                self.b[i] += -epsilon * db[i]

            # optionally print the loss (expensive because it uses the whole dataset)
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
    
    def visualize_decision_boundary(self, X, y):
        '''
        param:
        - X: input data
        - y: given labels
        return:
        '''

        plot_decision_boundary(lambda x: self.predict(x), X, y, self.act_fun_type, self.nn_layer_dims)

def main():
    # # generate and visualize Make Moons or Make Blobs datasets
    X, y = generate_data_make_moons()
    # X, y = generate_data_make_blobs()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.title("'Make Moons' dataset")
    # plt.show()

    # testing different network depths
    # model = DeepNeuralNetwork(nn_layer_dims=[2,3,5], act_fun_type='tanh')
    # model = DeepNeuralNetwork(nn_layer_dims=[2,3,3,5], act_fun_type='tanh')
    # model = DeepNeuralNetwork(nn_layer_dims=[2,3,3,3,5], act_fun_type='tanh')
    # model = DeepNeuralNetwork(nn_layer_dims=[2,3,3,3,3,5], act_fun_type='tanh')
    # model = DeepNeuralNetwork(nn_layer_dims=[2,3,3,3,3,3,5], act_fun_type='tanh')

    # testing different nn_layer_dims for deep network
    # model = DeepNeuralNetwork(nn_layer_dims=[2,3,3,3,3,5], act_fun_type='tanh')
    # model = DeepNeuralNetwork(nn_layer_dims=[2,5,5,5,5,5], act_fun_type='tanh')
    # model = DeepNeuralNetwork(nn_layer_dims=[2,10,10,10,10,5], act_fun_type='tanh')
    # model = DeepNeuralNetwork(nn_layer_dims=[2,20,20,20,20,5], act_fun_type='tanh')

    # testing different act_fun_types for deep network
    # model = DeepNeuralNetwork(nn_layer_dims=[2,3,3,3,3,5], act_fun_type='tanh')
    # model = DeepNeuralNetwork(nn_layer_dims=[2,3,3,3,3,5], act_fun_type='sigmoid')
    # model = DeepNeuralNetwork(nn_layer_dims=[2,3,3,3,3,5], act_fun_type='relu')

    # fit model
    # model.fit_model(X, y)
    # model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()
