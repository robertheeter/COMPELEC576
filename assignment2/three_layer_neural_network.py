'''
three_layer_neural_network.py
ELEC 576 HW 1
Robert Heeter
4 October 2023
'''

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

def generate_data_make_moons():
    '''
    generate data_make_moons
    param:
    return:
    - X: input data
    - y: given labels
    '''

    np.random.seed(0)
    X, y = datasets.make_moons(n_samples=200, noise=0.20)

    return X, y

def plot_decision_boundary(pred_func, X, y, act_fun_type, nn_layer_dims):
    '''
    plot the decision boundary
    param:
    - pred_func: function used to predict the label
    - X: input data
    - y: given labels
    - act_fun_type: 'tanh', 'sigmoid', 'relu'
    - nn_layer_dims: the number of units in each layer (input + hidden + output)
    return:
    '''

    # set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title(f'Make Blobs: act_fun_type = [{act_fun_type}], nn_layer_dims = {nn_layer_dims}')
    plt.show()

class NeuralNetwork(object):
    '''
    builds and trains a neural network
    '''
    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, act_fun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        param:
        - nn_input_dim: input dimension
        - nn_hidden_dim: the number of hidden units
        - nn_output_dim: output dimension
        - act_fun_type: 'tanh', 'sigmoid', 'relu'
        - reg_lambda: regularization coefficient
        - seed: random seed
        return:
        '''

        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.act_fun_type = act_fun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def act_fun(self, z, type):
        '''
        computes the activation functions
        param:
        - z: net input
        - type: 'tanh', 'sigmoid', 'relu'
        return:
         - activations
        '''

        if type == 'tanh':
            return np.tanh(z)

        elif type == 'sigmoid':
            return 1/(1+np.exp(-z))

        elif type == 'relu':
            return np.maximum(z, 0)
        
        else:
            raise Exception(f'[{type}] is not a valid activation type')

    def diff_act_fun(self, z, type):
        '''
        computes the derivatives of the activation functions w.r.t. the net input
        param:
        - z: net input
        - type: tanh, sigmoid, or relu
        return:
        - derivatives of the activation functions w.r.t. the net input
        '''
        
        if type == 'tanh':
            f = np.tanh(z)
            return 1-np.power(f, 2)
            
        elif type == 'sigmoid':
            f = 1/(1+np.exp(-z))
            return f*(1-f)
        
        elif type == 'relu':
            f = z
            f[f > 0] = 1
            return f
        
        else:
            raise Exception(f'[{type}] is not a valid activation type')

    def feedforward(self, X):
        '''
        builds a 3-layer neural network and computes the two probabilities, one for class 0 and one for class 1
        param:
        - X: input data
        return:
        '''

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.act_fun(self.z1, type=self.act_fun_type)
        self.z2 = np.dot(self.a1, self.W2) + self.b2

        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def calculate_loss(self, X, y):
        '''
        computes the loss for prediction
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
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        # data_loss = (-1. / num_examples) * data_loss

        return data_loss

    def predict(self, X):
        '''
        infers the label of a given data point X
        param:
        - X: input data
        return:
        - label inferred
        '''

        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backpropagation(self, X, y):
        '''
        implements backpropagation to compute the gradients used to update the parameters in the backward step
        param:
        - X: input data
        - y: given labels
        return:
        - dW2: dL/dW2
        - db2: dL/db2
        - dW1: dL/dW1
        - db1: dL/db1
        '''

        num_examples = len(X)
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        delta2 = np.dot(delta3, self.W2.T) * (self.diff_act_fun(self.z1, type=self.act_fun_type))
        
        dW2 = np.dot(self.a1.T, delta3) # dW2 = dL/dW2
        db2 = np.sum(delta3, axis=0) # db2 = dL/db2
        dW1 = np.dot(X.T, delta2) # dW1 = dL/dW1
        db1 = np.sum(delta2, axis=0) # db1 = dL/db1

        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        uses backpropagation to train the network
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
            dW1, dW2, db1, db2 = self.backpropagation(X, y)

            # add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            # optionally print the loss (expensive because it uses the whole dataset)
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        plots the decision boundary created by the trained network
        param:
        - X input data
        - y: given labels
        return:
        '''

        plot_decision_boundary(lambda x: self.predict(x), X, y, self.act_fun_type, [self.nn_input_dim, self.nn_hidden_dim, self.nn_output_dim])

def main():
    # # generate and visualize Make Moons dataset
    X, y = generate_data_make_moons()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.title("'Make Moons' dataset")
    # plt.show()

    # testing different act_fun_type
    model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, act_fun_type='tanh')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3 , nn_output_dim=2, act_fun_type='sigmoid')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3 , nn_output_dim=2, act_fun_type='relu')

    # testing different nn_hidden_dim
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=1 , nn_output_dim=2, act_fun_type='tanh')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=2 , nn_output_dim=2, act_fun_type='tanh')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3 , nn_output_dim=2, act_fun_type='tanh')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=4 , nn_output_dim=2, act_fun_type='tanh')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=5 , nn_output_dim=2, act_fun_type='tanh')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=6 , nn_output_dim=2, act_fun_type='tanh')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=7 , nn_output_dim=2, act_fun_type='tanh')
    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=8 , nn_output_dim=2, act_fun_type='tanh')   

    # fit model
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()
