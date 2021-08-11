import sys, os
sys.path.append(os.pardir)
import numpy as np
from Function.mnist import load_mnist

def step_function(sfx):
    return np.array(sfx > 0, dtype=np.int)

def sigmoid(sx):
    return 1/(1+np.exp(-sx))

def reLU(rLUx):
    return np.maximum(0, rLUx)

def softmax(smx):
    c = np.max(smx)
    
    exp_a = np.exp(smx-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def get_data():
    (x_train, t_train), (x_test, t_test) =\
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test
