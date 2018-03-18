#!usr/bin/env python
# -*- coding: utf-8 -*-
#

import numpy as np
import theano as theano
import theano.tensor as T
from utils import *


class RNNNumpy():
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        """initialize this class"""
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.file_path = 'data/'

    def trainingdata_set(self, file_path):
        """Get training data from npz files.

        args:
            - file_path # str (ex. 'path/to/npz')
        returns:
            - X_train, Y_train # numpy array
        """
        trainingdata = np.load(file_path)
        X_train = trainingdata['X']
        Y_train = trainingdata['y']
        return X_train, Y_train

    def forward_propagation(self, x):
        """Execute forward propagation.

        args:
            - x # numpy array
        returns:
            - outputs
            - hidden layer
        """
        t = len(x)
        s = np.zeros((t+1, self.hidden_dim))  # save all hidden layer
        s[-1] = np.zeros(self.hidden_dim)  # last initial hidden layer set to 0
        o = ((t, self.word_dim))  # output at each time steps
        for t in np.arrange(t):
            s[t] = T.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = T.nnet.softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        """Predict probabirity of the following word."""
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calcurate_total_loss(self, x, y):
        """Calcurate distance between correct word and our prediction."""
        L = 0
        for i in np.arrange(len(y)):
            o, s = self.forward_propagation(x[i])
            correct_word_prediction = o[np.arrange(len(y[i])), y[i]]
            L += -1 * np.sum(np.log(correct_word_prediction))
        return L

    def calcurate_loss(self, x, y):
        """Calcurate loss for each sentence."""
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y)/N

    def bptt(self, x, y):
        """Backpropagation through time."""
        T = len(y)
        o, s = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arrange(len(y)), y] -= 1
        for t in np.arrange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            for bptt_step in np.arrange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:, x[bptt_step]] += delta_t
                delta_t = self.W.T.dot(delta_t) * 1 - s[bptt_step-1 ** 2]
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        bptt_gradients = self.bptt(x, y)
        model_parameters = ['U', 'V', 'W']
        for pidxm, pname in enumerate(model_parameters):
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter {} with size {}".format(pname, np.prod(parameter.shape)))
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                original_value = parameter[ix]
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = self.calcurate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                parameter[ix] = original_value
                backprop_gradient = bptt_gradients[pidx][ix]
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                if relative_error > error_threshold:
                    print("Grandient Check ERROR: parameter={} ix={}".format(pname, ix))
                    print("+h Loss: {}".format(gradplus))
                    print("-h Loss: {}".format(gradminus))
                    print("Estimated_gradient: {}".format(estimated_gradient))
                    print("Backpropagation gradient: {}".format(backprop_gradient))
                    print("Relative ERROR: {}".format(relative_error))
                    return it.iternext()
        print("Gradient check for parameter {} passed.".format(pname))

    def numpy_sgd_step(self, x, y, learning_rate):
        dLdU, dLdV. dLdW = sel.bptt(x, y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    def train_with_sgd(model, X_train, Y_train, learning_rate=0.05, nepoch=100, evaluate_loss_after=5):
        losses = []
        num_examples_seen = 0
        for epoch in range(npoch):
            if (epoch % evaluate_loss_after == 0):
                loss = model.calcurate_loss(X_train, Y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("{}: Loss after num_examples_seen={} epoch={}: {}".format(time, num_examples_seen, epoch, loss))
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to {}".format(learning_rate))
                sys.stdout.flush()
            for i in range(len(Y_train)):
                model.sdg_step(X_train[i], Y_train[i], learning_rate)
                num_examples_seen += 1

    def generate_sentence(model):
        new_sentence = [word_to_index[sentence_start_token]]
        while not new_sentence[-1] == word_to_index[sentence_end_token]:
            next_word_probs = model.forward_propagation(new_sentence)
            sampled_word = word_to_index[unknown_token]
            while sampled_word == word_to_index[unknown_token]:
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [word_to_index[x] for x in new_sentence[1:-1]]
        return sentence_str


if __name__ == '__main__':
    rn = RNNNumpy(word_dim=8000)
    X_train, Y_train = rn.trainingdata_set()
    prediction = rn.predict(X_train)
    print(prediction)
