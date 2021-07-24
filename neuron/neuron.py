# -*- coding: utf-8 -*-

#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    #our activation function: f(x) = 1/(1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


# class Neuron:
#     def __init__(self, weights, bias):
#         self.weights = weights
#         self.bias = bias


    # def feedforward(self, inputs):
    #     #weight inputs, add bias, then use the activation function
    #     total = np.dot(self.weights, inputs) + self.bias
    #     return sigmoid(total)


class OurNeuralNetwork:
    def __init__(self):
        # 权重，Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # 截距项，Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

        # weights = np.array([0,1])
        # bias = 0
        #
        # #The Neuron class here is from the previous section
        # self.h1 = Neuron(weights, bias)
        # self.h2 = Neuron(weights, bias)
        # self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

        # out_h1 = self.h1.feedforward(x)
        # out_h2 = self.h2.feedforward(x)
        #
        # #The inputs for o1 are the outputs from h1 and h2
        # out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        #
        # return out_o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000  # number of times to loop through the entire dataset
        pic_x = []
        pic_y = []
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                pic_x.append(epoch)
                pic_y.append(loss)
                print("Epoch %d loss: %.3f" % (epoch, loss))
        plt.plot(pic_x, pic_y)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

if __name__=='__main__':
        # Define dataset
    data = np.array([
      [-2, -1],  # Alice
      [25, 6],   # Bob
      [17, 4],   # Charlie
      [-15, -6], # Diana
    ])
    all_y_trues = np.array([
      1, # Alice
      0, # Bob
      0, # Charlie
      1, # Diana
    ])
    
    # Train our neural network!
    network = OurNeuralNetwork()
    network.train(data, all_y_trues)
    
    # Make some predictions
    emily = np.array([-7, -3])
    frank = np.array([20, 2])
    print("Emily: %.3f" % network.feedforward(emily))
    print("Frank: %.3f" % network.feedforward(frank))
    
    # weights = np.array([0, 1])
    # bias = 4
    # n = Neuron(weights, bias)
    #
    # x = np.array([2, 3])
    # print (n.feedforward(x))
    
    
    # network = OurNeuralNetwork()
    # x = np.array([2,3])
    # print(network.feedforward(x))
