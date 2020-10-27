#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:32:17 2019

@author: Isabella
"""

# %matplotlib inline
import matplotlib.pyplot as plt
# %matplotlib notebook
# %pylab inline
import pandas as pd
import numpy as np
from unicodedata import *
#from copy import deepcopy
#import sys

# T-maze task
class TaskTmaze(object):
    def __init__(self, cor_length, rand_number, discount_factor=0., noise=None):
        self.cor_length = cor_length
        if noise:
            self.get_next_input = self.get_next_input_noisy
        else:
            self.get_next_input = self.get_next_input_noiseless
        self.position = 0
        self.t = 0

        self.total_reward = 0.
        self.pos_reward = 4.0
        #self.neg_reward = -0.2
        self.neg_reward = discount_factor

        if rand_number < 0.5:
            # encode N
            self.startsignal = np.array([1., 1., 0.])
        else:
            # encode S
            self.startsignal = np.array([0., 1., 1.])

    def next_step(self, action):
        if self.t == 0:
            reward = 0.
        else:
            if self.position == 0:
                if action == 1:
                    self.position = 1
                    reward = 0
                else:
                    reward = self.neg_reward
            elif 0 < self.position < self.cor_length:
                if action == 1:
                    self.position += 1
                    reward = 0
                elif action == 3:
                    self.position -= 1
                    reward = 0 #self.neg_reward
                else:
                    reward = self.neg_reward
            elif self.position == self.cor_length:
                if action == 1:
                    reward = self.neg_reward
                elif action == 3:
                    self.position -= 1
                    reward = 0#self.neg_reward
                else:
                    self.position = 'end'
                    if (action == 0 and self.startsignal[0] == 1) or (action == 2 and self.startsignal[2] == 1):
                        reward = self.pos_reward
                    else:
                        reward = self.neg_reward
            else:
                print ('task error')

        next_input = self.get_next_input()

        self.total_reward += reward
        self.t += 1
        return next_input, reward

    def get_next_input_noiseless(self):
        if self.t == 0:
            result = self.startsignal
        elif self.t > 1.2 * self.cor_length + 2:
            result = np.array([0.])
        elif self.position == 'end':
            result = np.array([0.])
        elif self.position < self.cor_length:
            result = np.array([1., 0., 1.])
        elif self.position == self.cor_length:
            result = np.array([0., 1., 0.])
        else:
            print('task error')
        return result
    
    def get_next_input_noisy(self):
        if self.t == 0:
            result = self.startsignal
        elif self.t > 1.2 * self.cor_length + 2:
            result = np.array([0.])
        elif self.position == 'end':
            result = np.array([0.])
        elif self.position < self.cor_length:
            result = np.array([np.random.rand(), 0., np.random.rand()])
        elif self.position == self.cor_length:
            result = np.array([0., 1., 0.])
        else:
            print('task error')
        return result

    def get_total_reward(self):
        return self.total_reward

    def get_t(self):
        return self.t - 1.

class sigmoid(object):
    def __init__(self):
        self.transform = np.vectorize(self.transform)
        self.derivative = np.vectorize(self.derivative)

    def transform(self, s):
        return 1. / (1. + np.exp(-s))

    def derivative(self, s):
        return (1. / (1. + np.exp(-s))) * (1. - 1. / (1. + np.exp(-s)))

class sigmoid_adj(object):
    def __init__(self):
        self.transform = np.vectorize(self.transform)
        self.derivative = np.vectorize(self.derivative)

    def transform(self, s):
        return 10. * 1. / (1. + np.exp(-s / 10.)) - 5.

    def derivative(self, s):
        return (1. / (1. + np.exp(-s / 10.))) * (1. - 1. / (1. + np.exp(-s / 10.)))

class tanh(object):
    def __init__(self):
        self.transform = np.vectorize(self.transform)
        self.derivative = np.vectorize(self.derivative)

    def transform(self, s):
        return (1. - np.exp(-2. * s)) / (1. + np.exp(-2. * s))

    def derivative(self, s):
        return 1. - ((1. - np.exp(-2. * s)) / (1. + np.exp(-2. * s))) ** 2

class identity(object):
    def __init__(self):
        self.transform = np.vectorize(self.transform)
        self.derivative = np.vectorize(self.derivative)

    def transform(self, s):
        return s

    def derivative(self, s):
        return 1.


class activation_unit(object):
    def __init__(self, activ_fun_type):
        self.a = 0.
        self.b = 0.

        if activ_fun_type == 'sigmoid':
            self.activ_fun = sigmoid()
        elif activ_fun_type == 'tanh':
            self.activ_fun = tanh()
        elif activ_fun_type == 'sigmoid_adj':
            self.activ_fun = sigmoid_adj()
        elif activ_fun_type == 'identity':
            self.activ_fun = identity()
        else:
            print ('unknown activation function\n')

    def transform(self, s):
        self.a = s
        output = self.activ_fun.transform(s)
        self.b = output
        return output

    def derivative(self, s):
        return self.activ_fun.derivative(s)

    def reset(self):
        self.a = 0.
        self.b = 0.


# Use this
class CEC(object):
    # track states in list (as in sigmoid and tanh)
    def __init__(self):
        self.state = 0.

    def transform(self, s, f_gate):
        self.state = f_gate * self.state + s #* 4.
        return self.state

    def reset(self):
        self.state = 0.

# No peepholes, f-gate, o-gate
# No biases, no self recurrency
class LSTM(object):
    def __init__(self, dim_x, w_init, alpha, gamma, lam):
        # components
        self.i_gate = activation_unit('sigmoid')
        self.i_cell = activation_unit('sigmoid')
        self.CEC = CEC()
        self.o_cell = activation_unit('identity')

        # weights
        self.w_init = w_init
        self.w_xc = np.random.uniform(low=-self.w_init, high=self.w_init, size=dim_x)
        self.w_xi = np.random.uniform(low=-self.w_init, high=self.w_init, size=dim_x)

        self.e_xc = np.zeros(dim_x)
        self.e_xi = np.zeros(dim_x)

        # others
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam

        # gradient vectors
        self.d_c = np.zeros(dim_x)
        self.d_i = np.zeros(dim_x)

        # input vectors
        self.x_inputs = 0.

    def forward_pass(self, x):
        self.x_inputs = x

        i_gate_val = self.i_gate.transform(np.dot(self.w_xi, x))
        i_cell_val = self.i_cell.transform(np.dot(self.w_xc, x))
        self.CEC.transform(i_gate_val * i_cell_val, 1.)
        cell_output = self.o_cell.transform(self.CEC.state)

        return cell_output

    def backward_pass(self, e_c_input, exploration):
        # Truncated RTRL (see Section 2.2: F.A. Gers, 2002)
        e_s = self.o_cell.derivative(self.o_cell.a) * e_c_input #* 4

        self.d_c = self.d_c + self.i_cell.derivative(self.i_cell.a) * self.i_gate.b * self.x_inputs
        self.d_i = self.d_i + self.i_cell.b * self.i_gate.derivative(self.i_gate.a) * self.x_inputs

        self.e_xc = self.gamma * self.lam * self.e_xc * (not exploration) + e_s * self.d_c
        self.e_xi = self.gamma * self.lam * self.e_xi * (not exploration) + e_s * self.d_i

    def update_weights(self, ETD):
        self.w_xc += self.alpha * ETD * self.e_xc
        self.w_xi += self.alpha * ETD * self.e_xi

    def reset(self):
        self.e_xc.fill(0.)
        self.e_xi.fill(0.)
        self.d_c = 0.
        self.d_i = 0.
        self.x_inputs = 0.

        self.i_gate.reset()
        self.i_cell.reset()
        self.CEC.reset()
        self.o_cell.reset()

class LSTM_test(object):
    def __init__(self, weights, i):
        # components
        self.i_gate = activation_unit('sigmoid')
        self.i_cell = activation_unit('sigmoid')
        self.CEC = CEC()
        self.o_cell = activation_unit('identity')

        self.w_xi = weights[0][i]
        self.w_xc = weights[1][i]
        # input vectors
        self.x_inputs = 0.
        
    def forward_pass(self, x):
        self.x_inputs = x

        i_gate_val = self.i_gate.transform(np.dot(self.w_xi, x))
        i_cell_val = self.i_cell.transform(np.dot(self.w_xc, x))
        self.CEC.transform(i_gate_val * i_cell_val, 1.)
        cell_output = self.o_cell.transform(self.CEC.state)

        return cell_output

    def reset(self):
        self.e_xc.fill(0.)
        self.e_xi.fill(0.)
        self.d_c = 0.
        self.d_i = 0.
        self.x_inputs = 0.

        self.i_gate.reset()
        self.i_cell.reset()
        self.CEC.reset()
        self.o_cell.reset()
        
# Uses 'sequential network construction' (see section 4.7: S. Hochreiter, 1997)
# Learns task in iter=18000 for p=4 with w_init=0.4 and alpha=0.5
# No biases, no self recurrency
# Network adjusted
# Things to do: remove all lists and deepcopy, just remember last input/output (including in the activation functions)
# Big clean up needed for Network and LSTM classes
class Network(object):
    def __init__(self, n_inputs, n_h_normal, n_h_mem, n_outputs, w_init, alpha, gamma, lam):
        self.w_init = w_init
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam

        self.n_inputs = n_inputs
        self.n_h_normal = n_h_normal
        self.n_h_mem = n_h_mem
        self.n_outputs = n_outputs

        self.input_layer = identity()
        self.h_normal_layer = tanh()
        self.h_mem_layer = [LSTM(n_inputs, self.w_init, alpha, gamma, lam) for i in range(n_h_mem)]
        self.output_layer = identity()

        self.w_hnormalo = np.random.uniform(low=-self.w_init, high=self.w_init, size=(n_h_normal, n_outputs))
        self.w_hmemo = np.random.uniform(low=-self.w_init, high=self.w_init, size=(n_h_mem, n_outputs))
        self.w_ihnormal = np.random.uniform(low=-self.w_init, high=self.w_init, size=(n_inputs, n_h_normal))
        self.b_o = np.random.uniform(low=-self.w_init, high=self.w_init, size=(1, n_outputs))

        self.e_hnormalo = np.zeros((n_h_normal, n_outputs))
        self.e_hmemo = np.zeros((n_h_mem, n_outputs))
        self.e_ihnormal = np.zeros((n_inputs, n_h_normal))
        self.e_bo = np.zeros((1, n_outputs))

        self.input_layer_outputs = 0.
        self.hnormal_layer_outputs = 0.
        self.hnormal_layer_inputs = 0.
        self.hmem_layer_outputs = 0.
        self.output_layer_inputs = 0.

    def set_fun_type(self, fun_type):
        if fun_type == 'no_lstm':
            self.forward_pass = self.forward_pass_no_lstm
            self.backward_pass = self.backward_pass_no_lstm
            self.update_weights = self.update_weights_no_lstm
        elif fun_type == 'lstm':
            self.forward_pass = self.forward_pass_lstm
            self.backward_pass = self.backward_pass_lstm
            self.update_weights = self.update_weights_lstm
        else:
            print ('fun_type not possible')

    def forward_pass_no_lstm(self, x):
        input_layer_b = self.input_layer.transform(x)

        hnormal_layer_a = input_layer_b.dot(self.w_ihnormal)
        hnormal_layer_b = self.h_normal_layer.transform(hnormal_layer_a)

        output_layer_a = hnormal_layer_b.dot(self.w_hnormalo) + np.ones(1).dot(self.b_o)
        output_layer_b = self.output_layer.transform(output_layer_a)

        self.input_layer_outputs = input_layer_b
        self.hnormal_layer_outputs = hnormal_layer_b
        self.hnormal_layer_inputs = hnormal_layer_a
        self.output_layer_inputs = output_layer_a

        return output_layer_b
    
    def forward_pass_lstm(self, x):
        input_layer_b = self.input_layer.transform(x)

        hnormal_layer_a = input_layer_b.dot(self.w_ihnormal)
        hnormal_layer_b = self.h_normal_layer.transform(hnormal_layer_a)

        hmem_layer_b = np.zeros(self.n_h_mem)
        for i in range(self.n_h_mem):
            hmem_layer_b[i] = self.h_mem_layer[i].forward_pass(input_layer_b)

        output_layer_a = hnormal_layer_b.dot(self.w_hnormalo) + hmem_layer_b.dot(self.w_hmemo) + np.ones(1).dot(
            self.b_o)
        output_layer_b = self.output_layer.transform(output_layer_a)

        self.input_layer_outputs = input_layer_b
        self.hnormal_layer_outputs = hnormal_layer_b
        self.hnormal_layer_inputs = hnormal_layer_a
        self.hmem_layer_outputs = hmem_layer_b
        self.output_layer_inputs = output_layer_a

        return output_layer_b

    def backward_pass_no_lstm(self, e_o_input, exploration):
        d_output_layer = e_o_input * self.output_layer.derivative(self.output_layer_inputs)
        d_hnormal_layer = self.w_hnormalo.dot(d_output_layer) * self.h_normal_layer.derivative(
            self.hnormal_layer_inputs)

        w_hnormalo_deriv = np.outer(self.hnormal_layer_outputs, d_output_layer)
        w_ihnormal_deriv = np.outer(self.input_layer_outputs, d_hnormal_layer)
        b_o_deriv = np.outer(np.ones(1), d_output_layer)

        self.e_hnormalo = self.gamma * self.lam * self.e_hnormalo * (not exploration) + w_hnormalo_deriv
        self.e_ihnormal = self.gamma * self.lam * self.e_ihnormal * (not exploration) + w_ihnormal_deriv
        self.e_bo = self.gamma * self.lam * self.e_bo * (not exploration) + b_o_deriv

    def backward_pass_lstm(self, e_o_input, exploration):
        d_output_layer = e_o_input * self.output_layer.derivative(self.output_layer_inputs)
        d_hnormal_layer = self.w_hnormalo.dot(d_output_layer) * self.h_normal_layer.derivative(
            self.hnormal_layer_inputs)

        w_hnormalo_deriv = np.outer(self.hnormal_layer_outputs, d_output_layer)
        w_ihnormal_deriv = np.outer(self.input_layer_outputs, d_hnormal_layer)
        w_hmemo_deriv = np.outer(self.hmem_layer_outputs, d_output_layer)
        b_o_deriv = np.outer(np.ones(1), d_output_layer)

        self.e_hnormalo = self.gamma * self.lam * self.e_hnormalo * (not exploration) + w_hnormalo_deriv
        self.e_ihnormal = self.gamma * self.lam * self.e_ihnormal * (not exploration) + w_ihnormal_deriv
        self.e_hmemo = self.gamma * self.lam * self.e_hmemo * (not exploration) + w_hmemo_deriv
        self.e_bo = self.gamma * self.lam * self.e_bo * (not exploration) + b_o_deriv

        e_c = self.w_hmemo.dot(d_output_layer)

        for i in range(self.n_h_mem):
            self.h_mem_layer[i].backward_pass(e_c[i], exploration)

    def update_weights_no_lstm(self, ETD):
        self.w_hnormalo += self.alpha * ETD * self.e_hnormalo
        self.w_ihnormal += self.alpha * ETD * self.e_ihnormal
        self.b_o += self.alpha * ETD * self.e_bo

    def update_weights_lstm(self, ETD):
        self.w_hnormalo += self.alpha * ETD * self.e_hnormalo
        self.w_ihnormal += self.alpha * ETD * self.e_ihnormal
        self.w_hmemo += self.alpha * ETD * self.e_hmemo
        self.b_o += self.alpha * ETD * self.e_bo

        for i in range(self.n_h_mem):
            self.h_mem_layer[i].update_weights(ETD)

    def reset(self):
        self.e_hnormalo.fill(0.)
        self.e_hmemo.fill(0.)
        self.e_ihnormal.fill(0.)
        self.e_bo.fill(0.)
        self.input_layer_outputs = 0.
        self.hnormal_layer_outputs = 0.
        self.hnormal_layer_inputs = 0.
        self.hmem_layer_outputs = 0.
        self.output_layer_inputs = 0.

        for i in range(self.n_h_mem):
            self.h_mem_layer[i].reset()

class Network_test(object):
    def __init__(self, weights, n_inputs, n_h_normal, n_h_mem, n_outputs):

        self.n_inputs = n_inputs
        self.n_h_normal = n_h_normal
        self.n_h_mem = n_h_mem
        self.n_outputs = n_outputs

        self.input_layer = activation_unit('identity')
        self.h_normal_layer = activation_unit('tanh')
        self.h_mem_layer = [LSTM_test(weights[-2:], i) for i in range(n_h_mem)]
        self.output_layer = activation_unit('identity')

        self.w_hnormalo = weights[0]
        self.w_hmemo = weights[1]
        self.w_ihnormal = weights[2]
        self.b_o = weights[3]

        self.input_layer_outputs = 0.
        self.hnormal_layer_outputs = 0.
        self.hnormal_layer_inputs = 0.
        self.hmem_layer_outputs = 0.
        self.output_layer_inputs = 0.
        
    def forward_pass(self, x):
        input_layer_b = self.input_layer.transform(x)

        hnormal_layer_a = input_layer_b.dot(self.w_ihnormal)
        hnormal_layer_b = self.h_normal_layer.transform(hnormal_layer_a)

        hmem_layer_b = np.zeros(self.n_h_mem)
        for i in range(self.n_h_mem):
            hmem_layer_b[i] = self.h_mem_layer[i].forward_pass(input_layer_b)

        output_layer_a = hnormal_layer_b.dot(self.w_hnormalo) + hmem_layer_b.dot(self.w_hmemo) + np.ones(1).dot(
            self.b_o)
        output_layer_b = self.output_layer.transform(output_layer_a)

        self.input_layer_outputs = input_layer_b
        self.hnormal_layer_outputs = hnormal_layer_b
        self.hnormal_layer_inputs = hnormal_layer_a
        self.hmem_layer_outputs = hmem_layer_b
        self.output_layer_inputs = output_layer_a

        return output_layer_b
    
    def reset(self):
        self.e_hnormalo.fill(0.)
        self.e_hmemo.fill(0.)
        self.e_ihnormal.fill(0.)
        self.e_bo.fill(0.)
        self.input_layer_outputs = 0.
        self.hnormal_layer_outputs = 0.
        self.hnormal_layer_inputs = 0.
        self.hmem_layer_outputs = 0.
        self.output_layer_inputs = 0.

        for i in range(self.n_h_mem):
            self.h_mem_layer[i].reset()


def get_action(prediction, epsilon=0.05):
    exploration = False
    temperature = 1.

    if np.random.rand() < epsilon:
        probs = np.exp(prediction / temperature) / (np.exp(prediction / temperature).sum())
        action = np.random.choice(np.array([0, 1, 2, 3]), p=probs)

        if action != np.argmax(prediction):
            exploration = True
    else:
        action = np.argmax(prediction)

    return action, exploration


def get_error(Vt, At, Vtnext, reward, gamma, kappa):
    ETD = Vt + (reward + gamma * Vtnext - Vt) / kappa - At

    return ETD


def train_network(cor_length, alpha, n_trainings, disc_factor, noise=None):
    w_init = 0.2

    gamma = 0.98
    lam = 0.8
    kappa = 0.1

    all_rewards = []
    all_nt = []

    network = Network(3, 12, 3, 4, w_init, alpha, gamma, lam)
    network.set_fun_type('lstm')

    total_ts = 0

    for training_iter in range(n_trainings):

        rand_number = np.random.rand()

        task = TaskTmaze(cor_length, rand_number, discount_factor=disc_factor, noise=noise)

        action = 'none'

        [next_input, reward] = task.next_step(action)
        prediction = network.forward_pass(next_input)

        while next_input.any() != 0:
            [action, exploration] = get_action(prediction)
            At = prediction[action]
            Vt = prediction.max()

            e_o_input = np.zeros(4)
            e_o_input[action] = 1.
            network.backward_pass(e_o_input, exploration)
            [next_input, reward] = task.next_step(action)

            if next_input.any() != 0:
                prediction = network.forward_pass(next_input)
                Vtnext = prediction.max()
            else:
                Vtnext = 0.

            ETD = get_error(Vt, At, Vtnext, reward, gamma, kappa)
            network.update_weights(ETD)

        network.reset()

        all_rewards.append(task.get_total_reward())
        all_nt.append(task.t)

        total_ts += task.get_t()

        if (training_iter + 1) % 100 == 0:
            print ('iter', training_iter+1, ' average reward:', (np.round(sum(all_rewards[-100:])/100., 2)))
            
#     plt.scatter(np.arange(n_trainings), all_rewards, lw = 0.05)
#     plt.ylabel('Training iter', fontsize=18)
#     plt.xlabel('Reward', fontsize=18)
#     plt.tick_params(labelsize=14)
#     plt.show()
    return network


def start_training(rng_seed, cor_length, alpha, n_trainings, discount_factor, noise=None):
    import time
    tick = time.time()

    np.random.seed(rng_seed)
    network_trained = train_network(cor_length, alpha, n_trainings, discount_factor, noise=noise)

    elapsed = time.time() - tick
    print ('Elapsed: '+ str(np.int(elapsed)) + ' seconds.')
    weights_xi = [network_trained.h_mem_layer[i].w_xi for i in range(3)]
    weights_xc = [network_trained.h_mem_layer[i].w_xc for i in range(3)]
    weights = [network_trained.w_hnormalo, network_trained.w_hmemo, network_trained.w_ihnormal, 
               network_trained.b_o, weights_xi, weights_xc]
    return weights


def testing(weights, rand_number, cor_length, noise=None):
  arrows = [lookup("UPWARDS ARROW"),lookup("RIGHTWARDS ARROW"),lookup("DOWNWARDS ARROW"),lookup("LEFTWARDS ARROW")]
  network = Network_test(weights, 3, 12, 3, 4)
#  np.random.seed(rng_seed)
#  rand_number = np.random.rand()
#  print("--random number:", np.round(rand_number,2))
#  rand_number = 0.1
  print("--random number:", rand_number)
  
  predictions_df = pd.DataFrame(columns=[lookup("UPWARDS ARROW"),lookup("RIGHTWARDS ARROW"),lookup("DOWNWARDS ARROW"),lookup("LEFTWARDS ARROW")])
  CECstate_df = pd.DataFrame(columns=['CEC_state1','CEC_state2','CEC_state3'])
  S_df = pd.DataFrame(columns=['inputcell'])
  
  task = TaskTmaze(cor_length, rand_number, noise=noise)

  action = 'none'
  [next_input, reward] = task.next_step(action)
  t = 0;

  predictions_df.loc[0] = [0.,0.,0.,0.]
  while next_input.any() != 0:
      t += 1
      prediction = network.forward_pass(next_input)
      predictions_df.loc[t] = prediction
      CECstate_df.loc[t] = [network.h_mem_layer[0].CEC.state,
                      network.h_mem_layer[1].CEC.state,
                      network.h_mem_layer[2].CEC.state]
      
      [action, exploration] = get_action(prediction, 0.)
      print(arrows[action], end = ' ')
      [next_input, reward] = task.next_step(action)


  print("\n reward = ", reward)  
  return {'predictions' :predictions_df, 'CEC' :CECstate_df, 'S' :S_df}

