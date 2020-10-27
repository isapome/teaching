#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:30:34 2019

@author: Isabella
"""
import TMaze_task_tutorial_2019 as TMaze
import pickle
import matplotlib.pyplot as plt

#this script runs a TMaze experiment (TMaze_task_tutorial_2019.py) given as input the seed, 
#the length of the corridor, the learning rate, the discount factor, the noise and the number
# of trainings.

def plot_qvalues(predictions, save_name=None):
    ax = predictions.plot(lw = 2)
    ax.set_ylabel('Q-values', fontsize=18)
    ax.set_xlabel('Action #', fontsize=18)
    ax.tick_params(labelsize=14)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0., fontsize=12)

    plt.tight_layout()
    plt.show()
    if save_name:
        plt.savefig(save_name, bbox_inches='tight') 
    return

def experiment(mode, save_weights=False, filename=None):
    if noise:
        str_noise = 'noisy'
    else:
        str_noise = 'noiseless'
    print("TMaze "+str_noise+" experiment. Corridor length: ", corridor_length, "\n")
    if filename:
        if filename[-4:]!= '.pkl':
            raise ValueError('The filename needs to end with .pkl') 
    if mode=='train':
        print("Training process:")
        weights = TMaze.start_training(seed, corridor_length, learning_rate, n_trainings, discount_factor, noise=noise)
        if save_weights == True:
            if filename:
                with open(filename, 'wb') as file:
                    pickle.dump(weights, file)
            else:
                raise ValueError('For save mode the name of the weight file needs to be provided.')  
    elif mode=='load':
        print("Loading the weights...")
        if filename:
            with open(filename, 'rb') as file:
                weights = pickle.load(file, encoding='latin1')
                print("...weights loaded.")
        else:
            raise ValueError('For load mode the name of the weight file needs to be provided.') 
    else:
        raise ValueError('Unknown mode. Choose \'train\' or \'load\'.')
        
    print(" ")
    print("Testing the network:")
    Results = {}
    for i in [0.1,  0.9]:
        print( " ")
        results = TMaze.testing(weights, i, corridor_length, noise=noise)
        plot_qvalues(results['predictions'])
        Results['results'+str(i)] = results
    return Results


#----Test 3 different pretrained networks: 
# filenames = ['weights_trial_10_noiseless.pkl', 'weights_trial_10_noisy.pkl', 'weights_trial_20_noiseless.pkl']
# corridor_lengths = [10, 10, 20]
# noises = [None, True, None]
# for i in range(len(filenames)):
#     noise = noises[i]
#     corridor_length = corridor_lengths[i]
#     results = experiment('load', filename=filenames[i])

#----Train network and save weights (if save_weights==True)
#seed = 0
#corridor_length = 10
#learning_rate = 0.0005
#discount_factor = -0.4
#noise = True #choose None for noiseless, anything else for noisy
#n_trainings = 50000 
#weights_filename = 'weights_trial.pkl'
#results = experiment('train', save_weights=False, filename=weights_filename)


