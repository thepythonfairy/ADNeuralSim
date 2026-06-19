#File: neuron_types.py
#Assigned to: Piyush Singh

#Task: Build Neuron Types Model

import numpy as np

def assign_neuron_types(num_neurons:int, inhibitory_ratio: float):

    if num_neurons <= 0:
        return np.array([], dtype=int)
    
    num_inhibitory = int(num_neurons * inhibitory_ratio)

    types = np.ones(num_neurons, dtype=int)

    inhibitory_indices = np.random.choice(num_neurons, num_inhibitory, replace=False)
    types[inhibitory_indices] = -1

    return types