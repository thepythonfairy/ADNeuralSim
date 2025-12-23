#File: connectivity.py
#Assigned to: Janet Barba

#Task: Build Connectivity Matrix

'''
Instructions: Write a function that generates a random connectivity matric between neurons. It should take num_neurons
and connection_prob as inputs and return a NumPy array of shape (num_neurons, num_neurons) with binary values
(1 for connection and 0 for no conenction)

Purpose: This module should define which neurons are connected

Notes: This function will be called from main_simulation.ipynb using the parameters from parameters.py. We are dealing
with parameter abstraction so please do not code any actual values. Accept inpiuts and return the matrix.

Integration: 
from connectivity import generate_connectivity_matrix
matrix = generate_connectivity_matrix(100, 0.1)

Edge Cases:
num_neurons = 0, return empty matrix
connection_prob = 0 or 1, return 0
No self connection allowed, diagonal should be all 0s

'''

import numpy as np

def generate_connectivity_matrix(num_neurons: int, connection_prob: float)->np.ndarray:
