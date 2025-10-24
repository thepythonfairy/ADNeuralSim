#File: synaptic delay
#Assigned to: Yael Robert

#Task: Build Synaptic Delay Generator

'''
Instructions: Write a function that assigns synaptic delays to each connection in a cpnnectvity matrix. It should return
a matrix of the same shape with random delays between 1 and 20 ms.

Purpose: This module should assign random delays in ms to each connection in the connectivity matric.

Notes: this function will be called from main_simulation.ipynb using the paramnters from paramters.py. We are dealing
with parameter abstraction so please do not code any actua; values. Accept inputs and return the matrix.

Integration:
from synaptic_delay import generate_synaptic_delays
delays = generate_synaptic_delays_(matrix)

Edge Cases: Handle all empty matrices, all zero matrices, and min = max cases

Test: Add a _main_block that prints a sample delay matrix for a 5x5 connectivity matrix

'''

import numpy as np
def generate_synaptic_delays(connectivity_matrix: np.ndarray, min_delay: int = 1, max_delay: int = 20)->np.ndarray
