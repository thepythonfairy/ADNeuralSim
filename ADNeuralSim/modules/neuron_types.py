#File: neuron_types.py
#Assigned to: Piyush Singh

#Task: Build Neuron Type Assigner

'''
Instructions: Write a function that assigns neuron types (e.g., excitatory (E) or inhibitory (I)) to each neuron. It should take 
num_neurons and inhibitory_ratio as inputs and return a list of strings: "E" or "I".

Purpose: This module should lable neurons as excitatory or inhibitory.

Notes: This function will be called from main_simulation.ipynb using the parameters.py. We are dealing with parameter abstraction
so please do not code any actual values. 

Integration:
from neuron_types import assign_neuron_types
types = assign_neuron_types(100)

Edge Cases:
Ratio = 0 or 1, rounding errors

Test: Print types of num_neurons = 10, inhibitory_ratio = 0.3

'''

def assign_neuron_types(num_neurons: int, inhibitory_ratio: float = 0.2)->list