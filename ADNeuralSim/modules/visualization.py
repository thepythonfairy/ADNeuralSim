#File: visualization.py
#Assigned to: Karthik Emani

#Task: Build Visualization

'''
Instructions: Write a function that plots spike trains using matplotlib. It should take a list of spiek times and display a raster plot.

Purpose: This module should plot spike times as a raster plot.

Notes: This function will be called from main_simulation.ipynb using the parameters from parameters.py. We are dealing with parameter abstraction
so please do not code any actual values.

Integration:
from visualization import plot_spike_train
plot_spike_train(spikes)

Edge Cases: If empty spike list, show warning and skip plot

Test: Plot [10, 50, 90, 200]

'''

import matplotlib.pyplot as plt 
def plot_spike_train(spike_times: list, neuron_id: int = 0):
    if not spike_times:
        print("Warning: No spikes to plot for neuron {neuron_id}.")
        return

        #Continue the rest here