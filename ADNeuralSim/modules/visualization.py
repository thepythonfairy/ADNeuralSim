#File: visualization.py
#Assigned to: Janet Barba

#Task: Build Visualization

'''
Instructions: Write a function that plots spike trains using matplotlib. It should take a list of spiek times and display a raster plot.

Purpose: This module should plot spike times as a raster plot.

Notes: This function will be called from main_simulation.ipynb using the parameters from parameters.py. We are dealing with parameter abstraction
so please do not code any actual values.

Integration:
from visualization import plot_spike_train
plot_spike_train([10, 50, 90, 200], neuron_id=0)

from visualization import plot_population_spike_trains
plot_population_spike_trains(spike_monitor)

Edge Cases: If empty spike list, show warning and skip plot

Test: Plot [10, 50, 90, 200]

'''

import matplotlib.pyplot as plt 
#This plots for a single neuron (this part is for testing purposes)
def plot_spike_train(spike_times: list, neuron_id: int = 0):
    
    #Plot spike times for a single neuron as a raster line
    if not spike_times:
        print(f"Warning: No spikes to plot for neuron {neuron_id}.")    #There was a syntax error so f is now included
        return

    plt.figure(figsize=(8, 2.5))
    plt.eventplot([spike_times], colors='black', lineoffsets=1, linelengths=0.8)
    plt.yticks([1], [f"Neuron {neuron_id}"])
    plt.xlabel("Time (ms)")
    plt.title("Spike train raster")
    plt.tight_layout()
    plt.show()
    
#Population raster Brian2 spikeMonitor

#This plots for multiple neurons its the population raster
def plot_population_spike_trains(spike_monitor):
    
    #Plot spike trains for all neurons recorded by Brian2 SpikeMonitor
    if len(spike_monitor.t) == 0:
       print("Warning: No spikes recorded in population")
       return

    num_neurons = int(spike_monitor.source.N)
    spike_times = [spike_monitor.spike_trains()[i] for i in range(num_neurons)]

    plt.figure(figsize=(10, 6))
    plt.eventplot(spike_times, colors='black')
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron index")
    plt.title("Population Spike Raster")
    plt.tight_layout()
    plt.show()

    