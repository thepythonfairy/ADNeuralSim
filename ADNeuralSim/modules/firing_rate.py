#File: firing_rate.py
#Assigned to: Janet Barba

#Task: Build Firing Rate Calculator

'''
Instructions: Write a function that calculates firing rat efrom a list of spike times. It should take spike_times in ms and a time window in ms (e.g., 1000ms)
and return the firing rate in Hz.

Purpose: This modules should convert spike times into firing rates for comparison and analysis.

Notes: This function will be called from main_simulation.ipynb using the partameters from parameters.py. We are dealing with parameter abstraction
so please do not code any actual values.

Integration:
from firing_rate import calculate_firing_rate
rate = calculate_firing_rate(spikes)

Edge Cases:
Empty spike list returns 0
Zero or negative window, raise ValueError

Test: Print firing rate for [30, 50, 180, 500] in a 1000 ms window

'''


def calculate_firing_rate(spike_times: list, window_ms: int = 1000)->float
