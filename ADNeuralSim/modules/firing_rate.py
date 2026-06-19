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

def calculate_firing_rate(spike_times: list, window_ms: int = 1000)->float:
    if window_ms is None or window_ms <= 0:
        raise ValueError("window_ms must be a positive integer")
    if not spike_times:
        return 0.0

    #Count spikes within the window [0, window_ms]
    count = sum(0 <= t <= window_ms for t in spike_times)
    #Convert to Hz: spikes per second
    return (count / window_ms) * 1000.0

    #Test
    print(calculate_firing_rate([30, 50, 100, 500], window_ms=1000)) #Expected value is 4 Hz