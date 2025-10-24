#File: hh_model.py
#Assigned to: Steven Dang

#Task: Build Hodgkin Huxley Model

'''
Instructions: Implement a fucntion that returns a matrix or data structure representing the evolution of membrane potential 
and gating variables over time.

Purpose: To capture the voltage-dependant behavior of sodium and potassium channels and their contribution to action
potential generation.

Notes: This function will be called from main_simulation.ipynb using the parameters from parameters.py.

Integration:

Edge Cases:

Test:

'''

from brian2 import*
def build_hh_neuron_group(num_neurons: int, params: dict) -> NeuronGroup:
    #add the equations
    eqs = '''
    dv/dt = (I - gNa*(m**3)*h*(v - ENa) - gK*(n**4)*(v - EK) - gL*(v-EL)) / Cm : volt
    dm/dt = alpha_m*(1 - m) - beta_m*m : 1
    dh/dt = alpha_h*(1 - h) - beta_h*h : 1
    dn/dt = alpha_n*(1 - n) - beta_n*n : 1
    alpha_m = 0.1*(mV**-1)*(v + 40*mV)/(1 - exp(-(v + 40*mV)/(10*mV)))/ms : Hz
    beta_m = 4*exp(-(v + 65*mV)/(18*mV))/ms : Hz
    alpha_h = 0.07*exp(-(v + 65*mV)/(20))/ms : Hz
    beta_h = 1/(1 + exp(-(v + 35*mV)/(10*mV)))/ms : Hz
    alpha_n = 0.01*(mV**-1)*(v + 55*mV)/(1 - exp(-(v + 55*mV)/(10*mV)))/ms : Hz
    beta_n = 0.125*exp(-(v + 65*mV)/(80*mV))/ms : Hz
    I : amp
    '''

    neurons = NeuronGroup(
        num_neurons,
        model=eqs,
        method='exponential_euler',
        dt=params['dt_ms'] * ms
    )

    #Assign parameters from the params dictionary here
    #Ex: neurons.gNa = params['gNa']
    #Make sure to set initial values for v, m, h, n

    return neurons