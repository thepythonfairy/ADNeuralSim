#File: hh_model.py
#Assigned to: Steven Dang

#Task: Build Hodgkin Huxley Model

'''
Instructions: Implement a fucntion that returns a matrix or data structure representing the evolution of membrane potential 
and gating variables over time.

Purpose: To capture the voltage-dependant behavior of sodium and potassium channels and their contribution to action
potential generation.

Notes: This function will be called from main_simulation.ipynb using the parameters from parameters.py.

The gate values used are standard rest values found at around 65m from the original squid axom experiment in 1952

Integration: This function will be imported and called from main_simulation.ipynb, it returns a brian2 neuron group 
constructed from parameters from parameters.py and models the time evolution of membrane potential and gating variables (m, h, n)

Edge Cases: 

Test: 

'''

from brian2 import*
def build_hh_neuron_group(num_neurons: int, params: dict) -> NeuronGroup:
    EL = params["EL"]
    Cm = params["Cm"]
    gNa = params["gNa"]
    gK = params["gK"]
    gL = params["gL"]
    Ena = params["ENa"]
    EK = params["EK"]
    I = params["I"]

    #add the equations
    eqs = '''
    dv/dt = (I - gNa*(m**3)*h*(v - ENa) - gK*(n**4)*(v - EK) - gL*(v-EL)) / Cm : volt
    dm/dt = alpha_m*(1 - m) - beta_m*m : 1
    dh/dt = alpha_h*(1 - h) - beta_h*h : 1
    dn/dt = alpha_n*(1 - n) - beta_n*n : 1
    alpha_m = 0.1*(mV**-1)*(v + 40*mV)/(1 - exp(-(v + 40*mV)/(10*mV)))/ms : Hz
    beta_m = 4*exp(-(v + 65*mV)/(18*mV))/ms : Hz
    alpha_h = 0.07*exp(-(v + 65*mV)/(20*mV))/ms : Hz
    beta_h = 1/(1 + exp(-(v + 35*mV)/(10*mV)))/ms : Hz
    alpha_n = 0.01*(mV**-1)*(v + 55*mV)/(1 - exp(-(v + 55*mV)/(10*mV)))/ms : Hz
    beta_n = 0.125*exp(-(v + 65*mV)/(80*mV))/ms : Hz
    I : amp
    '''

    neurons = NeuronGroup(
        num_neurons,
        model=eqs,
        threshold='v > -40*mV',
        refractory='v > -40*mV',
        method='exponential_euler',
        dt=params['dt_ms'] * ms
    )

    # input current
    neurons.I = .7 * nA * neurons.i / num_neurons


    # Initialize at rest (values from 1952 squid axom)
    v_rest = EL
    alpha_m = 0.1*(v_rest/mV + 40)/(1 - exp(-(v_rest/mV + 40)/10)) / ms
    beta_m = 4*exp(-(v_rest/mV + 65)/18) / ms
    alpha_h = 0.07*exp(-(v_rest/mV + 65)/20) / ms
    beta_h = 1/(1 + exp(-(v_rest/mV + 35)/10)) / ms
    alpha_n = 0.01*(v_rest/mV + 55)/(1 - exp(-(v_rest/mV + 55)/10)) / ms
    beta_n = 0.125*exp(-(v_rest/mV + 65)/80) / ms

    m_inf = alpha_m / (alpha_m + beta_m)
    h_inf = alpha_h / (alpha_h + beta_h)
    n_inf = alpha_n / (alpha_n + beta_n)

    neurons.v = EL
    neurons.m = m_inf
    neurons.h = h_inf
    neurons.n = n_inf

    #Assign parameters from the params dictionary here
    #Ex: neurons.gNa = params['gNa']
    #Make sure to set initial values for v, m, h, n


    return neurons
