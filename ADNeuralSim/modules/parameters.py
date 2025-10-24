#This module defines the parameters of the ENTIRE simulation
#Super Imporant: Citations for the values must be included in Methodology section of the Literature Review

#Hodkin-Huxley Parameters (used in hh_model.py), units are specified in Brian2 syntax
from brian2 import mV, ms, uF, mS, nA, cm

HH_PARAMS = {
    'Cm': 1 * uF / cm**2,
    'gNa': 120 * mS / cm**2,
    'gK': 36 * mS / cm**2,
    'gL': 0.3 * mS / cm**2,
    'ENa': 50 * mV,
    'EK': -77 * mV,
    'EL': -54.4 * mV,
    'I': 10 * nA,
    'dt_ms': DT_MS
}

#Network Parameters
NUM_NEURONS = 30
CONNECTION_PROB = 0.1
INHIBITORY_RATIO = 0.2
MIN_DELAY_MS = 1
MAX_DELAY_MS = 5

#Simulation Parameters
TIME_WINDOW_MS = 1000
DT_MS = 1

#Condition Parameters
HEALTHY_RATE_RANGE = (5,15) #Hz
AD_RATE_RANGE = (3,8) #Hz