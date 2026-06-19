# ADNeuralSim

A simulation framework for modeling neural activity in Alzheimer's Disease using Hodgkin-Huxely dynamics and spike train analysis.

## Folder Structure
- 'notebooks/' - Contains the main file (main_simulation.ipynb)
- 'modules/' - Python modules for neuron models, connectiivity, synaptic delays, etc.
- 'docs/' - README file, documentation, onboarding notes, etc.
- 'archive/' - For experimental or old notebooks/modules

## How to run
1. Open 'main_simulation.ipynb' in VS Code
2. Run cells sequentially
3. Outputs will appear in the notebook or 'outputs/' folder

## Simulation Overview

### 1. 'main_simulation.ipynb'
Central notebooks that imports all modules and coordinates the simulation pipeline

**Divided into cells for**
-Optional spike train test (outside Brain2)
-Module imports
-Network building
-Brain2 simulation
-Visualization

The main file calls all other modules using abstracted parameters from 'parameter.py'

---

### 2. 'parameters.py'
Defines all constraints for the simulation

- **Neuron & network:** 'NUM_NEURONS',' CONNECTION_PROB', 'INHIBITORY_RATIO'
- **Timing:** 'TIME_WINDOW_MS', 'DT_MS', 'MIN_DELAY_MS', 'MAX_DELAY_MS'
- **Biologically based Model:** 'HH_PARAMS', (Hodkin-Huxley constants)
- **Conditions:** 'HEALTHY_RATE_RANGE', 'AD_RATE_RANGE'

*Citation Reminder:* These values must be cited in our literature review. Refer back to the parameters file via the shared google drive to understand where I derived the values and approximations from for each parameter.

---

### 3. Network Construction Modules
ALL modules use parameter abstraction - there are no hard coded values.

- **'connectivity.py':** Builds binary connectivity matrix, assigned to - Janet Barba
- **'synaptic_delay.py':** Assigns random delays to connections, assigned to - Yael Robert
- **'neuron_types.py':** Lables neurons as excitatory/inhibitory, assigned to - Piyush Singh
- **'hh_model.py':** Constructs HH neuron group in Brian2, assigned to - Steven Dang

These modules utilize the inputs from 'parameters.py' and will be called from 'main_simulation.ipynb'

---

### 4. Spike Train Generation
Spike trains are used for firing rate analysis and visualization

- **'spike_train.py':** Generates Poisson spike trains outside Brian2 (this is an additional test I included, will probably remove later since it's redundant)
- **Brain2 Simulation:** Uses 'SpikeMonitor' and 'StateMonitor to record spikes and voltages

---

### 5. Firing Rate Analysis
Useful for comparing healthy vs AD conditions, can be used post-simulation to analyze neuron behavior

- **'firing_rate.py':** Converst spike times to firing rate (Hz), - assigned to Janet Barba

---

### 6. Visualization
Assists in validating neuron firing and membrane potential evolution, gives warning if no spikes are recorded.

- **'vizualization.py:'** Raster plot of population spike trains, - assigned to Karthik Emani

---

## Important Info
- Each module is parameter-driven and independantly testable.
- The main file imports the modules and passes the parameters
- Add comments throughout your code where necessary to ensure clarity
- Everyone in the team should have access to our github repo, if you don't then email me ASAP
- When you finish your assigned module, email it to me so I can update it to the github repo

The structure of this simulation allows for everyone to work in parallel without interfering with the simulation's framework or breaking anything lol


