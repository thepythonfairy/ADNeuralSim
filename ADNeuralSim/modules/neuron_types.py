#File: neuron_types.py
#Assigned to: Piyush Singh


"""
File: neuron_types.py
Purpose: Define neuron models for simulating neural activity in Alzheimer's disease research

This module contains:
1. PoissonNeuron: Generates spike trains using Poisson process
2. HodgkinHuxleyNeuron: Simulates action potentials using simplified HH equations

These classes are used to model both healthy and degraded neural signals in AD.
"""

import numpy as np
import matplotlib.pyplot as plt


class PoissonNeuron:
    """
    A neuron that generates spikes following a Poisson process.
    
    In a Poisson process, spikes occur randomly but at a constant average rate (lambda).
    This is useful for modeling spontaneous neural activity.
    """
    
    def __init__(self, firing_rate):
        """
        Initialize the Poisson neuron.
        
        Parameters:
        -----------
        firing_rate : float
            Average firing rate in Hz (spikes per second).
            Example: 10 Hz means on average 10 spikes per second.
        """
        self.firing_rate = firing_rate  # Store the firing rate (lambda)
        self.spike_times = []  # Will store the times when spikes occur
    
    def generate_spikes(self, duration_ms, dt_ms=1.0):
        """
        Generate spike times for a given duration.
        
        Parameters:
        -----------
        duration_ms : float
            Total simulation time in milliseconds
        dt_ms : float
            Time step in milliseconds (default: 1 ms)
        
        Returns:
        --------
        spike_times : numpy array
            Array of times (in ms) when spikes occurred
        """
        # Convert firing rate from Hz to probability per time step
        # firing_rate is in Hz (spikes/second), dt_ms is in ms
        # Probability = rate * time_step
        rate_per_ms = self.firing_rate / 1000.0  # Convert Hz to spikes/ms
        spike_probability = rate_per_ms * dt_ms  # Probability of spike in each time step
        
        # Calculate number of time steps
        num_steps = int(duration_ms / dt_ms)
        
        # Create time array
        time_array = np.arange(0, duration_ms, dt_ms)
        
        # Generate random numbers for each time step
        # If random number < probability, a spike occurs
        random_numbers = np.random.rand(num_steps)
        
        # Find where spikes occur (where random number < probability)
        spike_mask = random_numbers < spike_probability
        
        # Get the actual spike times
        self.spike_times = time_array[spike_mask]
        
        return self.spike_times
    
    def get_spike_count(self):
        """
        Returns the total number of spikes generated.
        """
        return len(self.spike_times)
    
    def plot_spikes(self, duration_ms=None):
        """
        Plot the spike times as a raster plot.
        
        Parameters:
        -----------
        duration_ms : float
            Total duration to show on x-axis. If None, uses max spike time.
        """
        if len(self.spike_times) == 0:
            print("No spikes to plot. Run generate_spikes() first.")
            return
        
        plt.figure(figsize=(10, 3))
        
        # Plot spikes as vertical lines
        plt.eventplot(self.spike_times, colors='blue', linewidths=1.5)
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Spikes')
        plt.title(f'Poisson Neuron Spike Train (λ = {self.firing_rate} Hz)')
        plt.ylim([0.5, 1.5])
        
        if duration_ms:
            plt.xlim([0, duration_ms])
        
        plt.tight_layout()
        plt.show()


class HodgkinHuxleyNeuron:
    """
    A neuron that simulates action potentials using the Hodgkin-Huxley model.
    
    The HH model describes how action potentials are initiated and propagated
    in neurons through voltage-gated sodium and potassium channels.
    """
    
    def __init__(self, params=None):
        """
        Initialize the Hodgkin-Huxley neuron with default or custom parameters.
        
        Parameters:
        -----------
        params : dict (optional)
            Dictionary of HH parameters. If None, uses standard values.
        """
        # Set default parameters (standard HH values)
        if params is None:
            self.params = {
                'C_m': 1.0,      # Membrane capacitance (μF/cm²)
                'g_Na': 120.0,   # Maximum sodium conductance (mS/cm²)
                'g_K': 36.0,     # Maximum potassium conductance (mS/cm²)
                'g_L': 0.3,      # Leak conductance (mS/cm²)
                'E_Na': 50.0,    # Sodium reversal potential (mV)
                'E_K': -77.0,    # Potassium reversal potential (mV)
                'E_L': -54.4,    # Leak reversal potential (mV)
                'I_ext': 10.0    # External applied current (μA/cm²)
            }
        else:
            self.params = params
        
        # Initialize state variables
        self.V = -65.0    # Membrane potential (mV), starts at resting potential
        self.m = 0.05     # Sodium activation gate (0 to 1)
        self.h = 0.6      # Sodium inactivation gate (0 to 1)
        self.n = 0.32     # Potassium activation gate (0 to 1)
        
        # Storage for simulation results
        self.time_array = None
        self.V_trace = None
        self.m_trace = None
        self.h_trace = None
        self.n_trace = None
    
    def alpha_m(self, V):
        """Sodium activation rate constant"""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    def beta_m(self, V):
        """Sodium activation rate constant"""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    def alpha_h(self, V):
        """Sodium inactivation rate constant"""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    def beta_h(self, V):
        """Sodium inactivation rate constant"""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def alpha_n(self, V):
        """Potassium activation rate constant"""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    def beta_n(self, V):
        """Potassium activation rate constant"""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    def I_Na(self, V, m, h):
        """Sodium current"""
        return self.params['g_Na'] * (m ** 3) * h * (V - self.params['E_Na'])
    
    def I_K(self, V, n):
        """Potassium current"""
        return self.params['g_K'] * (n ** 4) * (V - self.params['E_K'])
    
    def I_L(self, V):
        """Leak current"""
        return self.params['g_L'] * (V - self.params['E_L'])
    
    def simulate(self, duration_ms, dt_ms=0.01, I_ext=None):
        """
        Simulate the neuron's membrane potential over time.
        
        Parameters:
        -----------
        duration_ms : float
            Total simulation time in milliseconds
        dt_ms : float
            Time step in milliseconds (default: 0.01 ms for stability)
        I_ext : float or callable (optional)
            External current. Can be:
            - A constant value (μA/cm²)
            - A function that takes time (ms) and returns current
            If None, uses the default I_ext from params
        
        Returns:
        --------
        time_array : numpy array
            Time points (ms)
        V_trace : numpy array
            Membrane potential at each time point (mV)
        """
        # Set up time array
        num_steps = int(duration_ms / dt_ms)
        self.time_array = np.linspace(0, duration_ms, num_steps)
        
        # Initialize arrays to store results
        self.V_trace = np.zeros(num_steps)
        self.m_trace = np.zeros(num_steps)
        self.h_trace = np.zeros(num_steps)
        self.n_trace = np.zeros(num_steps)
        
        # Set initial conditions
        self.V_trace[0] = self.V
        self.m_trace[0] = self.m
        self.h_trace[0] = self.h
        self.n_trace[0] = self.n
        
        # Determine external current function
        if I_ext is None:
            I_ext_func = lambda t: self.params['I_ext']
        elif callable(I_ext):
            I_ext_func = I_ext
        else:
            I_ext_func = lambda t: I_ext
        
        # Run simulation using Euler method
        for i in range(1, num_steps):
            # Get current state
            V = self.V_trace[i-1]
            m = self.m_trace[i-1]
            h = self.h_trace[i-1]
            n = self.n_trace[i-1]
            t = self.time_array[i-1]
            
            # Calculate rate constants at current voltage
            am = self.alpha_m(V)
            bm = self.beta_m(V)
            ah = self.alpha_h(V)
            bh = self.beta_h(V)
            an = self.alpha_n(V)
            bn = self.beta_n(V)
            
            # Calculate currents
            I_Na_val = self.I_Na(V, m, h)
            I_K_val = self.I_K(V, n)
            I_L_val = self.I_L(V)
            I_ext_val = I_ext_func(t)
            
            # Calculate derivatives
            dV = (I_ext_val - I_Na_val - I_K_val - I_L_val) / self.params['C_m']
            dm = am * (1 - m) - bm * m
            dh = ah * (1 - h) - bh * h
            dn = an * (1 - n) - bn * n
            
            # Update state variables using Euler method
            self.V_trace[i] = V + dV * dt_ms
            self.m_trace[i] = m + dm * dt_ms
            self.h_trace[i] = h + dh * dt_ms
            self.n_trace[i] = n + dn * dt_ms
        
        # Update final state
        self.V = self.V_trace[-1]
        self.m = self.m_trace[-1]
        self.h = self.h_trace[-1]
        self.n = self.n_trace[-1]
        
        return self.time_array, self.V_trace
    
    def plot_voltage(self):
        """
        Plot the membrane potential over time.
        """
        if self.V_trace is None:
            print("No simulation data. Run simulate() first.")
            return
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.time_array, self.V_trace, 'b-', linewidth=1.5)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.title('Hodgkin-Huxley Neuron: Action Potential')
        plt.grid(True, alpha=0.3)
        
        # Plot gating variables
        plt.subplot(2, 1, 2)
        plt.plot(self.time_array, self.m_trace, 'r-', label='m (Na activation)', linewidth=1.5)
        plt.plot(self.time_array, self.h_trace, 'g-', label='h (Na inactivation)', linewidth=1.5)
        plt.plot(self.time_array, self.n_trace, 'b-', label='n (K activation)', linewidth=1.5)
        plt.xlabel('Time (ms)')
        plt.ylabel('Gating Variable')
        plt.title('Gating Variables (Ion Channel States)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("="*60)
    print("TESTING NEURON MODELS FOR ALZHEIMER'S DISEASE SIMULATION")
    print("="*60)
    
    # Test 1: Poisson Neuron (Spike Train Generation)
    print("\n--- Test 1: Poisson Neuron ---")
    
    # Simulate a healthy neuron with 10 Hz firing rate
    print("Simulating healthy neuron (10 Hz)...")
    healthy_neuron = PoissonNeuron(firing_rate=10.0)
    healthy_spikes = healthy_neuron.generate_spikes(duration_ms=1000.0, dt_ms=1.0)
    
    print(f"Generated {healthy_neuron.get_spike_count()} spikes")
    print(f"First 10 spike times (ms): {healthy_spikes[:10]}")
    
    # Simulate an AD-affected neuron with reduced firing rate
    print("\nSimulating AD-affected neuron (5 Hz)...")
    ad_neuron = PoissonNeuron(firing_rate=5.0)
    ad_spikes = ad_neuron.generate_spikes(duration_ms=1000.0, dt_ms=1.0)
    
    print(f"Generated {ad_neuron.get_spike_count()} spikes")
    print(f"First 10 spike times (ms): {ad_spikes[:10]}")
    
    # Plot both spike trains for comparison
    print("\nPlotting spike trains...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(2, 1, 1)
    plt.eventplot(healthy_spikes, colors='blue', linewidths=1.5)
    plt.title('Healthy Neuron (10 Hz)')
    plt.ylabel('Spikes')
    plt.xlim([0, 1000])
    plt.ylim([0.5, 1.5])
    
    plt.subplot(2, 1, 2)
    plt.eventplot(ad_spikes, colors='red', linewidths=1.5)
    plt.title('AD-Affected Neuron (5 Hz)')
    plt.ylabel('Spikes')
    plt.xlabel('Time (ms)')
    plt.xlim([0, 1000])
    plt.ylim([0.5, 1.5])
    
    plt.tight_layout()
    plt.show()
    
    # Test 2: Hodgkin-Huxley Neuron (Action Potential)
    print("\n--- Test 2: Hodgkin-Huxley Neuron ---")
    
    # Simulate a healthy HH neuron with constant current injection
    print("Simulating healthy HH neuron with constant current...")
    hh_neuron = HodgkinHuxleyNeuron()
    time, voltage = hh_neuron.simulate(duration_ms=50.0, dt_ms=0.01, I_ext=10.0)
    
    print(f"Simulation completed: {len(time)} time points")
    print(f"Voltage range: {np.min(voltage):.2f} to {np.max(voltage):.2f} mV")
    
    # Plot the action potential
    print("\nPlotting action potential...")
    hh_neuron.plot_voltage()
    
    #Test 3: HH Neuron with Pulsed Current
    print("\n--- Test 3: HH Neuron with Current Pulse ---")
    
    # Define a current pulse (on from 10-40 ms)
    def current_pulse(t):
        if 10.0 <= t <= 40.0:
            return 10.0  # μA/cm²
        else:
            return 0.0
    
    print("Simulating HH neuron with current pulse (10-40 ms)...")
    hh_neuron2 = HodgkinHuxleyNeuron()
    time2, voltage2 = hh_neuron2.simulate(duration_ms=80.0, dt_ms=0.01, I_ext=current_pulse)
    
    print(f"Peak voltage reached: {np.max(voltage2):.2f} mV")
    
    # Plot the result
    plt.figure(figsize=(12, 6))
    
    # Plot membrane potential
    plt.subplot(2, 1, 1)
    plt.plot(time2, voltage2, 'b-', linewidth=1.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('HH Neuron Response to Current Pulse')
    plt.grid(True, alpha=0.3)
    
    # Plot the applied current
    plt.subplot(2, 1, 2)
    current_trace = [current_pulse(t) for t in time2]
    plt.plot(time2, current_trace, 'r-', linewidth=1.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Applied Current (μA/cm²)')
    plt.title('External Current Injection')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nThese neuron models can now be used to simulate")
    print("signal degradation in Alzheimer's disease research.")
