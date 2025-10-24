#Assigned to: Janet Barba

import numpy as np
def generate_spike_train(duration_ms=1000, rate_hz=10):
    num_spikes = int(duration_ms / 1000 * rate_hz)
    spike_times = np.sort(np.random.uniform(0, duration_ms, num_spikes))
    return spike_times