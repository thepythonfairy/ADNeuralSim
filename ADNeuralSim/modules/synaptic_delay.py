# File: synaptic delay
# Assigned to: Yael Robert

'''
Purpose: This module should assign random delays in ms to each connection in the connectivity matrix.
'''

import numpy as np

def generate_synaptic_delays(connectivity_matrix: np.ndarray, min_delay: int = 1, max_delay: int = 20):
    """
    Generates a matrix of synaptic delays for each connection in the connectivity matrix.
    
    Parameters:
    - connectivity_matrix: A numpy array representing the connectivity matrix.
    - min_delay: Minimum delay in milliseconds (default is 1 ms).
    - max_delay: Maximum delay in milliseconds (default is 20 ms).
    
    Returns:
    - A numpy array of the same shape as connectivity_matrix with random delays between min_delay and max_delay.
    """
    if connectivity_matrix.size == 0: # Check if the matrix is empty
        return connectivity_matrix # No delays for an empty matrix
    if np.all(connectivity_matrix == 0): # Check if the matrix is all zeros
        return connectivity_matrix # No delays for no connections
    if min_delay == max_delay: # Check if min_delay and max_delay are the same
        return np.full_like(connectivity_matrix, min_delay) # All delays are the same
    
    random_delays = np.random.randint(min_delay, max_delay + 1, size=connectivity_matrix.shape)
    delays = np.where(connectivity_matrix != 0, random_delays, 0)
    return delays

# Test the function
def test_generate_synaptic_delays():
    # Test with a 5x5 connectivity matrix
    connectivity_matrix = np.ones((5, 5))
    delays = generate_synaptic_delays(connectivity_matrix)
    assert delays.shape == connectivity_matrix.shape, "The shape of the delays matrix is incorrect."
    assert np.all(delays >= 1) and np.all(delays <= 20), "Delays are out of the specified range."
    
    # Test with an empty matrix
    empty_matrix = np.array([])
    delays = generate_synaptic_delays(empty_matrix)
    assert delays.size == 0, "The delays matrix for an empty input should also be empty."
    
    # Test with a zero matrix
    zero_matrix = np.zeros((3, 3))
    delays = generate_synaptic_delays(zero_matrix)
    assert delays.shape == zero_matrix.shape, "The shape of the delays matrix for a zero input is incorrect."
    assert np.all(delays == 0), "Delays for a zero input should remain zero (no connections)."

    # Test with matrix where some connections are present
    mixed_matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    delays = generate_synaptic_delays(mixed_matrix)
    assert delays.shape == mixed_matrix.shape, "The shape of the delays matrix for a mixed input is incorrect."
    assert np.all(delays[delays != 0] >= 1) and np.all(delays[delays != 0] <= 20), "Delays for non-zero connections are out of the specified range." # Check if non-zero delays are within the range
    assert np.all(delays[mixed_matrix == 0] == 0), "Delays for zero connections should remain zero." # Check if zero delays remain zero
    
    # Test with min_delay equal to max_delay
    min_max_matrix = np.ones((2, 2))
    delays = generate_synaptic_delays(min_max_matrix, min_delay = 5, max_delay = 5)
    assert np.all(delays == 5), "Delays should all be 5 when min_delay equals max_delay."
    
    print("All tests passed!")

if __name__ == "__main__":
    test_generate_synaptic_delays()