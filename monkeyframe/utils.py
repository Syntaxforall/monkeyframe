import numpy as np

def _check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    return np.random.RandomState(seed)