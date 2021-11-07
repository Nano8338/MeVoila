import numpy as np

# region Monte Carlo


class MonteCarloGenerator:
    def __init__(self, seed=0):
        self._generator = np.random.Generator(np.random.MT19937(seed))

    def simulate_normal(self, simulations_N):
        return self._generator.standard_normal(simulations_N)


# endregion
