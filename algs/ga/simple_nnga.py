from problems.RewardFitness import RewardFitness
import numpy as np

class SimpleNNGAIndividual(np.ndarray):
    def __init__(self):
        super().__init__(self)


if __name__ == "__main__":
    sol = SimpleNNGAIndividual([1, 2, 3])
    np.zeros(5)
    print(sol)

