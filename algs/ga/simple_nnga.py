import numpy as np
import numpy.random as rnd
from deap import creator



class SimpleNNGA:
    def __init__(self, max_layers, layer_options):
        self.max_layers = max_layers
        self.layer_options = layer_options


    def individual(self):
        return rnd.choice(self.layer_options, rnd.randint(1, self.max_layers))

    def mutation(self, solution):
        options = [self.mut2]
        if len(solution) > 1:
            options.append(self.mut3)
        if len(solution) < self.max_layers:
            options.append(self.mut1)
        selected_mutation = rnd.choice(options, 1)
        idx = rnd.randint(len(solution))
        selected_mutation(solution, idx)

    def mut1(self, solution, idx):
        solution = np.insert(solution, idx, rnd.choice(self.layer_options, 1))

    def mut2(self, solution, idx):
        solution[idx] = rnd.choice(self.layer_options, 1)

    def mut3(self, solution, idx):
        solution = np.delete(solution, idx)

    def crossover(self, sol1, sol2):
        l1 = len(sol1)
        l2 = len(sol2)

        if l1 < l2:
            l1, l2 = l2, l1
            sol1, sol2 = sol2, sol1
        c1 = np.zeros(l1)
        c2 = np.zeros(l2)


if __name__ == "__main__":
    alg