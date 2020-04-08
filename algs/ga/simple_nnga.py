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
        selected_mutation = rnd.choice(options, 1)[0]
        idx = rnd.randint(len(solution))
        solution = selected_mutation(solution, idx)
        return solution,

    def mut1(self, solution, idx):
        solution = np.insert(solution, idx, rnd.choice(self.layer_options, 1))
        return solution

    def mut2(self, solution, idx):
        solution[idx] = rnd.choice(self.layer_options, 1)
        return solution

    def mut3(self, solution, idx):
        solution = np.delete(solution, idx)
        return solution

    def crossover(self, sol1, sol2):
        l1 = len(sol1)
        l2 = len(sol2)

        if l1 < l2:
            l1, l2 = l2, l1
            sol1, sol2 = sol2, sol1
        c1 = np.zeros(l1, dtype=sol1.dtype)
        c2 = np.zeros(l2, dtype=sol2.dtype)
        idx = rnd.randint(l2)
        for i in range(idx):
            c1[i] = sol1[i]
            c2[i] = sol2[i]
        for i in range(idx, l2):
            c1[i] = sol2[i]
            c2[i] = sol1[i]
        for i in range(l2, l1):
            c1[i] = sol1[i]

        c1 = creator.Individual(c1)
        c2 = creator.Individual(c2)
        return c1, c2


if __name__ == "__main__":
    alg = SimpleNNGA(5, [16, 32, 64, 128, 256])
    s1 = alg.individual()
    s2 = alg.individual()
    s3 = alg.individual()

    c1, c2 = alg.crossover(s1, s2)
    from problems.cartpole_problem import CartPoleProblem

    problem = CartPoleProblem()
    print("s1")
    problem.evaluate(s1)
    print("s2")
    problem.evaluate(s2)
    print("s3")
    problem.evaluate(s3)
    alg.mutation(s3)
    print("mutated s3")
    problem.evaluate(s3)
    print("c1")
    problem.evaluate(c1)
    print("c2")
    problem.evaluate(c2)
