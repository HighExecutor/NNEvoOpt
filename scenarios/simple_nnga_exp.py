from deap import tools, base
from multiprocessing import Pool
from algs.ga.ga_scheme import eaMuPlusLambda
from numpy import random as rnd
import numpy as np
from deap import creator

class SimpleNNGAExperiment:
    def __init__(self, problem):
        self.pop_size = 10
        self.iterations = 50
        self.mut_prob = 0.3
        self.cross_prob = 0.3

        self.problem = problem
        self.external_solution = None

        engine = base.Toolbox()
        engine.register("map", map)

        engine.register("individual", tools.initIterate, creator.TreeIndividual, SimpleNNGAGenerator)
        engine.register("population", tools.initRepeat, list, engine.individual, self.pop_size)


if __name__ == "__main__":
    scenario = SimpleNNGAExperiment()