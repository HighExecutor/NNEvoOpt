from deap import tools, base
from multiprocessing import Pool
from algs.ga.ga_scheme import eaMuPlusLambda
from numpy import random as rnd
import numpy as np
from deap import creator
from algs.ga.simple_nnga import SimpleNNGA
from algs.ga.ga_scheme import eaMuPlusLambda

creator.create("BaseFitness", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.BaseFitness)


class SimpleNNGAExperiment:
    def __init__(self, problem):
        self.pop_size = 10
        self.iterations = 50
        self.mut_prob = 0.3
        self.cross_prob = 0.3

        self.problem = problem
        self.external_solution = None

        alg = SimpleNNGA(5, [16, 32, 64, 128, 256])

        self.engine = base.Toolbox()
        self.engine.register("map", map)

        self.engine.register("individual", tools.initIterate, creator.Individual, alg.individual)
        self.engine.register("population", tools.initRepeat, list, self.engine.individual, self.pop_size)
        self.engine.register("mate", alg.crossover)
        self.engine.register("mutate", alg.mutation)
        self.engine.register("select", tools.selRoulette)
        self.engine.register("evaluate", self.problem.evaluate)

    def run(self):
        pop = self.engine.population()

        def similar(x, y):
            if len(x) != len(y):
                return False
            return all(x==y)

        hof = tools.HallOfFame(1, similar)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = eaMuPlusLambda(pop, self.engine, self.pop_size, self.pop_size, cxpb=0.5, mutpb=0.2, ngen=self.iterations,
                                  stats=stats, halloffame=hof, verbose=True)
        print(log)
        print("Best = {}".format(hof[0]))

if __name__ == "__main__":
    from problems.cart_pole_problem import CartPoleProblem

    problem = CartPoleProblem()
    scenario = SimpleNNGAExperiment(problem)
    scenario.run()
