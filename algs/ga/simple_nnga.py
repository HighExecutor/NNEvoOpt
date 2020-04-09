import numpy as np
import numpy.random as rnd
from deap import creator
from multiprocessing import Pool
from deap import tools, base
from algs.ga.ga_scheme import eaMuPlusLambda
from algs.dqn.sarsa_actor import DQNAgent
from algs.dqn.sarsa_learning import train_from_memory

creator.create("BaseFitness", base.Fitness, weights=(1.0, ))
creator.create("Individual", np.ndarray, fitness=creator.BaseFitness)


class SimpleNNGA:
    def individual(self):
        return list(rnd.randint(low=self.min_layer_size, high=self.max_layer_size, size=rnd.randint(1, self.max_layers)))

    def mutation(self, solution):
        options = [self.mut2]
        if len(solution) > 1:
            options.append(self.mut3)
        if len(solution) < self.max_layers:
            options.append(self.mut1)
        selected_mutation = rnd.choice(options, 1)[0]
        idx = rnd.randint(len(solution))
        solution = selected_mutation(solution, idx)
        return creator.Individual(list(solution)),

    def mut1(self, solution, idx):
        solution = np.insert(solution, idx, rnd.randint(self.min_layer_size, self.max_layer_size))
        return solution

    def mut2(self, solution, idx):
        solution[idx] = np.clip(solution[idx] + rnd.randint(-32, 32), self.min_layer_size, self.max_layer_size)
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

    def evaluate(self, solution):
        agent = DQNAgent()
        agent.build_model(self.problem.state_size, self.problem.action_size, solution, 1)
        history = train_from_memory(problem, agent, datapath=self.memory_path, n=10, plots=0)
        acc = np.mean(history['acc'][-5:])
        loss = np.mean(history['loss'][-5:])
        reward = np.mean(history['reward'][-5:])
        return acc,

    def __init__(self, max_layers, max_layer_size, min_layer_size, problem, memory_path):
        self.max_layers = max_layers
        self.max_layer_size = max_layer_size
        self.min_layer_size = min_layer_size
        self.memory_path = memory_path

        self.pop_size = 5
        self.iterations = 10
        self.mut_prob = 0.4
        self.cross_prob = 0.4

        self.problem = problem
        self.external_solution = None

        self.engine = base.Toolbox()
        # self.pool = Pool(5)
        # self.engine.register("map", self.pool.map)
        self.engine.register("map", map)

        self.engine.register("individual", tools.initIterate, creator.Individual, self.individual)
        self.engine.register("population", tools.initRepeat, list, self.engine.individual, self.pop_size)
        self.engine.register("mate", self.crossover)
        self.engine.register("mutate", self.mutation)
        self.engine.register("select", tools.selRoulette)
        self.engine.register("evaluate", self.evaluate)

    def run(self):
        pop = self.engine.population()

        def similar(x, y):
            if len(x) != len(y):
                return False
            return all(x == y)

        hof = tools.HallOfFame(1, similar)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = eaMuPlusLambda(pop, self.engine, self.pop_size, self.pop_size, cxpb=self.cross_prob,
                                  mutpb=self.mut_prob, ngen=self.iterations,
                                  stats=stats, halloffame=hof, verbose=True)
        print(log)
        print("Best = {}".format(hof[0]))
        return pop, log


if __name__ == "__main__":
    from problems.cartpole_problem import CartPoleProblem

    problem = CartPoleProblem()
    memory_path = "D:\\data\\cartpole\\last_memory.mem"
    alg = SimpleNNGA(5, 128, 8, problem, memory_path)
    pop, log = alg.run()
