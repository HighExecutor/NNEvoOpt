from deap import base


class CartPoleProblem:
    def __init__(self):
        self.name = "cartpole"
        self.input_dim = 4


    def evaluate(self, solution):
        result = 0.0
        rwd = self.launch(solution)
        result += rwd
        return result

    def launch(self, solution):
        agent = None
        return 0.0

