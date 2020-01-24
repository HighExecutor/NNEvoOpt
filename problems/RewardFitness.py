from deap import base


class RewardFitness(base.Fitness):
    def __init__(self):
        super().__init__(self)
        self.weights = (1.0,)