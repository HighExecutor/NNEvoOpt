import gym
from problems.rl_problem import RLProblem

class CartPoleProblem(RLProblem):
    def __init__(self):
        self.env_name = "CartPole-v0"
        self.state_size = 4
        self.action_size = 2
        self.max_steps = 200
        self.env = gym.make(self.env_name)