import gym

class RLProblem:
    def __init__(self):
        self.env_name = None
        self.state_size = None
        self.action_size = None
        self.env = None
        self.max_steps = None

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()