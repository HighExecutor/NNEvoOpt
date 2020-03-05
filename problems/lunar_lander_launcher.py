import gym
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

def plot_durations(rewards, means):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards)
    plt.plot(means)
    plt.pause(0.01)

class LunarLanderLauncher:

    def __init__(self):
        self.env = gym.make('LunarLander-v2')


    def episodes(self, model, n, timestamps=1):
        plt.ion()
        rewards = list()
        means = list()
        for i in range(n):
            reward = self.episode(model, timestamps)
            rewards.append(reward)
            # Take 100 episode averages and plot them too
            if len(rewards) > 10:
                last_mean = np.mean(rewards[-100:])
                means.append(last_mean)
            else:
                means.append(0.0)
            plot_durations(rewards, means)
        plt.ioff()
        plot_durations(rewards, means)
        plt.show()



    def episode(self, model, timestamps=1, render=True):
        total_reward = 0
        t = 0
        max_steps = 1000
        input_shape = list(model.input_shape)[1:]
        statesize = input_shape[-1]
        # if len(input_shape) > 1:
        #     timestamps = input_shape[0]
        state = deque(maxlen=timestamps)
        for _ in range(timestamps-1):
            state.append(np.zeros(statesize))
        state.append(self.env.reset())

        while t < max_steps:
            t += 1
            if render:
                self.env.render()
            Qs = None
            if timestamps == 1:
                Qs = model.predict(np.array(state))[0]
            else:
                Qs = model.predict(np.array(state).reshape(1, timestamps, statesize))[0]
            action = np.argmax(Qs)
            # Take action, get new state and reward
            new_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state.append(new_state)

            if done:
                # print('Total reward: {}'.format(total_reward))
                return total_reward
