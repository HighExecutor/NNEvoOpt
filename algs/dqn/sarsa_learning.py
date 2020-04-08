import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from visualization.sarsa_plots import plot_durations, plot_history
from problems.cartpole_problem import CartPoleProblem
from algs.dqn.sarsa_actor import DQNAgent


def learning_episodes(env, agent, n=100):
    rewards = list()
    means = list()
    plt.ion()
    for ep in range(n):
        print("Ep {}, explore = {}".format(ep, agent.explore))
        reward = episode(env, agent)
        rewards.append(reward)
        # agent.replay(epochs=1)
        agent.converge_explore()
        # Take 100 episode averages and plot them too
        if len(rewards) > 100:
            last_mean = np.mean(rewards[-100:])
            means.append(last_mean)
        else:
            means.append(np.mean(rewards))
        plot_durations(rewards, means)
    agent.save_model("last_model")
    agent.save_memory("last_memory")
    plt.ioff()
    plot_durations(rewards, means, save=False)
    plt.show()


def episode(problem, agent, render=True, remember=True):
    state_size = problem.state_size
    timestamps = agent.timestamps
    state = deque(maxlen=timestamps)
    for _ in range(timestamps - 1):
        state.append(np.zeros(state_size))
    # env_state = np.append(env.reset(), 1.0)
    state.append(problem.reset())
    env_name = problem.env_name

    total_reward = 0
    t = 0
    max_steps = problem.max_steps
    # penalty_steps = 500
    # gaz = 1.0
    while t < max_steps:
        t += 1
        if render:
            problem.env.render()

        if env_name == "LunarLander" and timestamps == 1 and state[0][6] + state[0][7] == 2.0:
            print("trying to calm")
            action = 0
        else:
            action = agent.act(state)
        # Take action, get new state and reward
        new_state, reward, done, _ = problem.step(action)
        # if env_name == "LunarLander" and t > penalty_steps and action > 0:
        #     reward -= (t / penalty_steps)
        # print("step = {}; Reward = {}, done = {}".format(t, reward, done))
        total_reward += reward
        cur_state = np.array(state)
        state.append(new_state)
        if remember:
            if timestamps == 1:
                agent.remember((cur_state, action, reward, np.array(state), done))
            else:
                agent.remember((cur_state.reshape((1, timestamps, state_size)), action, reward,
                                np.array(state).reshape((1, timestamps, state_size)), done))

        agent.replay(epochs=1)
        # agent.converge_explore()

        if done:
            print("steps = {}; Reward = {}, done = {}".format(t, total_reward, done))
            return total_reward


def train_from_memory(env, model, datapath, timestamps=1, n=100):
    agent = DQNAgent(model, timestamps)
    agent.load_memory(datapath)
    agent.explore_start = 0.0
    agent.explore_stop = 0.0
    agent.batch_size = 2048

    history = dict()
    history['loss'] = list()
    history['acc'] = list()
    history['reward'] = list()
    plt.ion()
    for i in range(n):
        e_history = agent.replay(verbose=1, epochs=5)
        reward = episode(env, agent, timestamps, render=True, remember=False)
        history['loss'].append(e_history.history['loss'][0])
        history['acc'].append(e_history.history['acc'][0])
        history['reward'].append(reward)
        plot_history(history)
    plt.ioff()
    plot_history(history)
    plt.show()


if __name__ == "__main__":
    problem = CartPoleProblem()
    timestamps = 1
    layers = [20, 10]
    agent = DQNAgent()
    agent.build_model(problem.state_size, problem.action_size, layers, timestamps)
    learning_episodes(problem, agent, n=1000)
