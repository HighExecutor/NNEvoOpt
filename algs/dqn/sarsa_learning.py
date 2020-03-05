import gym
import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy
from keras.layers import Dense, SimpleRNN, LSTM
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import adam


def plot_durations(rewards, means):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards)
    plt.plot(means)
    plt.pause(0.01)


class DQNAgent:
    def __init__(self, model, timestamps=1):
        self.memory_size = 10000
        self.memory = deque(maxlen=self.memory_size)
        self.gamma = 0.9  # future reward discount
        # Exploration parameters
        self.explore_start = 0.6  # exploration probability at start
        self.explore_stop = 0.001  # minimum exploration probability
        self.explore_steps = 300  # exponential decay rate for exploration prob
        self.explore_decay = (self.explore_start - self.explore_stop) / self.explore_steps
        self.explore = self.explore_start
        # Memory parameters
        self.batch_size = 256
        self.model = model
        self.action_size = list(model.output_shape)[-1]
        self.state_size = list(model.input_shape)[-1]
        self.timestamps = timestamps

    def act(self, state):
        if rnd.rand() < self.explore:
            return rnd.randint(self.action_size)
        else:
            Qs = None
            if self.timestamps == 1:
                Qs = self.model.predict(np.array(state))[0]
            else:
                Qs = self.model.predict(np.array(state).reshape(1, self.timestamps, self.state_size))[0]
            action = np.argmax(Qs)
            return action

    def converge_explore(self):
        if self.explore > self.explore_stop:
            self.explore -= self.explore_decay

    def remember(self, sars):
        self.memory.append(sars)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        # Replay
        if self.timestamps == 1:
            inputs = np.zeros((self.batch_size, self.state_size))
        else:
            inputs = np.zeros((self.batch_size, self.timestamps, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))
        minibatch = self.sample()
        for i, (state_b, action_b, reward_b, next_state_b, done) in enumerate(minibatch):
            inputs[i:i + 1] = state_b
            target = reward_b
            if not (next_state_b == np.zeros(state_b.shape)).all():
                target_Q = self.model.predict(next_state_b)[0]
                target = reward_b + self.gamma * np.amax(target_Q) * (1-done)
            targets[i] = self.model.predict(state_b)
            targets[i][action_b] = target
        self.model.fit(inputs, targets, epochs=1, verbose=0)
        self.converge_explore()

    def sample(self):
        idx = np.random.choice(np.arange(len(self.memory)),
                               size=self.batch_size,
                               replace=False)
        return [self.memory[ii] for ii in idx]

    def load_data(self, data_path):
        timestamps = self.timestamps
        state_size = self.state_size
        action_size = self.action_size
        data = pd.read_csv(data_path, header=None, skiprows=1,
                           names=np.arange(0, 1 + state_size + action_size, 1)).values
        ep_ids = data[:, 0]  # id of episodes
        states = data[:, 1:1 + state_size]  # x_data
        actions = data[:, state_size + 1:state_size + action_size + 1]  # t_data
        data_size = len(states)
        del data

        if timestamps > 1:
            states_shaped = np.zeros(shape=(data_size, timestamps, state_size))
            cur_ep_id = -1
            state = None
            for i in range(len(ep_ids)):
                ep_id = ep_ids[i]
                if ep_id != cur_ep_id:
                    state = deque(maxlen=timestamps)
                    for _ in range(timestamps - 1):
                        state.append(np.zeros(state_size))
                    cur_ep_id = ep_id
                state.append(states[i])
                states_shaped[i] = np.array(state)
            del states
            states = states_shaped
            del states_shaped

        sarsa = list()
        sample_ids = rnd.choice(np.arange(len(ep_ids) - 1), min(self.memory_size, len(ep_ids) - 1))

        for idx in sample_ids:
            state = deque(maxlen=1)
            cur_ep_id = ep_ids[idx]

            state.append(states[idx])
            act = np.argmax(actions[idx])
            rwd = actions[idx][act]
            cur_state = np.array(state)
            if cur_ep_id != ep_ids[idx + 1]:
                sarsa.append((cur_state, act, rwd, np.zeros(shape=cur_state.shape), True))
                continue
            state.append(states[idx + 1])
            next_state = np.array(state)
            sarsa.append((cur_state, act, rwd, next_state, False))
        for s in sarsa:
            self.remember(s)

        print("data loaded")


def learning_episodes(env, model, dataset=None, n=100, timestamps=1):
    agent = DQNAgent(model, timestamps)
    if dataset is not None:
        agent.load_data(dataset)
    rewards = list()
    means = list()

    plt.ion()
    for ep in range(n):
        reward = episode(env, agent, timestamps)
        rewards.append(reward)
        agent.replay()
        # Take 100 episode averages and plot them too
        if len(rewards) > 100:
            last_mean = np.mean(rewards[-100:])
            means.append(last_mean)
        else:
            means.append(np.mean(rewards))
        plot_durations(rewards, means)
    plt.ioff()
    plot_durations(rewards, means)
    plt.show()


def episode(env, agent, timestamps=1, render=True):
    state_size = agent.state_size
    state = deque(maxlen=timestamps)
    for _ in range(timestamps - 1):
        state.append(np.zeros(state_size))
    state.append(env.reset())

    total_reward = 0
    t = 0
    max_steps = 2000
    while t < max_steps:
        t += 1
        if render:
            env.render()

        action = agent.act(state)
        # Take action, get new state and reward
        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        cur_state = np.array(state)
        state.append(new_state)
        if timestamps == 1:
            agent.remember((cur_state, action, reward, np.array(state), done))
        else:
            agent.remember((cur_state.reshape((1, timestamps, state_size)), action, reward, np.array(state).reshape((1, timestamps, state_size)), done))

        if done:
            # print('Total reward: {}'.format(total_reward))
            return total_reward
    return total_reward


def build_model(input, output):
    model = Sequential()
    model.add(Dense(150, input_dim=input, activation='relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(output, activation='linear'))
    model.compile(loss="mse", optimizer=adam(), metrics=['accuracy'])
    print(model.summary())
    return model


def build_rnn_model(input, output):
    model = Sequential()
    model.add(SimpleRNN(64, input_dim=input, activation='relu'))
    model.add(Dense(output, activation='linear'))
    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    print(model.summary())
    return model


def load_model(name):
    file_path = "C:\\wspace\\data\\nn_tests\\"
    json_file = open("{}{}.json".format(file_path, name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    # load weights into new model
    loaded_model.load_weights("{}{}.h5".format(file_path, name))
    print("Loaded model from disk")
    return loaded_model


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v0")
    input = 8
    output = 4

    timestamps = 1
    if timestamps == 1:
        # model = build_model(input, output)
        model_name = "lunarlander_random"
        model = load_model(model_name)
    else:
        # model = build_rnn_model(input, output)
        model_name = "lunarlander_rnn_random"
        model = load_model(model_name)
    # dataset = "C:\\wspace\\data\\nn_tests\\cartpole_sarsa_g9.csv"
    dataset = "C:\\wspace\\data\\nn_tests\\lunarlander_sarsa_g9.csv"
    # dataset = None
    learning_episodes(env, model, timestamps=timestamps, dataset=dataset, n=10000)
