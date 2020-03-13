import gym
import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Dense, SimpleRNN, LSTM
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import adam


def plot_history(history):
    plt.figure(1, figsize=(16, 5))
    plt.clf()
    plt.subplot(131)
    plt.plot(history['acc'])
    # plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(132)
    plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(133)
    plt.plot(history['reward'])
    plt.title('evaluation')
    plt.ylabel('evaluation')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')

    plt.tight_layout()
    plt.pause(0.1)
    pass


def plot_durations(rewards, means, save=False):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards)
    plt.plot(means)
    if save:
        plt.savefig("C:\\wspace\\data\\nn_tests\\{}.png".format("last_learn"))
    plt.pause(0.01)


class DQNAgent:
    def __init__(self, model, timestamps=1):
        self.memory_size = 500000
        self.memory = deque(maxlen=self.memory_size)
        self.gamma = 0.9  # future reward discount
        # Exploration parameters
        self.explore_start = 0.6  # exploration probability at start
        self.explore_stop = 0.001  # minimum exploration probability
        self.explore_steps = 500  # exponential decay rate for exploration prob
        self.explore_decay = (self.explore_start - self.explore_stop) / self.explore_steps
        self.explore = self.explore_start
        # Memory parameters
        self.batch_size = 1024
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

    def replay(self, epochs=1, verbose=0):
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
                target = reward_b + self.gamma * np.amax(target_Q) * (1 - done)
            targets[i] = self.model.predict(state_b)
            targets[i][action_b] = target
        history = self.model.fit(inputs, targets, epochs=epochs, verbose=verbose)
        return history

    def sample(self):
        idx = np.random.choice(np.arange(len(self.memory)),
                               size=self.batch_size,
                               replace=False)
        return [self.memory[ii] for ii in idx]

    def load_raw_data(self, data_path):
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

    def load_memory(self, datapath):
        file = open(datapath, "r")
        state_size = self.state_size
        timestamps = self.timestamps
        lines = file.readlines()
        file_timestamps = int(lines[0].split(" ")[1])
        assert file_timestamps == timestamps
        i = 1
        while i < len(lines):
            if "step" in lines[i]:
                i += 1
                continue
            if timestamps == 1:
                state = np.zeros((1, state_size))
                next_state = np.zeros((1, state_size))
            else:
                state = np.zeros((1, timestamps, state_size))
                next_state = np.zeros((1, timestamps, state_size))
            for j in range(timestamps):
                arr = np.array([float(v) for v in lines[i + j].split(",")])
                if timestamps == 1:
                    state[0] = arr
                else:
                    state[0][j] = arr
            i += timestamps
            action = int(lines[i])
            reward = float(lines[i + 1])
            i += 2
            for j in range(timestamps):
                arr = np.array([float(v) for v in lines[i + j].split(",")])
                if timestamps == 1:
                    next_state[0] = arr
                else:
                    next_state[0][j] = arr
            i += timestamps
            done = eval(lines[i])
            i += 1
            sars = (state, action, reward, next_state, done)
            self.remember(sars)
        file.close()

    def save_model(self, path):
        file_path = "C:\\wspace\\data\\nn_tests\\"
        print("Saving model")
        self.model.save_weights("{}{}.h5".format(file_path, path), overwrite=True)
        json_model = self.model.to_json()
        with open("{}{}.json".format(file_path, path), "w") as outfile:
            outfile.write(json_model)
        print("Model saved: {}{}".format(file_path, path))
        return json_model

    def save_memory(self, filename):
        timestamps = self.timestamps
        states_file = open("C:\\wspace\\data\\nn_tests\\{}.mem".format(filename), "w")
        states_file.write("timestamps {}\n".format(timestamps))

        for idx in range(len(self.memory)):
            if idx % 100 == 0:
                print("step {}\n".format(idx))
            sars = self.memory[idx]
            state = sars[0][0]
            action = sars[1]
            reward = sars[2]
            next_state = sars[3][0]
            done = sars[4]

            states_file.write("sars {}\n".format(idx))

            for t in range(timestamps):
                if timestamps == 1:
                    arr = state
                else:
                    arr = state[t]
                arr_str = ""
                for j in range(len(arr) - 1):
                    arr_str += "{},".format(arr[j])
                arr_str += "{}".format(arr[-1])
                states_file.write("{}\n".format(arr_str))
            states_file.write("{}\n".format(action))
            states_file.write("{}\n".format(reward))
            for t in range(timestamps):
                if timestamps == 1:
                    arr = next_state
                else:
                    arr = next_state[t]
                arr_str = ""
                for j in range(len(arr) - 1):
                    arr_str += "{},".format(arr[j])
                arr_str += "{}".format(arr[-1])
                states_file.write("{}\n".format(arr_str))
            states_file.write("{}\n".format(done))
        states_file.close()


def learning_episodes(env, agent, n=100, timestamps=1):
    rewards = list()
    means = list()

    plt.ion()
    for ep in range(n):
        print("Ep {}".format(ep))
        reward = episode(env, agent, timestamps)
        rewards.append(reward)
        agent.replay()
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
    plot_durations(rewards, means, save=True)
    plt.show()


def episode(env, agent, timestamps=1, render=True, remember=True):
    state_size = agent.state_size
    state = deque(maxlen=timestamps)
    for _ in range(timestamps - 1):
        state.append(np.zeros(state_size))
    state.append(env.reset())
    env_name = env.spec._env_name

    total_reward = 0
    t = 0
    max_steps = 1000
    penalty_steps = 500
    while t < max_steps:
        t += 1
        if render:
            env.render()

        if env_name == "LunarLander" and timestamps == 1 and state[0][6] + state[0][7] == 2.0:
            print("trying to calm")
            action = 0
        else:
            action = agent.act(state)
        # Take action, get new state and reward
        new_state, reward, done, _ = env.step(action)
        if env_name == "LunarLander" and t > penalty_steps and action > 0:
            reward -= (t / penalty_steps)
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

        if done:
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


def build_model(input, output):
    model = Sequential()
    model.add(Dense(150, input_dim=input, activation='relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(output, activation='linear'))
    model.compile(loss="mse", optimizer=adam(lr=0.001), metrics=['accuracy'])
    print(model.summary())
    return model


def build_rnn_model(input, output):
    model = Sequential()
    model.add(SimpleRNN(128, input_dim=input, activation='relu'))
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
    dataset = None

    timestamps = 1
    if timestamps == 1:
        model = build_model(input, output)
        # model_name = "lunarlander_random"
        # model = load_model(model_name)
    else:
        model = build_rnn_model(input, output)
        # model_name = "lunarlander_rnn_random"
        # model = load_model(model_name)
    dataset = "C:\\wspace\\data\\nn_tests\\lunarlander\\lunarlander.mem"

    agent = DQNAgent(model, timestamps=timestamps)
    agent.load_memory(dataset)
    agent.replay(epochs=5)
    agent.replay(epochs=5)
    agent.replay(epochs=5)
    agent.replay(epochs=5)
    learning_episodes(env, agent, timestamps=timestamps, n=2000)
    # train_from_memory(env, model, datapath=dataset, timestamps=timestamps, n=1000)
