from collections import deque
import numpy as np
import numpy.random as rnd
from keras.models import model_from_json
import pandas as pd
from keras.layers import Dense, SimpleRNN, LSTM
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import adam
import random


class DQNAgent:
    def __init__(self):
        self.memory_size = 50000
        self.memory = deque(maxlen=self.memory_size)
        self.gamma = 0.99  # future reward discount
        # Exploration parameters
        self.explore_start = 1.0  # exploration probability at start
        self.explore_stop = 0.01  # minimum exploration probability
        self.explore_steps = 100  # exponential decay rate for exploration prob
        self.explore_decay = (self.explore_start - self.explore_stop) / self.explore_steps
        self.explore = self.explore_start
        # Memory parameters
        self.batch_size = 64
        self.model = None

    def build_model(self, state_size=4, action_size=2, layers=[150, 120], timestamps=1):
        self.timestamps = timestamps
        self.state_size = state_size
        self.action_size = action_size

        model = Sequential()
        if timestamps == 1:
            model.add(Dense(layers[0], input_dim=state_size, activation='relu'))
        else:
            model.add(SimpleRNN(layers[0], input_dim=(timestamps, state_size), activation='relu'))
        for i in range(1, len(layers)):
            if timestamps == 1:
                model.add(Dense(layers[i], activation='relu'))
            else:
                model.add(SimpleRNN(layers[i], activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss="mse", optimizer=adam(lr=0.001), metrics=['accuracy'])
        print(model.summary())
        self.model = model

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
            #     self.explore *= 0.996
            self.explore -= self.explore_decay

    def remember(self, sars):
        self.memory.append(sars)

    def replay(self, epochs=1, verbose=0):
        if len(self.memory) < self.batch_size:
            return
        minibatch = self.sample()
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        history = self.model.fit(states, targets_full, epochs=epochs, verbose=verbose)
        return history

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def load_memory(self, datapath):
        file = open(datapath, "r")
        state_size = self.state_size
        timestamps = self.timestamps
        lines = file.readlines()
        file_timestamps = int(lines[0].split(" ")[1])
        assert file_timestamps == timestamps
        i = 1
        while i < len(lines):
            if "sars" in lines[i]:
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

    def load_model(name, file_path):
        json_file = open("{}{}.json".format(file_path, name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
        # load weights into new model
        loaded_model.load_weights("{}{}.h5".format(file_path, name))
        print("Loaded model from disk")
        return loaded_model

    def save_model(self, file_path):
        print("Saving model")
        self.model.save_weights("{}.h5".format(file_path), overwrite=True)
        json_model = self.model.to_json()
        with open("{}.json".format(file_path), "w") as outfile:
            outfile.write(json_model)
        print("Model saved: {}".format(file_path))
        return json_model

    def save_memory(self, filename):
        timestamps = self.timestamps
        states_file = open("{}.mem".format(filename), "w")
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
