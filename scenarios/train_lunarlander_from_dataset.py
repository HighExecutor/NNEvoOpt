import numpy as np
import numpy.random as rnd
from keras import Sequential
from keras.layers import SimpleRNN, Dense, LSTM
from keras.losses import hinge
import json
from keras.models import model_from_json
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gym
from problems.lunar_lander_launcher import LunarLanderLauncher
from collections import deque


def build_model():
    model = Sequential()
    model.add(Dense(150, input_dim=8, activation='relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(4, activation='linear'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model_name = "lunarlander_random"
    print(model.summary())
    return model, model_name


def build_rnn_model():
    model = Sequential()
    model.add(SimpleRNN(20, input_dim=8, activation='relu'))
    model.add(Dense(4, activation='linear'))
    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    model_name = "lunarlander_rnn_random"
    print(model.summary())
    return model, model_name


def save_model(model, name):
    file_path = "C:\\wspace\\data\\nn_tests\\"
    print("Saving model")
    model.save_weights("{}{}.h5".format(file_path, name), overwrite=True)
    json_model = model.to_json()
    with open("{}{}.json".format(file_path, name), "w") as outfile:
        outfile.write(json_model)
    print("Model saved: {}{}".format(file_path, name))
    return json_model


def load_model(name):
    file_path = "C:\\wspace\\data\\nn_tests\\"
    json_file = open("{}{}.json".format(file_path, name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("{}{}.h5".format(file_path, name))
    print("Loaded model from disk")
    return loaded_model


def plot_history(history):
    plt.figure(1, figsize=(16, 5))
    plt.clf()
    plt.subplot(131)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(132)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(133)
    plt.plot(history['reward'])
    plt.title('evaluation')
    plt.ylabel('evaluation')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.tight_layout()
    plt.pause(0.1)
    pass


def learn_model(data_path, timestamps=1):
    if timestamps == 1:
        model, model_name = build_model()
    else:
        model, model_name = build_rnn_model()

    input_shape = list(model.input_shape)[1:]
    state_size = input_shape[-1]
    action_size = list(model.output_shape)[-1]

    data = pd.read_csv(data_path, header=None, skiprows=1, names=np.arange(0, 13, 1)).values
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
            pass
        x_train, x_test, y_train, y_test = train_test_split(states_shaped, actions, test_size=0.33)
        del states_shaped
    else:
        x_train, x_test, y_train, y_test = train_test_split(states, actions, test_size=0.33)
    del states
    del actions

    plt.ion()
    epochs = 300
    history = dict()
    history['val_loss'] = list()
    history['loss'] = list()
    history['acc'] = list()
    history['val_acc'] = list()
    history['reward'] = list()
    for e in range(epochs):
        e_history = model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_test, y_test), verbose=1)
        reward = test_episode_lunarlander_model(model, timestamps)
        history['val_loss'].append(e_history.history['val_loss'][0])
        history['loss'].append(e_history.history['loss'][0])
        history['acc'].append(e_history.history['acc'][0])
        history['val_acc'].append(e_history.history['val_acc'][0])
        history['reward'].append(reward)
        plot_history(history)

    save_model(model, model_name)

    plt.ioff()
    plot_history(history)
    plt.show()
    pass


lunarlander = LunarLanderLauncher()
def test_lunarlander_model(model_path, timestamps=1, model=None):
    if model is None:
        model = load_model(model_path)
    lunarlander.episodes(model, 20, timestamps)


def test_episode_lunarlander_model(model, timestamps=1):
    return lunarlander.episode(model, timestamps, render=False)


def model_evaluation():
    model = load_model("test_model")
    print(model.summary())


if __name__ == "__main__":
    data_path = "C:\\wspace\\data\\nn_tests\\lunarlander_sarsa_g9.csv"

    timestamps = 1
    if timestamps == 1:
        model_path = "lunarlander_random"
    else:
        model_path = "lunarlander_rnn_random"
    learn_model(data_path, timestamps)
    test_lunarlander_model(model_path, timestamps)
