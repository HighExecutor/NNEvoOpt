import numpy.random as rnd
import numpy as np
from copy import deepcopy
import gym

def lunar_lander_random_episode(env):
    state_list = list()
    reward_list = list()
    action_list = list()

    max_env_steps = 200
    action_size = 4
    state = np.array(env.reset())
    for act_time in range(max_env_steps):
        possible_actions = np.arange(action_size)
        rewards = np.zeros(action_size)
        for act_idx in possible_actions:
            env_copy = deepcopy(env)
            _, rwd, _, _ = env_copy.step(act_idx)
            rewards[act_idx] = rwd

        state_list.append(state)
        reward_list.append(rewards)

        action = rnd.choice(possible_actions)
        action_list.append(action)

        next_state, rwd, done, _ = env.step(action)
        state = next_state
        if done:
            env.close()
            break
    return state_list, action_list, reward_list


def make_dataset():
    data_name = "lunarlander_sarsa_g9"
    states_file = open("C:\\wspace\\data\\nn_tests\\{}.csv".format(data_name), "w")

    gamma = 0.9
    episodes = 1000
    start_idx = 0

    env = gym.make('LunarLander-v2')
    for idx in range(episodes):
        if idx % 100 == 0:
            print("Episode {}".format(idx))
        states, actions, rewards = lunar_lander_random_episode(env)

        for i in range(len(states)):
            states[i] = states[i].astype(np.float32)
            rewards[i] = rewards[i].astype(np.float32)

        for i in range(len(states) - 1):
            j = len(states) - i - 2
            rewards[j][actions[j]] += gamma * np.amax(rewards[j+1])

        for i in range(len(states)):
            result_list = list()
            result_list.append(idx+start_idx)
            result_list.extend(states[i])
            result_list.extend(rewards[i])

            result_str = ""
            for elem in range(len(result_list) -1):
                result_str += "{},".format(result_list[elem])
            result_str += str(result_list[-1])
            states_file.write(result_str)
            states_file.write("\n")

    states_file.close()



if __name__ == "__main__":
    make_dataset()