#
# https://medium.com/@yashpatel_86510/reinforcement-learning-w-keras-openai-698add10b4eb
#

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


def gather_data(env):

    num_trials = 10000
    sim_steps = 1000
    min_score = 100

    trainingX, trainingY = [], []
    scores = []

    for _ in range(num_trials):
        observation = env.reset()

        score = 0
        training_sampleX, training_sampleY = [], []
        for step in range(sim_steps):
            action = np.random.randint(0, 2)
            one_hot_action = np.zeros(2)
            one_hot_action[action] = 1

            training_sampleX.append(observation)
            training_sampleY.append(one_hot_action)

            # next
            observation, reward, done, info = env.step(action)
            score = score + reward

            if done:
                break

        if score > min_score:
            scores.append(score)
            trainingX += training_sampleX
            trainingY += training_sampleY

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    return trainingX, trainingY
