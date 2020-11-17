import sys
sys.path.insert(0, '..')

import gym
import numpy
import agents

import torch

import models.env_fc_model.src.model_env as IMModel

env = gym.make("LunarLander-v2")

state = env.reset()

actions_count   = env.action_space.n

im =  agents.ImaginationModule(IMModel, state.shape, actions_count, learning_rate = 0.001, buffer_size = 4096)


for i in range(100000):
    action = numpy.random.randint(actions_count)
    state_, reward, done, _ = env.step(action)

    reward = numpy.clip(reward/10.0, -1.0, 1.0)


    im.add(state, action, reward)

    loss = im.train()
    
    if loss is not None:
        if i%100 == 0:
            print("iteration = ", i, " loss = ", loss)

            state_prediction, reward_prediction = im.eval_np(state, action)

            print(numpy.round(state, 3))
            print(numpy.round(state_, 3))
            print(numpy.round(state_prediction, 3))
            print(reward, reward_prediction)

            print("\n\n\n")

    if done:
        state = env.reset()
    else:
        state = state_.copy()