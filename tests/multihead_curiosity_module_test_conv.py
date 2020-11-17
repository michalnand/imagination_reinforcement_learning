import sys
sys.path.insert(0, '..')

import gym
import numpy
import agents

import torch

from common.atari_wrapper import *

import models.env_conv_multihead_model.src.model_env as Model

path = "models/pacam_dqn/"

env = gym.make("MsPacmanNoFrameskip-v4")
env = AtariWrapper(env)


state = env.reset()

actions_count   = env.action_space.n

cm =  agents.CuriosityModule(Model, state.shape, actions_count, learning_rate = 0.001, buffer_size = 4096)


loss_filtered = 0

for i in range(100000):
    action = numpy.random.randint(actions_count)
    state_, reward, done, _ = env.step(action)

    reward = numpy.clip(reward/10.0, -1.0, 1.0)


    cm.add(state, action, reward, done)

    loss = cm.train()

    if loss is not None:
        loss_filtered = 0.99*loss_filtered + 0.01*loss

    if i%100 == 0 and loss is not None:
        print("iteration = ", i, " loss = ", loss_filtered, state.shape)

       
        curiosity, reward_prediction = cm.eval_np(state, state_, action)

        print(curiosity)
        print(reward, reward_prediction)

        print("\n\n\n")

    if done:
        state = env.reset()
    else:
        state = state_.copy()