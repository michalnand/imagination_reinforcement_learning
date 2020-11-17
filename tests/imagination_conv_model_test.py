import sys
sys.path.insert(0, '..')

import gym
import numpy
import agents
import common.atari_wrapper
import common.super_mario_wrapper


from PIL import Image
import torch

import models.env_conv_model.src.model_env as IMModel

env = gym.make("MsPacmanNoFrameskip-v4")
env = common.atari_wrapper.AtariWrapper(env)

#env = gym.make("PongNoFrameskip-v4")
#env = common.atari_wrapper.AtariWrapper(env)

#env = gym.make("SuperMarioBros-v0")
#env = common.super_mario_wrapper.SuperMarioWrapper(env)


state = env.reset()

actions_count   = env.action_space.n

im_path = "models/env_conv_model/"
im =  agents.ImaginationModule(IMModel, state.shape, actions_count, learning_rate = 0.001, buffer_size = 4096)


for i in range(10000):
    action = numpy.random.randint(actions_count)
    state_, reward, done, _ = env.step(action)

    reward = numpy.clip(reward/10.0, -1.0, 1.0)

    im.add(state, action, reward)

    loss = im.train()
    
    if loss is not None:
        if i%100 == 0:
            print("iteration = ", i, " loss = ", loss)
            im.save(im_path)

    if done:
        state = env.reset()
    else:
        state = state_.copy()



'''
im.load(im_path)

for i in range(100):
    action = numpy.random.randint(actions_count)
    state_, _, done, _ = env.step(action)

    if done:
        state = env.reset()
    else:
        state = state_.copy()




trajectory_steps    = 8


reference_trajectory = []
predicted_trajectory = []

state_prediction_input = state.copy()

for i in range(trajectory_steps):
    action = numpy.random.randint(actions_count)
    state_, _, done, _ = env.step(action)

    state_prediction_output, _ = im.eval_np(state_prediction_input, action)

    reference_trajectory.append(state_[0].copy())
    predicted_trajectory.append(state_prediction_output[0].copy())

    state_prediction_input = state_prediction_output.copy()

    if done:
        state = env.reset()
    else:
        state = state_.copy()

def trajectory_to_image(trajectory, width = 256, height = 256, spacing = 4):
    trajectory_steps = len(trajectory)
    output_width   = trajectory_steps*(width + spacing)
    output_height  = height

    result = numpy.zeros((output_height, output_width))

    for i in range(trajectory_steps):
        image = Image.fromarray(trajectory[i]*255)
        image = image.resize((width, height), Image.ANTIALIAS)

        image_np = numpy.array(image)


        result[0:height,i*width + i*spacing:(i+1)*width + i*spacing] = image_np

    return result

image_reference = trajectory_to_image(reference_trajectory)
image_reference = Image.fromarray(image_reference)
image_reference.show()


image_prediction = trajectory_to_image(predicted_trajectory)
image_prediction = Image.fromarray(image_prediction)
image_prediction.show()

'''