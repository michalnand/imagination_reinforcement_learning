import sys
sys.path.insert(0, '..')
import gym
import gym_super_mario_bros
import numpy
import time

from PIL import Image


import agents

import common.common_frame_wrapper




env = gym.make("CarRacing-v0")
env = common.common_frame_wrapper.CommonFrameWrapper(env, height=64, width=64, frame_stacking=1, frame_skipping=4)
obs = env.reset()

env.render(close=True)

agent = agents.AgentRandomContinuous(env)



print(obs)
print(obs.shape)

im = Image.fromarray(obs[0]*255)
im.show()

k = 0.02
fps = 0
while True:
    time_start = time.time()
    reward, done = agent.main()
    time_stop  = time.time()
    #env.render()

    fps = (1.0-k)*fps + k*1.0/(time_stop - time_start)


    if reward != 0:
        print("reward = ", reward)
    
    if done:
        print("FPS = ", round(fps, 1))
        print("DONE \n\n")
    
    time.sleep(0.01)
