import sys
sys.path.insert(0, '..')
import gym
import numpy
import time

import libs_agents

import libs_common.atari_wrapper

from PIL import Image



#env = gym.make("BreakoutNoFrameskip-v4")
#env = gym.make("MsPacmanNoFrameskip-v4")
env = gym.make("MontezumaRevengeNoFrameskip-v4")
env = libs_common.atari_wrapper.AtariWrapper(env)
env.reset()

agent = libs_agents.AgentRandom(env)


obs, _, _, _ = env.step(0)


print(obs)
print(obs.shape)


im = Image.fromarray(obs[0]*255.0)
im.show()

k = 0.02
fps = 0
while True:
    time_start = time.time()
    reward, done = agent.main()
    time_stop  = time.time()
    env.render()

    fps = (1.0-k)*fps + k*1.0/(time_stop - time_start)


    if reward != 0:
        print("reward = ", reward)
    
    if done:
        print("FPS = ", round(fps, 1))
        print("DONE \n\n")
    
    time.sleep(0.01)
