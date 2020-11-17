import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import gym_line_follower 

import numpy
import time


import agents


class Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        
    def observation(self, obs):
        return numpy.array(obs).astype(numpy.float32)

env = gym.make("LineFollower-v0", gui = False)
env = Wrapper(env)
state = env.reset()


print(state)


 
agent = agents.AgentRandomContinuous(env)

k = 0.02
fps = 0

while True:

    time_start = time.time()
    reward, done = agent.main()
    time_stop  = time.time()

    fps = (1.0-k)*fps + k*1.0/(time_stop - time_start)

    #print(reward)
    if done:
        print("FPS = ", fps)
        env.reset()
    #time.sleep(0.01)
