import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import numpy
import time


import agents




#env = gym.make("AntPyBulletEnv-v0")
env = gym.make("HalfCheetahPyBulletEnv-v0")
#env = gym.make("MinitaurBulletEnv-v0")

env.render()
state = env.reset()

print(state.shape)


 
agent = agents.AgentRandomContinuous(env)


agent.iterations = 0
while True:
    reward, done = agent.main()
    env.render()
    time.sleep(0.01)

    print(reward)

    if done:
        env.reset()
