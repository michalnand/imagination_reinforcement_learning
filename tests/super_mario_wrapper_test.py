import sys
sys.path.insert(0, '..')
import gym
import gym_super_mario_bros
import numpy
import time

import libs_agents

import libs_common.super_mario_wrapper




env = gym.make("SuperMarioBrosRandomStages-v0")
env = libs_common.super_mario_wrapper.SuperMarioWrapper(env)
env.reset()

agent = libs_agents.AgentRandom(env)


obs, _, _, _ = env.step(0)


print(obs)
print(obs.shape)

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
