import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import numpy
import time


import libs_agents

from libs_common.dynamic_state_graph import *



#env = gym.make("AntPyBulletEnv-v0")
env = gym.make("HalfCheetahPyBulletEnv-v0")
#env = gym.make("HopperPyBulletEnv-v0")

env.render()
state = env.reset()

print(state.shape)

dg = DynamicGraphState(state.shape[0])
 
agent = libs_agents.AgentRandomContinuous(env)


agent.iterations = 0
while True:
    reward, done = agent.main()

    state_masked, adjacency_matrix, edge_index = dg.process_state(agent.state)

    env.render()
    time.sleep(0.01)


    if done:
        print(adjacency_matrix, "\n\n")
        print(state_masked, "\n\n\n\n\n")
      
        dg.show_graph()
        env.reset()
