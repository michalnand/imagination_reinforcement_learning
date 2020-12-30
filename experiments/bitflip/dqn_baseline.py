import gym
import numpy
import time
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.atari_wrapper import *

import models.dqn_baseline.src.model            as Model
import models.dqn_baseline.src.config           as Config

import BitFlippingEnv 

path = "models/dqn_baseline/"

env = BitFlippingEnv.Make(8)
env.reset()

agent = libs_agents.AgentDQN(env, Model, Config)
 
max_iterations = 200000

trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main(False)
    

    env.render()

    time.sleep(0.1)
'''