import time
import gym
import gym_super_mario_bros
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.super_mario_wrapper import *

import models.dqn_baseline.src.model            as Model
import models.dqn_baseline.src.config           as Config

path = "models/dqn_baseline/"

env = gym.make("SuperMarioBros-v0")
env = SuperMarioWrapper(env)
env.reset()

agent = libs_agents.AgentDQN(env, Model, Config)

max_iterations = 10*(10**6)

'''
trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run() 
'''

agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main(False)

    env.render()
    time.sleep(0.01)
