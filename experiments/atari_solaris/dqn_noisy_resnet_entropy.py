import gym
import numpy
import time
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.atari_wrapper import *

import models.dqn_entropy_motivation.src.model          as Model
import models.dqn_entropy_motivation.src.model_entropy  as ModelEntropy
import models.dqn_entropy_motivation.src.config         as Config


path = "models/dqn_entropy_motivation/"

env = gym.make("SolarisNoFrameskip-v4")

env = AtariWrapper(env)
env.reset()


agent = libs_agents.AgentDQNEntropy(env, Model, ModelEntropy, Config)

max_iterations = 10*(10**6) 

trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main(True)

    env.render()
    time.sleep(0.01)
'''