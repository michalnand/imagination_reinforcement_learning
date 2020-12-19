import gym
import numpy
import time
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.atari_wrapper import *

import models.dqn_imagination.src.model_dqn         as ModelDQN
import models.dqn_imagination.src.model_im          as ModelIM
import models.dqn_imagination.src.config            as Config

path = "models/dqn_imagination/"

env = gym.make("Pitfall-v4")

env = AtariWrapper(env)
env.reset()

agent = libs_agents.AgentDQNImagination(env, ModelDQN, ModelIM, Config)

max_iterations = 10*(10**6) 

trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main(False)

    env.render()
    time.sleep(0.01)
'''