import gym
import pybullet_envs
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *


import models.ddpg_imagination_advanced.model.src.model_critic     as ModelCritic
import models.ddpg_imagination_advanced.model.src.model_actor      as ModelActor
import models.ddpg_imagination_advanced.model.src.model_forward    as ModelForward
import models.ddpg_imagination_advanced.model.src.config           as Config

path = "models/ddpg_imagination_advanced/model/"

env = pybullet_envs.make("AntBulletEnv-v0")
#env.render()

agent = libs_agents.AgentDDPGImaginationAdvanced(env, ModelCritic, ModelActor, ModelForward, Config)

max_iterations = 4*(10**6)
trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run()


'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()
    env.render()
    time.sleep(0.01)
'''