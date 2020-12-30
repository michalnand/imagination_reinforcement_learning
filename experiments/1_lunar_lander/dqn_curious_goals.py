import gym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *

import models.dqn_curious_goals.src.model_dqn       as ModelDQN
import models.dqn_curious_goals.src.model_forward   as ModelForward
import models.dqn_curious_goals.src.config          as Config

path = "models/dqn_curious_goals/"

class Wrapper(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward = reward / 10.0

        if reward < -1.0: 
            reward = -1.0

        if reward > 1.0:
            reward = 1.0

        return obs, reward, done, info


env = gym.make("LunarLander-v2")
env = Wrapper(env)
env.reset() 

agent = libs_agents.AgentDQNCuriousGoals(env, ModelDQN, ModelForward, Config)

max_iterations = 400000
trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
'''