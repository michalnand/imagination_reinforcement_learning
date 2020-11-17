import gym
import gym_line_follower
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *

import models.ddpg.src.model_critic     as ModelCritic
import models.ddpg.src.model_actor      as ModelActor
import models.ddpg.src.config           as Config
from libs_common.Training import *

path = "models/ddpg/"


class Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        
    def observation(self, state):
        state_np = numpy.array(state).astype(numpy.float32)
        return state_np


env = gym.make("LineFollower-v0", gui = True)
env = Wrapper(env)

agent = libs_agents.AgentDDPG(env, ModelCritic, ModelActor, Config)

max_iterations = (10**5)
#trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
#trainig.run() 


agent.load(path)
agent.disable_training()
while True:
    agent.main()
    env.render()
    time.sleep(0.01)
