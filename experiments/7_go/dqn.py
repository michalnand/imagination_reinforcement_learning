import sys
sys.path.insert(0, '../..')

import libs_agents
import libs_envs.go.go_wrapper
from libs_common.Training import *

import models.dqn.src.model            as Model
import models.dqn.src.config           as Config


path = "models/dqn/"

env = libs_envs.go.go_wrapper.GoEnv(size=19)

agent = libs_agents.AgentDQNDuel(env, Model, Config)

max_iterations = 100*(10**6) 

trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()
    env.render()
'''