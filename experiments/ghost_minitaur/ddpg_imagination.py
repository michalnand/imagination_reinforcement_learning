import numpy
import time
import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *


import models.ddpg_imagination.model.src.model_critic     as ModelCritic
import models.ddpg_imagination.model.src.model_actor      as ModelActor
import models.ddpg_imagination.model.src.model_forward    as ModelForward
import models.ddpg_imagination.model.src.config           as Config

from wrapper import *

path = "models/ddpg_imagination/model/"

env = env_create(render_eanbled=False)

agent = libs_agents.AgentDDPGEntropyTrajectoryImagination(env, ModelCritic, ModelActor, ModelForward, Config)

max_iterations = 6*(10**6)
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