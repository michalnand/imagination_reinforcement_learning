import numpy

import gym
from gym import spaces

class Make:
    def __init__(self, size = 8):
        self.size               = size
        
        self.action_space       = spaces.Discrete(self.size)
        self.observation_space  = spaces.Box(low=-1.0, high=1.0, shape=((2, self.size)), dtype=numpy.float32)


    def reset(self):
        self.steps        = 0 

        self.goal         = 1.0*numpy.random.randint(2, size=self.size)
        self.state        = 1.0*numpy.random.randint(2, size=self.size)

        observation  = numpy.stack((self.goal, self.state), axis = 0)

        return observation

        
    def step(self, action):

        #flip selected bit
        self.state[action] = 1 - self.state[action]

        distance = numpy.sum(numpy.abs(self.goal - self.state))

        self.steps+= 1

        #all bits equals to goal, win, end episode
        if distance < 0.001:
            done    = True
            reward  = 1.0
        #too many steps, loose, end episode
        elif self.steps >= self.size:
            done    = True
            reward  = -1.0
        #default return
        else:
            done    = False
            reward  = 0.0

        observation  = numpy.stack((self.goal, self.state), axis = 0)

        return observation, reward, done, None

   
    def render(self):
        print(self.goal, self.state)
