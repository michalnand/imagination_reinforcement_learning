import gym
import numpy

class Game2048Wrapper(gym.Wrapper):
    def __init__(self, env, size):
        gym.Wrapper.__init__(self, env)

      
        self.observation_space.shape = (1, size, size)
        self.score      = 0
        self.max_tile   = 0
        self.max_value  = 15

        self.obs_prev = None
        self.stats = {}
        self.stats_norm = {}

        for i in range(self.max_value):
            self.stats[int(2**(i+1))] = 0
            self.stats_norm[int(2**(i+1))] = 0

    def reset(self):
        obs = self.env.reset()
        
        self.score, self.max_tile = self._update_score(obs)
        self._update_stats()
       
        return self._parse_state(obs)
        
    def step(self, action):
        obs, reward, done, info     = self.env.step(action)

        if reward > 1.0:
            reward = numpy.log2(reward)/self.max_value
        elif done:
            reward = -1.0
        else:
            reward = 0.0
        
        self.score, self.max_tile = self._update_score(obs)
        self._update_stats()
       

        return self._parse_state(obs), reward, done, info 

    def _parse_state(self, state):
        state_norm = numpy.log2(numpy.clip(state, 1, 2**self.max_value))/self.max_value
        state_norm = numpy.expand_dims(state_norm, 0)

        return state_norm

    def _update_score(self, obs):

        max_tile  = numpy.max(obs)
        sum_tiles = numpy.sum(obs)

        return sum_tiles, max_tile

    def _update_stats(self):
        sum = 0
        self.stats[int(self.max_tile)]+= 1
        for v in self.stats:
            sum+= self.stats[v]
        
        self.stats_norm = {}
        for v in self.stats:
            self.stats_norm[v] = round(100.0*self.stats[v]/(sum + 0.0000001), 1)
