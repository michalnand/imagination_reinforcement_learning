import gym
import gym_go
import numpy

import time

class GoEnv:
    def __init__(self, size = 9):
        self.size = size
        self.env = gym.make('go-v0', size=self.size)
        self.reset()

        self.fps = 0

        self.time_start = time.time()
        self.time_stop  = time.time() + 1
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space


    def reset(self):
        self.observation = self.env.reset()
        self.legal_moves = self.find_legal_moves()
        return self.observation

        
    def step(self, action):
        self.observation, reward, done, info = self.env.step(action)
        self.legal_moves = self.find_legal_moves()

        k = 0.99
        self.fps        = k*self.fps + (1.0 - k)*1.0/(self.time_stop - self.time_start)

        self.time_start  = self.time_stop
        self.time_stop   = time.time()

        return self.observation, reward, done, info

    def step_e_greedy(self, q_values, epsilon):
        action = self._step_e_greedy(q_values, epsilon)
        
        self.observation, reward, done, info = self.step(action)

        return self.observation, reward, done, info, action

    def _step_e_greedy(self, q_values, epsilon):
        #mask q_values with legal moves and covnert into probs
        probs = self.legal_moves*numpy.exp(q_values - numpy.max(q_values))
        probs = probs/numpy.sum(probs)  

        move = -1 

        if numpy.random.rand() < epsilon:
            #choose random move
            move        = numpy.random.choice(range(len(probs)), p=probs)
        else:
            #choose best move
            move = numpy.argmax(probs)
        
        return int(move)

    def render(self, mode="terminal"):
        self.env.render(mode)

    def get_active_player(self):
        if self.observation[2][0][0] == 0:
            return "black"
        else:
            return "white"


   



    def find_legal_moves(self):
        #obtain legal moves froms tate
        raw_legal_moves_mask    = 1 - self.observation[3].reshape(-1)

        #add pass move, always 1
        pass_move   = numpy.ones(1, dtype=float)
        legal_moves = numpy.concatenate((raw_legal_moves_mask, pass_move), axis=0)

        return legal_moves

   