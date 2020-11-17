import torch
import numpy


class PolicyBuffer:

    def __init__(self, buffer_size, state_shape, actions_size, device, multi_value_buffer = False):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.device         = device
        self.multi_value_buffer = multi_value_buffer

        self.clear()

    def add(self, state, logits, value, action, reward, done):
        
        if self.ptr < self.buffer_size:

            self.states_b[self.ptr]    = state
            self.logits_b[self.ptr]    = logits
            self.values_b[self.ptr]    = value
            self.actions_b[self.ptr]   = action
            self.rewards_b[self.ptr]   = reward
            self.dones_b[self.ptr]     = done
        
            self.ptr = self.ptr + 1 

    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 

    def clear(self):
        self.states_b           = torch.zeros((self.buffer_size, ) + self.state_shape).to(self.device)
        self.logits_b           = torch.zeros((self.buffer_size, self.actions_size)).to(self.device)
        
        if self.multi_value_buffer:
            self.values_b           = torch.zeros((self.buffer_size, self.actions_size)).to(self.device)
        else:
            self.values_b           = torch.zeros((self.buffer_size, 1)).to(self.device)

        self.actions_b          = torch.zeros((self.buffer_size, ), dtype=int).to(self.device)
        self.rewards_b          = torch.zeros((self.buffer_size, )).to(self.device)
        self.dones_b            = torch.zeros((self.buffer_size, )).to(self.device)
       
        self.returns_b         = torch.zeros((self.buffer_size, )).to(self.device)

        self.ptr = 0 

    def cut_zeros(self):
        self.states_b           = self.states_b[0:self.ptr]
        self.logits_b           = self.logits_b[0:self.ptr]
        self.values_b           = self.values_b[0:self.ptr]
        self.actions_b          = self.actions_b[0:self.ptr]
        self.rewards_b          = self.rewards_b[0:self.ptr]
        self.dones_b            = self.dones_b[0:self.ptr]
       
        self.returns_b         = self.returns_b[0:self.ptr]


    def compute_returns(self, gamma, normalise = False):
        
        if normalise:
            self.rewards_b = (self.rewards_b - self.rewards_b.mean())/(self.rewards_b.std() + 0.00001)
        
        q = 0.0
        for n in reversed(range(len(self.rewards_b))):

            if self.dones_b[n]:
                gamma_ = 0.0
            else:
                gamma_ = gamma

            q = self.rewards_b[n] + gamma_*q
            self.returns_b[n] = q