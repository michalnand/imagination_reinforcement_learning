import numpy
import torch



class ExperienceBuffer():

    def __init__(self, size):
        self.size   = size
       
        self.ptr      = 0 
        self.state_b  = []
        self.action_b = []
        self.reward_b = []
        self.done_b   = []

    
    def length(self):
        return len(self.state_b)

    def is_full(self):
        if self.length() == self.size:
            return True
            
        return False

    def add(self, state, action, reward, done):

        if done != 0:
            done_ = 1.0
        else:
            done_ = 0.0

        if self.length() < self.size:
            self.state_b.append(state.copy())
            self.action_b.append(int(action))
            self.reward_b.append(reward)
            self.done_b.append(done_)
            
        else:
            self.state_b[self.ptr]  = state.copy()
            self.action_b[self.ptr] = int(action)
            self.reward_b[self.ptr] = reward
            self.done_b[self.ptr]   = done_

            self.ptr = (self.ptr + 1)%self.length()


    def _print(self):
        for i in range(self.length()):
            #print(self.state_b[i], end = " ")
            print(self.action_b[i], end = " ")
            print(self.reward_b[i], end = " ")
            print(self.done_b[i], end = " ")
            print("\n")

   
    def sample(self, batch_size, device):
        
        state_shape     = (batch_size, ) + self.state_b[0].shape[0:]
        action_shape    = (batch_size, )
        reward_shape    = (batch_size, )
        done_shape      = (batch_size, )
      

        state_t         = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        action_t        = torch.zeros(action_shape,  dtype=int)
        reward_t        = torch.zeros(reward_shape,  dtype=torch.float32)
        state_next_t    = torch.zeros(state_shape,  dtype=torch.float32).to(device)
        done_t          = torch.zeros(done_shape,  dtype=torch.float32).to(device)

        indices = self.find_indices_random(batch_size)

        for j in range(batch_size): 
            n = indices[j]

            state_t[j]         = torch.from_numpy(self.state_b[n]).to(device)
            action_t[j]        = self.action_b[n]
            state_next_t[j]    = torch.from_numpy(self.state_b[n + 1]).to(device)
            
            reward_t[j]        = self.reward_b[n]
            done_t[j]          = self.done_b[n]

        reward_t    = reward_t.to(device)
        done_t      = done_t.to(device) 
        
        return state_t.detach(), action_t, reward_t.detach(), state_next_t.detach(), done_t.detach()

    def find_indices_random(self, count):
        indices = numpy.zeros(count, dtype=int)
        for i in range(count):
            indices[i]  = numpy.random.randint(self.length() - 1)
        
        return indices