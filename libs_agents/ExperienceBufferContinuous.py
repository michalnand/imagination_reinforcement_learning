import numpy
import torch


class ExperienceBufferContinuous():

    def __init__(self, size):
        self.size   = size
       
        self.ptr    = 0 
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
            self.action_b.append(action)
            self.reward_b.append(reward)
            self.done_b.append(done_)
        else:
            self.state_b[self.ptr]  = state.copy()
            self.action_b[self.ptr] = action.copy()
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
        action_shape    = (batch_size, ) + self.action_b[0].shape[0:]
        reward_shape    = (batch_size, )
        done_shape      = (batch_size, )

        state_t         = torch.zeros(state_shape,  dtype=torch.float32)
        action_t        = torch.zeros(action_shape,  dtype=torch.float32)
        reward_t        = torch.zeros(reward_shape,  dtype=torch.float32)
        state_next_t    = torch.zeros(state_shape,  dtype=torch.float32)
        done_t          = torch.zeros(done_shape,  dtype=torch.float32)

        self.indices = []
        for ib in range(batch_size):
            self.indices.append(numpy.random.randint(self.length() - 1))

        for b in range(batch_size):
            n               = self.indices[b]
            state_t[b]      = torch.from_numpy(self.state_b[n])
            action_t[b]     = torch.from_numpy(self.action_b[n])
            reward_t[b]     = torch.from_numpy(numpy.asarray(self.reward_b[n]))
            state_next_t[b] = torch.from_numpy(self.state_b[n+1])
            done_t[b]       = torch.from_numpy(numpy.asarray(self.done_b[n]))

        state_t         = state_t.to(device).detach()
        action_t        = action_t.to(device).detach()
        reward_t        = reward_t.to(device).detach()
        state_next_t    = state_next_t.to(device).detach()
        done_t          = done_t.to(device).detach()

        return state_t, action_t, reward_t, state_next_t, done_t

    def sample_sequence(self, batch_size, sequence_length, device, use_indices = True):
        state_seq_shape     = (batch_size, sequence_length) + self.state_b[0].shape[0:]
        action_seq_shape    = (batch_size, sequence_length) + self.action_b[0].shape[0:]
        state_shape         = (batch_size, ) + self.state_b[0].shape[0:]
        
        state_seq_t     = torch.zeros(state_seq_shape,  dtype=torch.float32)
        action_seq_t        = torch.zeros(action_seq_shape,  dtype=torch.float32)
        state_next_t    = torch.zeros(state_shape,  dtype=torch.float32)

        for b in range(batch_size):
            if use_indices:
                n = self.indices[b]
            else:
                n  = sequence_length + numpy.random.randint(self.length() - 1 - sequence_length)

            for s in reversed(range(sequence_length)):
                state_seq_t[b][s]      = torch.from_numpy(self.state_b[n - s])
                action_seq_t[b][s]     = torch.from_numpy(self.action_b[n - s])

            state_next_t[b] = torch.from_numpy(self.state_b[n+1])
 
        state_seq_t     = state_seq_t.to(device).detach()
        action_seq_t    = action_seq_t.to(device).detach()
        state_next_t    = state_next_t.to(device).detach()

        return state_seq_t, action_seq_t, state_next_t
    
if __name__ == "__main__":
    state_shape     = (3, 13, 17)
    action_shape    = (7,)

    replay_buffer = ExperienceBufferContinuous(107)

    for i in range(1000):
        state   = numpy.random.randn(state_shape[0], state_shape[1], state_shape[2])
        action  = numpy.random.randn(action_shape[0])
        reward  = numpy.random.rand(1)
        done    = numpy.random.randint(2)

        replay_buffer.add(state, action, reward, done)

        if i > 1:
            state_t, action_t, reward_t, state_next_t, done_t = replay_buffer.sample(5, device="cpu")


    print(state_t.shape)
    print(action_t.shape)
    print(reward_t.shape)
    print(state_next_t.shape)
    print(done_t.shape)

