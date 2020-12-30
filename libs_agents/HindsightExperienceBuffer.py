import numpy
import torch


class HindsightExperienceBuffer(object):
  
    def __init__(self, size, state_shape, actions_count, additional_goals_count = 4, state_dtype = numpy.float32):

        self.size                   = size
        self.additional_goals_count = additional_goals_count
       
        self.current_idx     = 0 

        self.state_b        = numpy.zeros((self.size, ) + state_shape, dtype=state_dtype)
        self.state_next_b   = numpy.zeros((self.size, ) + state_shape, dtype=state_dtype)
        self.action_b       = numpy.zeros((self.size, actions_count), dtype=int)
        self.reward_b       = numpy.zeros((self.size, ), dtype=numpy.float32)
        self.done_b         = numpy.zeros((self.size, ), dtype=numpy.float32)

        self._clear()

    def length(self):
        return self.size

    def is_full(self):
        if self.length() == self.size:
            return True
            
        return False

    def add(self, state, state_next, action, reward, done):
        self._add(state, state_next, action, reward, done)
        
        self.state_episode.append(state)
        self.state_next_episode.append(state_next)
        self.action_episode.append(action)
        self.reward_episode.append(reward)
        self.done_episode.append(done)

        episode_length = len(self.state_episode)

        if done or episode_length > 2048:

            for g in range(self.additional_goals_count):
                idx      = numpy.random.randint(episode_length)
                new_goal = self.state_episode[idx][0].copy()

                for t in range(episode_length):
                    
                    dif = ((self.state_episode[t][0] - new_goal)**2).mean()

                    if dif < 0.0001:
                        reward_ = 1.0
                        done_   = True
                    else:
                        reward_ = 0.0
                        done_   = False

                    new_state_episode       = numpy.stack((new_goal, self.state_episode[t][0]), axis=0)
                    new_state_next_episode  = numpy.stack((new_goal, self.state_next_episode[t][0]), axis=0)

                    self._add(new_state_episode, new_state_next_episode, self.action_episode[t], reward_, done_)

            self._clear()
                

    def _add(self, state, state_next, action, reward, done): 

        if done != 0: 
            done_ = 1.0
        else:
            done_ = 0.0

        self.state_b[self.current_idx]          = state.copy()
        self.state_next_b[self.current_idx]     = state_next.copy()
        self.action_b[self.current_idx]         = int(action)
        self.reward_b[self.current_idx]         = reward
        self.done_b[self.current_idx]           = done_

        self.current_idx = (self.current_idx + 1)%self.length()

    def sample_goal_state(self):
        indices = numpy.nonzero(self.done_b)[0]

        if len(indices) == 0:
            idx     = numpy.random.randint(self.size)      
        else: 
            idx     = numpy.random.choice(indices)  

        return self.state_b[idx].copy()
        
    def sample(self, batch_size, device):
        indices         = numpy.random.randint(0, self.size, size=batch_size)

        state_t         = torch.from_numpy(numpy.take(self.state_b,         indices, axis=0)).to(device)
        state_next_t    = torch.from_numpy(numpy.take(self.state_next_b,    indices, axis=0)).to(device)
        action_t        = torch.from_numpy(numpy.take(self.action_b,        indices, axis=0)).to(device)
        reward_t        = torch.from_numpy(numpy.take(self.reward_b,        indices, axis=0)).to(device)
        done_t          = torch.from_numpy(numpy.take(self.done_b,          indices, axis=0)).to(device)

        return state_t, state_next_t, action_t, reward_t, done_t


    def _clear(self):
        self.state_episode      = []
        self.state_next_episode = []
        self.action_episode     = []
        self.reward_episode     = []
        self.done_episode       = []
    