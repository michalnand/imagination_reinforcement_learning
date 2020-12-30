import numpy
import torch

class ExperienceBufferGoals():
    def __init__(self, size, state_shape, actions_count):

        self.size           = size       
        self.current_idx    = 0 
        self.initialized    = False

        self.state_shape        = state_shape
        self.actions_count      = actions_count

        self._clear()


    def add(self, state, action, reward, done, motivation):

        self.state_episode.append(state)
        self.action_episode.append(action)
        self.reward_episode.append(reward)
        self.done_episode.append(done)
        self.motivation_episode.append(motivation)

        episode_length = len(self.state_episode)

        if done or episode_length > 4096:
            #find state with highest motivation in episode
            goal_episode_idx = numpy.argmax(self.motivation_episode)

            #index into global array where goal state will be stored
            goal_buffer_idx  = (self.current_idx + goal_episode_idx)%self.size
   
            for i in range(episode_length):                
                self._add(self.state_episode[i], self.action_episode[i], self.reward_episode[i], self.done_episode[i], goal_buffer_idx, self.motivation_episode[i])

            self._clear()



    def get_goal_by_motivation(self):        
        if self.initialized == False:
            return numpy.zeros(self.state_shape, dtype=numpy.float32)

        probs = numpy.exp(self.motivation_b - numpy.max(self.motivation_b))
        probs = probs/numpy.sum(probs) 

        idx = numpy.random.choice(range(self.size), p=probs)

        return self.state_b[idx] 


    def _add(self, state, action, reward, done, goal, motivation): 
        self._initialize()

        if done != 0: 
            done_ = 1.0
        else:
            done_ = 0.0

        self.state_b[self.current_idx]          = state.copy()
        self.action_b[self.current_idx]         = int(action)
        self.reward_b[self.current_idx]         = reward
        self.done_b[self.current_idx]           = done_
        self.goals_b[self.current_idx]          = goal
        self.motivation_b[self.current_idx]     = motivation

        self.current_idx = (self.current_idx + 1)%self.size

    def sample(self, batch_size, device = "cpu"):
        indices         = numpy.random.randint(0, self.size, size=batch_size)
        indices_next    = (indices + 1)%self.size

        state_t         = torch.from_numpy(numpy.take(self.state_b,     indices, axis=0)).to(device)
        state_next_t    = torch.from_numpy(numpy.take(self.state_b,     indices_next, axis=0)).to(device)
        action_t        = torch.from_numpy(numpy.take(self.action_b,    indices, axis=0)).to(device)
        reward_t        = torch.from_numpy(numpy.take(self.reward_b,    indices, axis=0)).to(device)
        done_t          = torch.from_numpy(numpy.take(self.done_b,      indices, axis=0)).to(device)
        motivation_t    = torch.from_numpy(numpy.take(self.motivation_b,      indices, axis=0)).to(device)

        goals_indices   = numpy.take(self.goals_b, indices, axis=0)
        goals_t         = torch.from_numpy(numpy.take(self.state_b,      goals_indices, axis=0)).to(device)


        return state_t, state_next_t, action_t, reward_t, done_t, goals_t, motivation_t

    
    def _initialize(self):
        if self.initialized == False:
            self.state_b        = numpy.zeros((self.size, ) + self.state_shape, dtype=numpy.float32)
            self.action_b       = numpy.zeros((self.size, ), dtype=int)
            self.reward_b       = numpy.zeros((self.size, ), dtype=numpy.float32)
            self.done_b         = numpy.zeros((self.size, ), dtype=numpy.float32)
            self.goals_b        = numpy.zeros((self.size), dtype=int)
            self.motivation_b   = numpy.zeros((self.size, ), dtype=numpy.float32)

            self.initialized    = True


    def _clear(self):
        self.state_episode      = []
        self.action_episode     = []
        self.reward_episode     = []
        self.done_episode       = []
        self.motivation_episode = []
    