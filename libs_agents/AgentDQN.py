import numpy
import torch
from .ExperienceBuffer import *

import cv2


class AgentDQN():
    def __init__(self, env, Model, Config):
        self.env = env
 
        config = Config.Config()

        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma

        self.target_update      = config.target_update
        self.update_frequency   = config.update_frequency        
               
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.experience_replay = ExperienceBuffer(config.experience_replay_size)

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_target   = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)


        self.state    = env.reset()

        self.iterations     = 0

        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self, show_activity = False):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
             
        state_t     = torch.from_numpy(self.state).to(self.model.device).unsqueeze(0).float()
        
        action_idx_np, _,  = self._sample_action(state_t, epsilon)

        self.action = action_idx_np[0]

        state_new, self.reward, done, self.info = self.env.step(self.action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, self.action, self.reward, done)


        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

            if self.iterations%self.target_update == 0:
                self.model_target.load_state_dict(self.model.state_dict())

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        if show_activity:
            self._show_activity(self.state)

        self.iterations+= 1

        return self.reward, done
    
    def _show_activity(self, state, alpha = 0.6):
        activity_map    = self.model.get_activity_map(state)
        activity_map    = numpy.stack((activity_map,)*3, axis=-1)*[0, 0, 1]

        state_map    = numpy.stack((state[0],)*3, axis=-1)
        image        = alpha*state_map + (1.0 - alpha)*activity_map

        image        = (image - image.min())/(image.max() - image.min())

        image = cv2.resize(image, (400, 400), interpolation = cv2.INTER_AREA)
        cv2.imshow('state activity', image)
        cv2.waitKey(1)
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model.device)

        #q values, state now, state next
        q_predicted      = self.model.forward(state_t)
        q_predicted_next = self.model_target.forward(state_next_t)

        #compute target, n-step Q-learning
        q_target         = q_predicted.clone()
        for j in range(self.batch_size): 
            action_idx              = action_t[j]
            q_target[j][action_idx] = reward_t[j] + self.gamma*torch.max(q_predicted_next[j])*(1- done_t[j])
 
        #train DQN model
        loss  = ((q_target.detach() - q_predicted)**2)
        loss  = loss.mean() 

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()

       
    def _sample_action(self, state_t, epsilon):

        batch_size = state_t.shape[0]

        q_values_t          = self.model(state_t).to("cpu")

        #best actions indices
        q_max_indices_t     = torch.argmax(q_values_t, dim = 1)

        #random actions indices
        q_random_indices_t  = torch.randint(self.actions_count, (batch_size,))

        #create mask, which actions will be from q_random_indices_t and which from q_max_indices_t
        select_random_mask_t= torch.tensor((torch.rand(batch_size) < epsilon).clone(), dtype = int)

        #apply mask
        action_idx_t    = select_random_mask_t*q_random_indices_t + (1 - select_random_mask_t)*q_max_indices_t
        action_idx_t    = torch.tensor(action_idx_t, dtype=int)

        #create one hot encoding
        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model.device)

        #numpy result
        action_idx_np       = action_idx_t.detach().to("cpu").numpy().astype(dtype=int)

        return action_idx_np, action_one_hot_t

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    



