import numpy
import torch
from .ExperienceBufferIM import *

import cv2
import time


class AgentDQNEntropy():
    def __init__(self, env, Model, ModelEntropy, Config):
        self.env = env

        config = Config.Config()

        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma
        
        if hasattr(config, "tau"):
            self.soft_update        = True
            self.tau                = config.tau
        elif hasattr(config, "target_update"):
            self.soft_update        = False
            self.target_update      = config.target_update
        else:
            self.soft_update        = False
            self.target_update      = 10000

        self.update_frequency       = config.update_frequency        
        self.bellman_steps          = config.bellman_steps
        
       
        self.entropy_beta           = config.entropy_beta

        self.exploration            = config.exploration

        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n


        self.experience_replay = ExperienceBufferIM(config.experience_replay_size, self.bellman_steps)

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_target   = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)


        self.model_entropy      = ModelEntropy.Model(self.state_shape, self.actions_count)
        self.optimizer_entropy  = torch.optim.Adam(self.model_entropy.parameters(), lr= config.entropy_learning_rate)

    
        self.state              = env.reset()
        self.iterations         = 0
        self.enable_training()

        self._compute_running_entropy(None, True)

        self.entropy_loss       = 0.0
        self.entropy_motivation = 0.0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self, show_activity = False):
        if self.enabled_training:
            self.exploration.process()
            self.epsilon = self.exploration.get()
        else:
            self.epsilon = self.exploration.get_testing()

        state_t     = torch.from_numpy(self.state).to(self.model.device).unsqueeze(0).float()
        
        action_idx_np, _ = self._sample_action(state_t, self.epsilon)

        self.action = action_idx_np[0]

        state_new, self.reward, done, self.info = self.env.step(self.action)
 
        if self.enabled_training:
            variance = self._compute_running_entropy(self.state, done)
            self.experience_replay.add(self.state, self.action, self.reward, done, variance)


        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()
            
            if self.soft_update:
                for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                    target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
            else:
                if self.iterations%self.target_update == 0:
                    self.model_target.load_state_dict(self.model.state_dict())


        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        if show_activity:
            self._show_entropy_activity(state_t)

        
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
        state_t, action_t, reward_t, state_next_t, done_t, entropy_t = self.experience_replay.sample(self.batch_size, self.model.device)
        
        #entropy model prediction
        action_one_hot_t      = self._one_hot_encoding(action_t)
        entropy_predicted_t   = self.model_entropy(state_t, state_next_t, action_one_hot_t)

        #env model loss
        entropy_loss = (entropy_t - entropy_predicted_t)**2
        entropy_loss = entropy_loss.mean()

        #update env model
        self.optimizer_entropy.zero_grad()
        entropy_loss.backward() 
        self.optimizer_entropy.step()

        entropy_motivation = torch.tanh(self.entropy_beta*entropy_predicted_t)
                
        #q values, state now, state next
        q_predicted      = self.model.forward(state_t)
        q_predicted_next = self.model_target.forward(state_next_t)

        #compute target, n-step Q-learning
        q_target         = q_predicted.clone()
        for j in range(self.batch_size):
            gamma_        = self.gamma

            reward_sum = 0.0
            for i in range(self.bellman_steps):
                if done_t[j][i]:
                    gamma_ = 0.0
                reward_sum+= reward_t[j][i]*(gamma_**i)

            action_idx    = action_t[j]
            q_target[j][action_idx]   = reward_sum + (gamma_**self.bellman_steps)*torch.max(q_predicted_next[j]) + entropy_motivation[j]
 
        #train DQN model
        loss  = ((q_target.detach() - q_predicted)**2)
        loss  = loss.mean() 

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()

        k = 0.02
        self.entropy_loss       = (1.0 - k)*self.entropy_loss       + k*entropy_loss.mean().detach().to("cpu").numpy()
        self.entropy_motivation = (1.0 - k)*self.entropy_motivation + k*entropy_motivation.mean().detach().to("cpu").numpy()
        

    
    def save(self, save_path): 
        self.model.save(save_path)
        self.model_entropy.save(save_path)

    def load(self, load_path):
        self.model.load(load_path)
        self.model_entropy.load(load_path)
    
    def get_log(self):
        result = "" 
        result+= str(round(self.entropy_loss, 5)) + " "
        result+= str(round(self.entropy_motivation, 5)) + " "
        
        return result


    def _one_hot_encoding(self, action_t):
        batch_size          = action_t.shape[0]
        action_one_hot_t    = torch.zeros((batch_size, self.actions_count)).to(self.model.device)
         
        for b in range(batch_size):
            action_one_hot_t[b][action_t[b]] = 1.0

        return action_one_hot_t

    def _sample_action(self, state_t, epsilon):

        batch_size = state_t.shape[0]

        q_values_t  = self.model(state_t)

        action_idx_t     = torch.zeros(batch_size).to(self.model.device)

        action_one_hot_t = torch.zeros((batch_size, self.actions_count)).to(self.model.device)

        #e-greedy strategy
        for b in range(batch_size):
            action = torch.argmax(q_values_t[b])
            if numpy.random.random() < epsilon:
                action = numpy.random.randint(self.actions_count)

            action_idx_t[b]                 = action
            action_one_hot_t[b][action]     = 1.0
        
        action_idx_np       = action_idx_t.detach().to("cpu").numpy().astype(dtype=int)

        return action_idx_np, action_one_hot_t

    def _compute_running_entropy(self, state, done, k = 0.1, threshold = 0.01):
        if done:
            self.state_mean         = numpy.zeros(self.state_shape)
            self.state_variance     = numpy.zeros(self.state_shape)
        else:
            self.state_mean         = (1.0 - k)*self.state_mean + k*state
            var                     = (state - self.state_mean)**2

            self.state_variance     = (1.0 - k)*self.state_variance + k*var

        flatten  = self.state_variance.flatten()
        shrinked = flatten[flatten > threshold]

        if shrinked.shape[0] > 0:
            variance = shrinked.mean()
        else:
            variance = 0.0

        return variance



   