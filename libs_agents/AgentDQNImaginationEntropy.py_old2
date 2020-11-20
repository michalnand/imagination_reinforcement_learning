import numpy
import torch
from .ExperienceBuffer import *

import cv2


class AgentDQNImaginationEntropy():
    def __init__(self, env, ModelFeatures, ModelForward, ModelReward, ModelActor, Config):
        self.env    = env
        config      = Config.Config()

        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma
        
        self.target_update      = config.target_update
        self.update_frequency   = config.update_frequency        
        

        self.state_shape        = self.env.observation_space.shape
        self.actions_count      = self.env.action_space.n

        self.experience_replay  = ExperienceBuffer(config.experience_replay_size)

        self.model_features     = ModelFeatures.Model(self.state_shape)
        features_shape          = self.model_features.features_shape
        self.model_forward      = ModelForward.Model(features_shape,    self.actions_count)
        self.model_reward       = ModelReward.Model(features_shape,     self.actions_count)
        self.model_actor        = ModelActor.Model(features_shape,      self.actions_count)
        self.model_actor_target = ModelActor.Model(features_shape,      self.actions_count)

        self.optimizer_features = torch.optim.Adam(self.model_features.parameters(), lr=config.learning_rate_features)
        self.optimizer_forward  = torch.optim.Adam(self.model_forward.parameters(), lr=config.learning_rate_forward)
        self.optimizer_reward   = torch.optim.Adam(self.model_reward.parameters(), lr=config.learning_rate_reward)
        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr=config.learning_rate_actor)
        
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        self.state          = env.reset()
        self.iterations     = 0
        self.enable_training()

        self.loss_features  = 0.0
        self.loss_forward   = 0.0
        self.loss_reward    = 0.0
        self.loss_actor     = 0.0
        


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
             
        state_t     = torch.from_numpy(self.state).to(self.model_actor.device).unsqueeze(0).float()

        features, _ = self.model_features(state_t)
        
        action_idx_np, _,  = self._sample_action(features, epsilon)

        action = action_idx_np[0]

        state_new, reward, done, self.info = self.env.step(action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, action, reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self._training()
 
            if self.iterations%self.target_update == 0:
                self.model_actor_target.load_state_dict(self.model_actor.state_dict())

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        if show_activity:
            self._show_activity(self.state)

        self.iterations+= 1

        return reward, done
    
    def _show_activity(self, state, alpha = 0.6):
        activity_map    = self.model.get_activity_map(state)
        activity_map    = numpy.stack((activity_map,)*3, axis=-1)*[0, 0, 1]

        state_map    = numpy.stack((state[0],)*3, axis=-1)
        image        = alpha*state_map + (1.0 - alpha)*activity_map

        image        = (image - image.min())/(image.max() - image.min())

        image = cv2.resize(image, (400, 400), interpolation = cv2.INTER_AREA)
        cv2.imshow('state activity', image)
        cv2.waitKey(1)
        
    def _training(self):
        '''
        1, sample random minibatch
        '''
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_actor.device)
        action_one_hot_t   = self._action_one_hot(action_t)

        '''
        2, predict features for next state
        '''
        features_t, state_predicted_t = self.model_features(state_t)

        '''
        3, train features model, MSE loss
        '''
        loss_features  = ((state_t - state_predicted_t)**2)
        loss_features  = loss_features.mean() 

        self.optimizer_features.zero_grad()
        loss_features.backward()
        self.optimizer_features.step()

        '''
        4, compute features for next state
        '''
        features_next_t, _ = self.model_features(state_next_t)
        
        '''
        5, predict features for next state from current features and action
        train forward model, MSE loss
        note : loss_forward is also curiosity in features space
        '''
        features_next_predicted_t = self.model_forward(features_t.detach(), action_one_hot_t.detach())

        loss_forward  = ((features_next_t.detach() - features_next_predicted_t)**2)
        loss_forward  = loss_forward.mean() 

        self.optimizer_forward.zero_grad()
        loss_forward.backward()
        self.optimizer_forward.step()

        '''
        6, train reward model, MSE loss
        '''
        reward_predicted_t = self.model_reward(features_t.detach(), action_one_hot_t.detach())
        
        loss_reward  = ((reward_t - reward_predicted_t)**2)
        loss_reward  = loss_reward.mean() 

        self.optimizer_reward.zero_grad()
        loss_reward.backward()
        self.optimizer_reward.step()

        '''
        7, Q-learning for actor, MSE loss with gradient clipping
        features_t and features_next_t are used as state
        '''

        #q values, state now, state next
        q_predicted      = self.model_actor.forward(features_t.detach())
        q_predicted_next = self.model_actor_target.forward(features_next_t.detach())

        #compute target, Q-learning
        q_target         = q_predicted.clone()
        for j in range(self.batch_size): 
            action_idx              = action_t[j]
            q_target[j][action_idx] = reward_t[j] + self.gamma*torch.max(q_predicted_next[j])*(1- done_t[j])
 
        #train DQN model
        loss_actor  = ((q_target.detach() - q_predicted)**2)
        loss_actor  = loss_actor.mean() 

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        for param in self.model_actor.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer_actor.step()

        k = 0.02

        self.loss_features  = (1.0 - k)*self.loss_features + k*loss_features.detach().to("cpu").numpy()
        self.loss_forward   = (1.0 - k)*self.loss_forward + k*loss_forward.detach().to("cpu").numpy()
        self.loss_reward    = (1.0 - k)*self.loss_reward + k*loss_reward.detach().to("cpu").numpy()
        self.loss_actor     = (1.0 - k)*self.loss_actor + k*loss_actor.detach().to("cpu").numpy()

       
    def _sample_action(self, state_t, epsilon):

        batch_size = state_t.shape[0]

        q_values_t          = self.model_actor(state_t).to("cpu")

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
        action_one_hot_t = self._action_one_hot(action_idx_t)

        #numpy result
        action_idx_np       = action_idx_t.detach().to("cpu").numpy().astype(dtype=int)

        return action_idx_np, action_one_hot_t

    def _action_one_hot(self, action_idx_t):
        batch_size = action_idx_t.shape[0]

        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model_actor.device)

        return action_one_hot_t


    def _imagine_trajectory(self, features_initial_t, trajectory_length):
        '''
        #TODO
        1, compute Q values from actor, and Q values next from target actor
        2, use forward model to traverse trough imagined states
        3, use reward model to obtain rewards
        '''

        q_values        = torch.zeros((trajectory_length, self.batch_size, self.actions_count) ).to(self.model_actor.device)
        q_values_next   = torch.zeros((trajectory_length, self.batch_size, self.actions_count) ).to(self.model_actor.device)
        action_t        = torch.zeros((trajectory_length, self.batch_size, ), dtype=int).to(self.model_actor.device)
        reward_t        = torch.zeros((trajectory_length, self.batch_size, )).to(self.model_actor.device)

        for t in range(trajectory_length):
            pass


        return q_values, q_values_next, action_t, reward_t


    def save(self, save_path):
        self.model_features.save(save_path)
        self.model_forward.save(save_path)
        self.model_reward.save(save_path)
        self.model_actor.save(save_path)

    def load(self, load_path):
        self.model_features.load(load_path)
        self.model_forward.load(load_path)
        self.model_reward.load(load_path)
        self.model_actor.load(load_path)

    def get_log(self):
        result = "" 
        result+= str(round(self.loss_features, 5)) + " "
        result+= str(round(self.loss_forward, 5)) + " "
        result+= str(round(self.loss_reward, 5)) + " "
        result+= str(round(self.loss_actor, 5)) + " "

        return result




