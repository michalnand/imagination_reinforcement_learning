import numpy
import torch
from .ExperienceBufferContinuous import *

#from torchviz import make_dot
#import sys

class AgentDDPGImaginationEntropy():
    def __init__(self, env, ModelFeatures, ModelCritic, ModelActor, ModelForward, Config):
        self.env = env

        config = Config.Config()

        self.batch_size         = config.batch_size
        self.gamma              = config.gamma
        self.update_frequency   = config.update_frequency
        self.tau                =  config.tau

        self.exploration    = config.exploration

        self.imagination_rollouts   = config.imagination_rollouts
        self.imagination_steps      = config.imagination_steps
        self.entropy_beta           = config.entropy_beta
        self.curiosity_beta         = config.curiosity_beta

    
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.experience_replay = ExperienceBufferContinuous(config.experience_replay_size)

        self.model_features            = ModelFeatures.Model(self.state_shape)
        self.model_features_target     = ModelFeatures.Model(self.state_shape)
        features_shape                 = self.model_features.features_shape

        self.model_actor            = ModelActor.Model(features_shape, self.actions_count)
        self.model_actor_target     = ModelActor.Model(features_shape, self.actions_count)

        self.model_critic           = ModelCritic.Model(features_shape, self.actions_count)
        self.model_critic_target    = ModelCritic.Model(features_shape, self.actions_count)

        self.model_forward          = ModelForward.Model(features_shape, self.actions_count)

        for target_param, param in zip(self.model_features_target.parameters(), self.model_features.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_features = torch.optim.Adam(self.model_features.parameters(), lr= config.learning_rate_features)
        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr= config.learning_rate_actor)
        self.optimizer_critic   = torch.optim.Adam(self.model_critic.parameters(), lr= config.learning_rate_critic)
        self.optimizer_forward  = torch.optim.Adam(self.model_forward.parameters(), lr= config.learning_rate_forward)

        self.state          = env.reset()

        self.iterations     = 0

        self.enable_training()

        self.loss_forward   = 0.0
        self.loss_actor     = 0.0
        self.loss_critic    = 0.0
        self.entropy        = 0.0
        self.curiosity      = 0.0

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):
        if self.enabled_training:
            self.exploration.process()
            self.epsilon = self.exploration.get()
        else:
            self.epsilon = self.exploration.get_testing()
       
        state_t     = torch.from_numpy(self.state).to(self.model_actor.device).unsqueeze(0).float()

        features_t  = self.model_features(state_t)
        action_t, action = self._sample_action(features_t, self.epsilon)
 
        action = action.squeeze()

        state_new, self.reward, done, self.info = self.env.step(action)

        if self.enabled_training:
            self.experience_replay.add(self.state, action, self.reward, done)

        if self.enabled_training and self.experience_replay.length() > 0.1*self.experience_replay.size:
            if self.iterations%self.update_frequency == 0:
                self.train_model()

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        self.iterations+= 1

        return self.reward, done
        
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_critic.device)
        
        reward_t = reward_t.unsqueeze(-1)
        done_t   = (1.0 - done_t).unsqueeze(-1)

        '''
        predict features for state and next state
        '''
        features_t          = self.model_features(state_t)
        features_next_t     = self.model_features_target(state_next_t)

        action_next_t   = self.model_actor_target.forward(features_next_t).detach()
        value_next_t    = self.model_critic_target.forward(features_next_t, action_next_t).detach()


        '''
        imagine states, and compute their entropy
        '''        
        features_imagined_t = self._imagine_states(features_t.detach(), self.imagination_rollouts, self.imagination_steps, self.epsilon)

        entropy_t           = self._compute_entropy(features_imagined_t)
        entropy_t           = torch.tanh(self.entropy_beta*entropy_t)
        entropy_t           = entropy_t.detach()


        '''
        predict next features, and compute forward model loss 
        note : forward model learns next features
        '''
        features_predicted_t    = self.model_forward(features_t, action_t)

        loss_forward_   = ((features_next_t.detach() - features_predicted_t)**2)
        loss_forward    = loss_forward_.mean() 

        curiosity_t     = loss_forward_.view(loss_forward_.size(0), -1).mean(dim=1)
        curiosity_t     = torch.tanh(self.curiosity_beta*curiosity_t)
        curiosity_t     = curiosity_t.detach()



        #critic loss
        value_target    = entropy_t + curiosity_t + reward_t + self.gamma*done_t*value_next_t
        value_predicted = self.model_critic.forward(features_t, action_t)

        loss_critic     = ((value_target - value_predicted)**2)
        loss_critic     = loss_critic.mean()
     
        
        #actor loss
        loss_actor      = -self.model_critic.forward(features_t, self.model_actor.forward(features_t))
        loss_actor      = loss_actor.mean()


        #compute loss
        loss = loss_critic + loss_actor + loss_forward

        #train models    
        self.optimizer_features.zero_grad()
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        self.optimizer_forward.zero_grad()


        loss.backward() 


        self.optimizer_features.step()
        self.optimizer_actor.step()
        self.optimizer_critic.step()
        self.optimizer_forward.step()


        #update target networks 
        for target_param, param in zip(self.model_features_target.parameters(), self.model_features.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
       
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
       
        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)


        '''
        log some stats, using exponential smoothing
        '''
        k = 0.02

        self.loss_forward   = (1.0 - k)*self.loss_forward   + k*loss_forward.detach().to("cpu").numpy()
        self.loss_actor     = (1.0 - k)*self.loss_actor     + k*loss_actor.detach().to("cpu").numpy()
        self.loss_critic    = (1.0 - k)*self.loss_critic    + k*loss_critic.detach().to("cpu").numpy()
        self.entropy        = (1.0 - k)*self.entropy        + k*entropy_t.mean().detach().to("cpu").numpy()
        self.curiosity      = (1.0 - k)*self.curiosity      + k*curiosity_t.mean().detach().to("cpu").numpy()


        '''
        print(self.loss_forward, self.loss_actor, self.loss_critic, self.entropy, self.curiosity, "\n\n")

        make_dot(loss).render("model_graph", format="png")
        sys.exit()
        '''

    def _sample_action(self, features_t, epsilon):
        action_t    = self.model_actor(features_t)
        action_t    = action_t + epsilon*torch.randn(action_t.shape).to(self.model_actor.device)
        action_t    = action_t.clamp(-1.0, 1.0)

        action_np   = action_t.detach().to("cpu").numpy()

        return action_t, action_np


    def _imagine_states(self, features_initial_t, rollouts, steps, epsilon):
        batch_size = features_initial_t.shape[0]

        features_shape = features_initial_t.shape[1:]

       
        '''
        features_initial = (imagination_rollouts, batch_size, features_shape)
        '''
        features_initial = torch.zeros((rollouts, batch_size, ) + features_shape).to(features_initial_t.device)
        for r in range(rollouts):
            features_initial[r] = features_initial_t.clone()

        features_imagined_t = torch.zeros((rollouts, batch_size, ) + features_shape).to(features_initial_t.device)

        for s in range(steps):
            for r in range(rollouts):
                action_t, _             = self._sample_action(features_initial[r], epsilon)
                features_imagined_t[r]  = self.model_forward(features_initial[r], action_t).detach()
                features_initial[r]     = features_imagined_t[r].clone()

        '''
        swap axis to have batch first
        shape = (batch_size, imagination_rollouts, features_shape)
        '''
        features_imagined_t = features_imagined_t.transpose(1, 0)
     
        return features_imagined_t


    def _compute_entropy(self, x):
        batch_size  = x.shape[0]
        result      = torch.zeros(batch_size).to(x.device)

        for b in range(batch_size):
            flatten     = x[b].view(x[b].size(0), -1)
            var         = torch.std(flatten, dim=0)
            result[b]   = var.mean()
        
        return result


    def save(self, save_path):
        self.model_features.save(save_path)
        self.model_actor.save(save_path)
        self.model_critic.save(save_path)
        self.model_forward.save(save_path)

    def load(self, load_path):
        self.model_features.load(load_path)
        self.model_actor.load(load_path)
        self.model_critic.load(load_path)
        self.model_forward.load(load_path)

    def get_log(self):
        result = "" 
        result+= str(round(self.loss_forward, 7)) + " "
        result+= str(round(self.loss_actor, 7)) + " "
        result+= str(round(self.loss_critic, 7)) + " "
        result+= str(round(self.entropy, 7)) + " "
        result+= str(round(self.curiosity, 7)) + " "

        return result
