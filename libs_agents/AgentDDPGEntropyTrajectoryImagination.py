import numpy
import torch
from .ExperienceBufferContinuous import *

#from torchviz import make_dot
#import sys

class AgentDDPGEntropyTrajectoryImagination():
    def __init__(self, env, ModelCritic, ModelActor, ModelForward, Config):
        self.env = env

        config = Config.Config()

        self.batch_size         = config.batch_size
        self.gamma              = config.gamma
        self.update_frequency   = config.update_frequency
        self.tau                =  config.tau

        self.exploration    = config.exploration

        self.trajectory_length      = config.trajectory_length
        self.entropy_beta           = config.entropy_beta
        self.curiosity_beta         = config.curiosity_beta

    
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.experience_replay = ExperienceBufferContinuous(config.experience_replay_size)

        self.model_actor            = ModelActor.Model(self.state_shape, self.actions_count)
        self.model_actor_target     = ModelActor.Model(self.state_shape, self.actions_count)

        self.model_critic           = ModelCritic.Model(self.state_shape, self.actions_count)
        self.model_critic_target    = ModelCritic.Model(self.state_shape, self.actions_count)

        self.model_forward          = ModelForward.Model(self.state_shape, self.actions_count)

        
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

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

        action_t, action = self._sample_action(state_t, self.epsilon)
 
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

        action_next_t   = self.model_actor_target.forward(state_next_t).detach()
        value_next_t    = self.model_critic_target.forward(state_next_t, action_next_t).detach()


        '''
        intrinsics motivation
        '''
        states_imagined_t =  self._process_imagination(state_t)

        entropy_t    = self._compute_entropy(states_imagined_t)
        entropy_t    = torch.tanh(self.entropy_beta*entropy_t)
        entropy_t    = entropy_t.detach()


        state_next_predicted_t = self.model_forward(state_t, action_t)

        loss_forward_   = ((state_next_t.detach() - state_next_predicted_t)**2)
        loss_forward    = loss_forward_.mean() 

        curiosity_t     = loss_forward_.mean(-1)
        curiosity_t     = torch.tanh(self.curiosity_beta*curiosity_t)
        curiosity_t     = curiosity_t.detach()


        #update forward
        self.optimizer_forward.zero_grad()
        loss_forward.backward() 
        self.optimizer_forward.step()


        #critic loss
        value_target    = entropy_t + curiosity_t + reward_t + self.gamma*done_t*value_next_t
        value_predicted = self.model_critic.forward(state_t, action_t)

        loss_critic     = ((value_target - value_predicted)**2)
        loss_critic     = loss_critic.mean()
     
        #update critic
        self.optimizer_critic.zero_grad()
        loss_critic.backward() 
        self.optimizer_critic.step()

        #actor loss
        loss_actor      = -self.model_critic.forward(state_t, self.model_actor.forward(state_t))
        loss_actor      = loss_actor.mean()

        #update actor
        self.optimizer_actor.zero_grad()       
        loss_actor.backward()
        self.optimizer_actor.step()

        # update target networks 
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

        #print(self.loss_forward, self.loss_actor, self.loss_critic, self.entropy, self.curiosity, "\n\n")

    def _sample_action(self, features_t, epsilon):
        action_t    = self.model_actor(features_t)
        action_t    = action_t + epsilon*torch.randn(action_t.shape).to(self.model_actor.device)
        action_t    = action_t.clamp(-1.0, 1.0)

        action_np   = action_t.detach().to("cpu").numpy()

        return action_t, action_np

    def _compute_entropy(self, x):
        return x.std(dim = 1).mean(dim = 1)

    def _process_imagination(self, initial_states_t):
        imagined_states_t = torch.zeros((self.trajectory_length, self.batch_size) + self.state_shape).to(initial_states_t.device)

        imagined_states_t[0] = initial_states_t.clone()

        for n in range(self.trajectory_length - 1):
            action_t,   _           = self._sample_action(imagined_states_t[n], self.epsilon)
            imagined_states_t[n+1]  = self.model_forward(imagined_states_t[n], action_t)

        imagined_states_t = imagined_states_t.transpose(0, 1)

        return imagined_states_t

    def save(self, save_path):
        self.model_actor.save(save_path)
        self.model_critic.save(save_path)
        self.model_forward.save(save_path)

    def load(self, load_path):
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
