import numpy
import torch
from .ExperienceBufferContinuous import *

class AgentDDPGImaginationEntropy():
    def __init__(self, env, ModelCritic, ModelActor, ModelEnv, Config):
        self.env = env

        config = Config.Config()

        self.batch_size         = config.batch_size
        self.gamma              = config.gamma
        self.update_frequency   = config.update_frequency
        self.tau                = config.tau

        self.exploration        = config.exploration

        self.imagination_rollouts   = config.imagination_rollouts
        self.imagination_steps      = config.imagination_steps

        self.entropy_beta           = config.entropy_beta
        self.curiosity_beta         = config.curiosity_beta
        
        self.env_learning_rate      = config.env_learning_rate

    
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        
        self.experience_replay      = ExperienceBufferContinuous(config.experience_replay_size)

        self.model_actor            = ModelActor.Model(self.state_shape, self.actions_count)
        self.model_actor_target     = ModelActor.Model(self.state_shape, self.actions_count)

        self.model_critic           = ModelCritic.Model(self.state_shape, self.actions_count)
        self.model_critic_target    = ModelCritic.Model(self.state_shape, self.actions_count)

        self.model_env              = ModelEnv.Model(self.state_shape, self.actions_count)

        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr= config.actor_learning_rate)
        
        self.optimizer_critic   = torch.optim.Adam(self.model_critic.parameters(), lr= config.critic_learning_rate)
        
        self.optimizer_env      = torch.optim.Adam(self.model_env.parameters(), lr= config.env_learning_rate)
                        
        self.state          = env.reset()

        self.iterations     = 0

        self.enable_training()

        self.env_loss       = 0.0
        self.im_entropy     = 0.0
        self.im_curiosity   = 0.0
        self.im             = 0.0
       
        


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

        #environment model state prediction
        state_predicted_t   = self.model_env(state_t, action_t)

        #env model loss
        env_loss = (state_next_t.detach() - state_predicted_t)**2
        env_loss = env_loss.mean()

        #update env model
        self.optimizer_env.zero_grad()
        env_loss.backward() 
        self.optimizer_env.step()


        im_entropy, im_curiosity    = self.intrinsics_motivation(state_t, action_t, state_next_t, state_predicted_t)
        im_entropy                  = torch.tanh(self.entropy_beta*im_entropy)
        im_curiosity                = torch.tanh(self.entropy_beta*im_curiosity)
        intrinsics_motivation_t     = im_entropy + im_curiosity
           

        action_next_t       = self.model_actor_target.forward(state_next_t).detach()
        value_next_t        = self.model_critic_target.forward(state_next_t, action_next_t).detach()
 
        #target value, Q-learning
        value_target    = reward_t + intrinsics_motivation_t + self.gamma*done_t*value_next_t
        value_predicted = self.model_critic.forward(state_t, action_t)

        #critic loss
        critic_loss     = ((value_target - value_predicted)**2)
        critic_loss     = critic_loss.mean()
     
        #update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward() 
        self.optimizer_critic.step()

        #actor loss
        actor_loss      = -self.model_critic.forward(state_t, self.model_actor.forward(state_t))
        actor_loss      = actor_loss.mean()

        #update actor
        self.optimizer_actor.zero_grad()       
        actor_loss.backward()
        self.optimizer_actor.step()

        #update target networks 
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
       
        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)

        #log smoothed intrinsics motivsation values
        k = 0.02
        self.env_loss       = (1.0 - k)*self.env_loss       + k*env_loss.mean().detach().to("cpu").numpy()
        self.im_entropy     = (1.0 - k)*self.im_entropy     + k*im_entropy.mean().detach().to("cpu").numpy()
        self.im_curiosity   = (1.0 - k)*self.im_curiosity   + k*im_curiosity.mean().detach().to("cpu").numpy()
        self.im             = (1.0 - k)*self.im             + k*intrinsics_motivation_t.mean().detach().to("cpu").numpy()

    def save(self, save_path):
        self.model_critic.save(save_path)
        self.model_actor.save(save_path)
        self.model_env.save(save_path)
       
    def load(self, load_path):
        self.model_critic.load(load_path)
        self.model_actor.load(load_path)
        self.model_env.load(load_path)


    def get_log(self):
        result = "" 
        result+= str(round(self.env_loss, 5)) + " "
        result+= str(round(self.im_entropy, 5)) + " "
        result+= str(round(self.im_curiosity, 5)) + " "
        result+= str(round(self.im, 5)) + " "
        
        return result
      

    def _sample_action(self, state_t, epsilon):
        action_t    = self.model_actor(state_t)
        action_t    = action_t + epsilon*torch.randn(action_t.shape).to(self.model_actor.device)
        action_t    = action_t.clamp(-1.0, 1.0)

        action_np   = action_t.detach().to("cpu").numpy()

        return action_t, action_np

    '''
    compute imagined states

    states_t : initial (root states) in shape = (batch_size, ) + state_shape
    epsilon  : epsilon greedy factor, 1 = total random, 0 = total deterministic

    returns imagined states in shape
    shape = (batch_size, imagination_rollouts, ) + state_shape


    f.e. result[5] is imagined states from root state states_t[5]
    result[5][0] correspond to rollout 0 - imagined with action 0
    result[5][1] correspond to rollout 1 - imagined with action 1
    result[5][2] correspond to rollout 1 - imagined with action 2
    '''
    def _process_imagination(self, states_t, epsilon):

        self.model_actor.eval()
        self.model_critic.eval()
        self.model_env.eval()

        batch_size  = states_t.shape[0]

        states_imagined_t      = torch.zeros((self.imagination_rollouts, batch_size, ) + self.state_shape ).to(self.model_env.device)

        for r in range(self.imagination_rollouts):
            states_imagined_t[r] = states_t.clone()

        '''
        reshape, to create one huge batch - much more faster
        shape = (imagination_rollouts*batch_size, state_shape)
        '''
        states_imagined_t = states_imagined_t.reshape((self.imagination_rollouts*batch_size, ) + self.state_shape)

        for s in range(self.imagination_steps):
            action_t, _             = self._sample_action(states_imagined_t, epsilon)
            states_imagined_next_t  = self.model_env(states_imagined_t, action_t)
            states_imagined_t       = states_imagined_next_t.clone()

        '''
        reshape back
        shape = (imagination_rollouts, batch_size, state_shape)
        '''
        states_imagined_t = states_imagined_t.reshape((self.imagination_rollouts, batch_size, ) + self.state_shape)

        '''
        swap axis to have batch first
        shape = (batch_size, imagination_rollouts, state_shape)
        '''
        states_imagined_t = states_imagined_t.transpose(1, 0)


        self.model_actor.train()
        self.model_critic.train()
        self.model_env.train()

        return states_imagined_t

    

    def _compute_entropy(self, states_t, states_initial_t):
        batch_size  = states_t.shape[0]
        result      = torch.zeros(batch_size).to(self.model_env.device)

        for b in range(batch_size):
            s_dif       = states_t[b] - states_initial_t[b]
            flatten     = s_dif.view(s_dif.size(0), -1)
            
            result[b]   = torch.var(flatten, dim=0).mean()

        return result


    def intrinsics_motivation(self, state_t, action_t, state_next_t, state_predicted_t):
        #compute imagined states, use state_t as initial root state
        states_imagined_t   = self._process_imagination(state_t, self.epsilon)
        
        #compute entropy of imagined states
        im_entropy           = self._compute_entropy(states_imagined_t.detach(), state_t)
       
        #compute curiosity
        im_curiosity         = ((state_next_t - state_predicted_t)**2).mean(dim = 1).detach()
        

        return im_entropy, im_curiosity


      