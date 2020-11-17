import numpy
import torch

from torch.distributions import Categorical

from .PolicyBuffer import *

class AgentPPO():
    def __init__(self, env, Model, Config):
        self.env = env

        config = Config.Config()
 
        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip
        self.batch_size         = config.batch_size
        self.episodes_to_train  = config.episodes_to_train
        self.training_epochs    = config.training_epochs

        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.policy_buffer = PolicyBuffer(self.batch_size, self.state_shape, self.actions_count, self.model.device)

        self.state = env.reset()

        self.enable_training()
        
        self.iterations = 0
        self.passed_episodes = 0



    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

   
    def main(self):
        state_t   = torch.tensor(self.state, dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
        
        logits, value   = self.model.forward(state_t)

        action = self._sample_action(logits)
            
        self.state, reward, done, _ = self.env.step(action)
        
        if self.enabled_training:
            self.policy_buffer.add(state_t.squeeze(0), logits.squeeze(0), value.squeeze(0), action, reward, done)

            if done:
                self.passed_episodes+= 1

            if self.passed_episodes >= self.episodes_to_train or self.policy_buffer.is_full():

                self.policy_buffer.cut_zeros()
                
                for e in range(self.training_epochs):
                    self._train()

                self.policy_buffer.clear()   
                
                self.passed_episodes = 0

        if done:
            self.state = self.env.reset()

            if hasattr(self.model, "reset"):
                self.model.reset()

        self.iterations+= 1

        return reward, done
    

    
     
    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    
   

    def _sample_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t.item()

    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t  
    
    def _train(self):
        self.policy_buffer.compute_q_values(self.gamma) 
        
        loss = self._compute_loss()

        self.optimizer.zero_grad()        
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step() 

        


    def _compute_loss(self):
        
        target_values_b = self.policy_buffer.q_values_b.detach()


        probs_old     = torch.nn.functional.softmax(self.policy_buffer.logits_b, dim = 1).detach()
        log_probs_old = torch.nn.functional.log_softmax(self.policy_buffer.logits_b, dim = 1).detach()

        state_t = self.policy_buffer.states_b.detach().clone()

        logits, values   = self.model.forward(state_t)

        probs     = torch.nn.functional.softmax(logits, dim = 1)
        log_probs = torch.nn.functional.log_softmax(logits, dim = 1)

        actions_b = self._sample_actions(logits)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (target_values_b - values)**2
        loss_value = loss_value.mean()

        ''' 
        compute actor loss 
        '''
        log_probs_      = log_probs[range(len(log_probs)), self.policy_buffer.actions_b]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), self.policy_buffer.actions_b]
                        
        # Finding Surrogate Loss:
        advantage   = (target_values_b - values).detach()

        
        ratios = torch.exp(log_probs_ - log_probs_old_)
        surr1 = ratios*advantage
        surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantage
        
        loss_policy = -torch.min(surr1, surr2) 
        loss_policy = loss_policy.mean()
    

        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs*log_probs).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = loss_value + loss_policy + loss_entropy

        return loss




