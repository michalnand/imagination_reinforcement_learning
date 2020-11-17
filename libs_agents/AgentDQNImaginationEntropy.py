import numpy
import torch
from .ExperienceBuffer import *

import cv2
import time


class AgentDQNImaginationEntropy():
    def __init__(self, env, Model, ModelEnv, Config):
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
        
       
                 
        self.imagination_rollouts   = config.imagination_rollouts
        self.imagination_steps      = config.imagination_steps

        self.entropy_beta           = config.entropy_beta
        self.curiosity_beta         = config.curiosity_beta

        self.exploration            = config.exploration



        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n


        self.experience_replay = ExperienceBuffer(config.experience_replay_size, self.bellman_steps)

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_target   = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)


        self.model_env          = ModelEnv.Model(self.state_shape, self.actions_count)
        self.optimizer_env      = torch.optim.Adam(self.model_env.parameters(), lr= config.env_learning_rate)

    
        self.state    = env.reset()
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
    
    def main(self, show_activity = False):
        if self.enabled_training:
            self.exploration.process()
            self.epsilon = self.exploration.get()
        else:
            self.epsilon = self.exploration.get_testing()

        state_t     = torch.from_numpy(self.state).to(self.model.device).unsqueeze(0).float()
        
        q_values    = self.model(state_t)
        q_values    = q_values.squeeze(0).detach().to("cpu").numpy()
 
        action_idx_np, _ = self._sample_action(state_t, self.epsilon)

        self.action = action_idx_np[0]

        state_new, self.reward, done, self.info = self.env.step(self.action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, self.action, self.reward, done)


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

    def _tensor_to_numpy(self, x, spacing = 4):
        x_np    = x.detach().to("cpu").numpy()
        height  = x_np.shape[2]
        width   = x_np.shape[3]

        rows    = 4
        cols    = 4
        
        result  = numpy.zeros((rows*(height+spacing), cols*(width+spacing)))
        for row in range(rows):
            for col in range(cols):
                idx = col + row*cols
                result[row*width + row*spacing:(row+1)*width + row*spacing, col*height + col*spacing:(col+1)*height + col*spacing] = x_np[idx][0]

        k = 1.0/(result.max() - result.min())
        q = 1.0 - k*result.max()   
        result = k*result + q

        return result


    def _show_entropy_activity(self, state_t, size = 256):
        if self.iterations%50 != 0:
            return 

        state_np  = state_t[0][0].detach().to("cpu").numpy()
        state_np  = cv2.resize(state_np, (size, size), interpolation = cv2.INTER_AREA)

        states_imagined_t   = self._process_imagination(state_t, self.epsilon).squeeze(0)

        states_dif_t        = state_t.squeeze(0) - states_imagined_t
 
        entropy_t           = torch.std(states_dif_t, dim=0).mean(dim=0)

        flatten     = torch.abs(states_dif_t.view(states_dif_t.size(0), -1))
        

        states_imagined_np  = self._tensor_to_numpy(states_imagined_t)
        states_imagined_np  = cv2.resize(states_imagined_np, (size, size), interpolation = cv2.INTER_AREA)

        #states_dif_np       = self._tensor_to_numpy(states_dif_t**2)
        #states_dif_np       = cv2.resize(states_dif_np, (size, size), interpolation = cv2.INTER_AREA)
        
        entropy_np          = entropy_t.detach().to("cpu").numpy()
        k = 1.0/(entropy_np.max() - entropy_np.min())
        q = 1.0 - k*entropy_np.max()   
        entropy_np = k*entropy_np + q

        entropy_np = cv2.resize(entropy_np, (size, size), interpolation = cv2.INTER_AREA)


        image   =  numpy.hstack((state_np, states_imagined_np, entropy_np))

        cv2.imshow('states_imagined_np', image)
        cv2.waitKey(1)
       
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model.device)
        
        #environment model state prediction
        action_one_hot_t    = self._one_hot_encoding(action_t)
        state_predicted_t   = self.model_env(state_t, action_one_hot_t)

        #env model loss
        env_loss = (state_next_t - state_predicted_t)**2
        env_loss = env_loss.mean()

        #update env model
        self.optimizer_env.zero_grad()
        env_loss.backward() 
        self.optimizer_env.step()



        im_entropy, im_curiosity    = self.intrinsics_motivation(state_t, action_t, state_next_t, state_predicted_t)
        intrinsics_motivation_t     = torch.tanh(self.entropy_beta*im_entropy) + torch.tanh(self.curiosity_beta*im_curiosity)

        
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
            q_target[j][action_idx]   = reward_sum + (gamma_**self.bellman_steps)*torch.max(q_predicted_next[j]) + intrinsics_motivation_t[j]
 
        #train DQN model
        loss = ((q_target.detach() - q_predicted)**2)
        loss  = loss.mean() 

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()


        k = 0.02
        self.env_loss       = (1.0 - k)*self.env_loss       + k*env_loss.mean().detach().to("cpu").numpy()
        self.im_entropy     = (1.0 - k)*self.im_entropy     + k*im_entropy.mean().detach().to("cpu").numpy()
        self.im_curiosity   = (1.0 - k)*self.im_curiosity   + k*im_curiosity.mean().detach().to("cpu").numpy()
        self.im             = (1.0 - k)*self.im             + k*intrinsics_motivation_t.mean().detach().to("cpu").numpy()


    
    def save(self, save_path):
        self.model.save(save_path)
        self.model_env.save(save_path)

    def load(self, load_path):
        self.model.load(load_path)
        self.model_env.load(load_path)
    

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


    '''
    compute imagined states

    states_t : initial (root states) in shape = (batch_size, ) + state_shape
    epsilon  : epsilon greedy factor, 1 = total random, 0 = total deterministic

    returns imagined states in shape
    shape = (batch_size, imagination_rollouts, ) + state_shape


    f.e. result[5] is imagined states from root state starting as states_t[5]
    result[5][0] correspond to rollout 0 - imagined with action 0
    result[5][1] correspond to rollout 1 - imagined with action 1
    result[5][2] correspond to rollout 1 - imagined with action 2
    '''
    def _process_imagination(self, states_t, epsilon):
        self.model.eval()
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
            _, action_t             = self._sample_action(states_imagined_t, epsilon)
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
       
        self.model.train()
        self.model_env.train()

        return states_imagined_t

    def _compute_entropy(self, states_t, states_initial_t, threshold):
        batch_size  = states_t.shape[0]
        result      = torch.zeros(batch_size).to(self.model_env.device)

        for b in range(batch_size):
            s_dif       = states_t[b] - states_initial_t[b]
            
            flatten     = s_dif.view(s_dif.size(0), -1)

            variance    = torch.std(flatten, dim=0)
            shrink      = variance[variance > threshold]
             
            if shrink.shape[0] > 0:
                result[b]   = shrink.mean()

        return result


    def _one_hot_encoding(self, action_t):
        batch_size          = action_t.shape[0]
        action_one_hot_t    = torch.zeros((batch_size, self.actions_count)).to(self.model.device)
         
        for b in range(batch_size):
            action_one_hot_t[b][action_t[b]] = 1.0

        return action_one_hot_t

    def get_log(self):
        result = ""
        result+= str(self.env_loss) + " "
        result+= str(self.im_entropy) + " "
        result+= str(self.im_curiosity) + " "
        result+= str(self.im) + " "

        return result

    def intrinsics_motivation(self, state_t, action_t, state_next_t, state_predicted_t, threshold = 0.01):
        
        #compute imagined states, use state_t as initial state
        states_imagined_t   = self._process_imagination(state_t, self.epsilon)
        
        #compute entropy of imagined states
        im_entropy           = self._compute_entropy(states_imagined_t.detach(), state_t, threshold)
       
        #compute curiosity
        state_dif    = ((state_next_t - state_predicted_t)**2).detach()
        flatten      = state_dif.view(state_dif.size(0), -1)    

        im_curiosity = torch.zeros(self.batch_size).to(self.model.device)

        for b in range(self.batch_size): 
            shrink  = flatten[b][flatten[b] > threshold]

            if shrink.shape[0] > 0:
                im_curiosity[b] = shrink.mean()

        
        return im_entropy, im_curiosity
