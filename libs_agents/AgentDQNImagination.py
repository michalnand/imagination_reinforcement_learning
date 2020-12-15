import numpy
import torch
from .ExperienceBuffer import *


class AgentDQNImagination():
    def __init__(self, env, Modeldqn, ModelIM, Config):
        self.env    = env
        config      = Config.Config()

        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma
        
        self.target_update      = config.target_update
        self.update_frequency   = config.update_frequency 

        self.rollouts               = config.rollouts
        self.entropy_beta           = config.entropy_beta
        self.curiosity_beta         = config.curiosity_beta

        self.state_shape        = self.env.observation_space.shape
        self.actions_count      = self.env.action_space.n

        self.experience_replay  = ExperienceBuffer(config.experience_replay_size)

        self.model_dqn             = Modeldqn.Model(self.state_shape, self.actions_count)
        self.model_dqn_target      = Modeldqn.Model(self.state_shape, self.actions_count)
        for target_param, param in zip(self.model_dqn_target.parameters(), self.model_dqn.parameters()):
            target_param.data.copy_(param.data)

        self.model_im       = ModelIM.Model(self.state_shape, self.actions_count)

        self.optimizer_dqn  = torch.optim.Adam(self.model_dqn.parameters(), lr=config.learning_rate_dqn)
        self.optimizer_im   = torch.optim.Adam(self.model_im.parameters(),  lr=config.learning_rate_im)
        
        self.state          = env.reset()
        self.iterations     = 0
        self.enable_training()

        self.loss_dqn       = 0.0
        self.loss_im   = 0.0
        self.entropy        = 0.0
        self.curiosity      = 0.0
        
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
             
        state_t     = torch.from_numpy(self.state).to(self.model_dqn.device).unsqueeze(0).float()
        
        action_idx_np, action_one_hot,  = self._sample_action(state_t, self.epsilon)

        action = action_idx_np[0]

        state_new, reward, done, self.info = self.env.step(action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, action, reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self._training()
 
            if self.iterations%self.target_update == 0:
                self.model_dqn_target.load_state_dict(self.model_dqn.state_dict())

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
        sample random minibatch
        '''
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_dqn.device)

        #intrinsic motivation
        curiosity_t = self._curiosity(state_t, state_next_t, action_t).detach()
        entropy_t   = self._entropy(state_t, self.epsilon).detach()
    
        '''
        predict next state, and compute forward model loss 
        note : forward model learns next features
        '''
        action_one_hot_t    = self._action_one_hot(action_t)
        
        action_predicted, features_state_next, features_state_next_predicted = self.model_im(state_t, state_next_t, action_one_hot_t)

        #inverse model, and forward model loss
        loss_im = 0.1*((action_one_hot_t.detach()   - action_predicted)**2).mean()
        loss_im+= ((features_state_next.detach()    - features_state_next_predicted)**2).mean()

        self.optimizer_im.zero_grad()
        loss_im.backward() 
        self.optimizer_im.step()

        '''
        predict Q-values using features, and dqn model
        '''
        q_predicted      = self.model_dqn(state_t)
        q_predicted_next = self.model_dqn_target(state_next_t)

        '''
        compute loss for Q values, using Q-learning
        '''
        #compute target
        q_target         = q_predicted.clone()
        for j in range(self.batch_size): 
            action_idx              = action_t[j]
            q_target[j][action_idx] = entropy_t[j] + curiosity_t[j] + reward_t[j] + self.gamma*torch.max(q_predicted_next[j])*(1- done_t[j])
 
        #compute dqn loss
        loss_dqn  = ((q_target.detach() - q_predicted)**2)
        loss_dqn  = loss_dqn.mean() 

        #train dqn
        self.optimizer_dqn.zero_grad()
        
        loss_dqn.backward() 
        for param in self.model_dqn.parameters():
            param.grad.data.clamp_(-10.0, 10.0)

        self.optimizer_dqn.step()

        '''
        log some stats, using exponential smoothing
        '''
        k = 0.02

        self.loss_im        = (1.0 - k)*self.loss_im   + k*loss_im.detach().to("cpu").numpy()
        self.loss_dqn       = (1.0 - k)*self.loss_dqn     + k*loss_dqn.detach().to("cpu").numpy()
        self.entropy        = (1.0 - k)*self.entropy        + k*entropy_t.mean().detach().to("cpu").numpy()
        self.curiosity      = (1.0 - k)*self.curiosity      + k*curiosity_t.mean().detach().to("cpu").numpy()

        #print(self.loss_im, self.loss_dqn, self.entropy, self.curiosity, "\n\n")

    def _sample_action(self, state_t, epsilon):

        batch_size = state_t.shape[0]

        q_values_t          = self.model_dqn(state_t).to("cpu")

        #best actions indices
        q_max_indices_t     = torch.argmax(q_values_t, dim = 1)

        #random actions indices
        q_random_indices_t  = torch.randint(self.actions_count, (batch_size,))

        #create mask, which actions will be from q_random_indices_t and which from q_max_indices_t
        select_random_mask_t= torch.tensor((torch.rand(batch_size) < epsilon).clone(), dtype = int)

        #apply mask
        action_idx_t    = select_random_mask_t*q_random_indices_t + (1 - select_random_mask_t)*q_max_indices_t
        action_idx_t    = torch.tensor(action_idx_t, dtype=int)

        action_idx_t    = torch.tensor(q_max_indices_t, dtype=int)

        #create one hot encoding
        action_one_hot_t = self._action_one_hot(action_idx_t)

        #numpy result
        action_idx_np       = action_idx_t.detach().to("cpu").numpy().astype(dtype=int)

        return action_idx_np, action_one_hot_t

    def _action_one_hot(self, action_idx_t):
        batch_size = action_idx_t.shape[0]

        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model_dqn.device)

        return action_one_hot_t

    def _curiosity(self, state_t, state_next_t, action_t):
        action_one_hot_t    = self._action_one_hot(action_t)
        
        _, features_state_next, features_state_next_predicted = self.model_im(state_t, state_next_t, action_one_hot_t)

        dif             = features_state_next - features_state_next_predicted

        curiosity_t     = (dif**2).view(dif.size(0), -1).mean(dim=1)
        curiosity_t     = torch.tanh(self.curiosity_beta*curiosity_t)

        return curiosity_t


    def _entropy(self, state_t, epsilon):
        #fill initial state
        states_initial_t  = torch.zeros((self.rollouts, self.batch_size) + self.state_shape).to(state_t.device)
        for i in range(self.rollouts):
            states_initial_t[i] = state_t.clone()

        #sample actions
        #for loop in rollouts is necessary when noisy nets used
        actions_t  = torch.zeros((self.rollouts, self.batch_size, self.actions_count)).to(state_t.device)
        for i in range(self.rollouts):
            _, actions_t_ = self._sample_action(state_t, epsilon)
            actions_t[i]  = actions_t_.clone()

        #create one big batch for faster run
        states_initial_t = states_initial_t.reshape((self.rollouts*self.batch_size, ) + self.state_shape)

        actions_t = actions_t.reshape((self.rollouts*self.batch_size, self.actions_count))

        #compute predicted state
        _, features_next_predicted_t = self.model_im.eval_next_features(states_initial_t, actions_t)

        #reshape back, to batch, rollouts, state.shape
        features_next_predicted_t = features_next_predicted_t.reshape((self.rollouts, self.batch_size, features_next_predicted_t.shape[1]))
        features_next_predicted_t = features_next_predicted_t.transpose(0, 1)
        features_next_predicted_t = features_next_predicted_t.view(features_next_predicted_t.size(0), features_next_predicted_t.size(1), -1)

        #compute entropy from variance, accross rollout dimension
        entropy_t   = torch.std(features_next_predicted_t, dim = 1).mean(dim = 1)
 
        entropy_t   = torch.tanh(self.entropy_beta*entropy_t)

        return entropy_t


    def save(self, save_path):
        self.model_im.save(save_path)
        self.model_dqn.save(save_path)

    def load(self, load_path):
        self.model_im.load(load_path)
        self.model_dqn.load(load_path)

    def get_log(self):
        result = "" 
        result+= str(round(self.loss_im, 7)) + " "
        result+= str(round(self.loss_dqn, 7)) + " "
        result+= str(round(self.entropy, 7)) + " "
        result+= str(round(self.curiosity, 7)) + " "
        return result