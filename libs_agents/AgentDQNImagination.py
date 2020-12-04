import numpy
import torch
from .ExperienceBuffer import *


class AgentDQNImagination():
    def __init__(self, env, ModelFeatures, ModelActor, ModelForward, Config):
        self.env    = env
        config      = Config.Config()

        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma
        
        self.target_update      = config.target_update
        self.update_frequency   = config.update_frequency 

        self.trajectory_length      = config.trajectory_length
        self.entropy_beta           = config.entropy_beta
        self.curiosity_beta         = config.curiosity_beta

        self.state_shape        = self.env.observation_space.shape
        self.actions_count      = self.env.action_space.n

        self.experience_replay  = ExperienceBuffer(config.experience_replay_size)

        self.model_features             = ModelFeatures.Model(self.state_shape)
        self.model_features_target      = ModelFeatures.Model(self.state_shape)
        for target_param, param in zip(self.model_features_target.parameters(), self.model_features.parameters()):
            target_param.data.copy_(param.data)

        self.features_shape          = self.model_features.features_shape

        self.model_actor             = ModelActor.Model(self.features_shape, self.actions_count)
        self.model_actor_target      = ModelActor.Model(self.features_shape, self.actions_count)
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)


        self.model_forward      = ModelForward.Model(self.features_shape,    self.actions_count)

        self.optimizer_features = torch.optim.Adam(self.model_features.parameters(),    lr=config.learning_rate_features)
        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(),       lr=config.learning_rate_actor)
        self.optimizer_forward  = torch.optim.Adam(self.model_forward.parameters(),     lr=config.learning_rate_forward)
        
       
        self.state          = env.reset()
        self.iterations     = 0
        self.enable_training()


        self.loss_actor     = 0.0
        self.loss_forward   = 0.0
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
             
        state_t     = torch.from_numpy(self.state).to(self.model_actor.device).unsqueeze(0).float()

        features    = self.model_features(state_t)
        
        action_idx_np, action_one_hot,  = self._sample_action(features, self.epsilon)

        action = action_idx_np[0]

        state_new, reward, done, self.info = self.env.step(action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, action, reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self._training()
 
            if self.iterations%self.target_update == 0:
                self.model_features_target.load_state_dict(self.model_features.state_dict())
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
        sample random minibatch
        '''
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model_actor.device)

        #intrinsic motivation
        state_seq_t, action_seq_t, state_next_seq_t = self.experience_replay.sample_sequence(self.batch_size, self.trajectory_length, self.model_features.device, True)

        features_imagined_t    =  self._process_imagination(state_t, action_seq_t)

        entropy_t, curiosity_t = self._intrinsic_motivation(features_imagined_t, state_next_seq_t)
      
        '''
        predict features for state and next state
        '''
        features_t          = self.model_features(state_t)
        features_next_t     = self.model_features_target(state_next_t)

        '''
        predict next features, and compute forward model loss 
        note : forward model learns next features
        '''
        action_one_hot_t        = self._action_one_hot(action_t)
        features_predicted_t    = self.model_forward(features_t, action_one_hot_t)

        loss_forward   = ((features_next_t.detach() - features_predicted_t)**2)
        loss_forward   = loss_forward.mean() 

        '''
        predict Q-values using features, and actor model
        '''
        q_predicted      = self.model_actor(features_t)
        q_predicted_next = self.model_actor_target(features_next_t)

        '''
        compute loss for Q values, using Q-learning
        '''
        #compute target
        q_target         = q_predicted.clone()
        for j in range(self.batch_size): 
            action_idx              = action_t[j]
            q_target[j][action_idx] = entropy_t[j] + curiosity_t[j] + reward_t[j] + self.gamma*torch.max(q_predicted_next[j])*(1- done_t[j])
 
        #compute loss
        loss_actor  = ((q_target.detach() - q_predicted)**2)
        loss_actor  = loss_actor.mean() 

        '''
        compute final loss, gradients clamp and train
        '''
        loss = loss_actor + loss_forward

        self.optimizer_features.zero_grad()
        self.optimizer_actor.zero_grad()
        self.optimizer_forward.zero_grad()

        loss.backward() 

        for param in self.model_features.parameters():
            param.grad.data.clamp_(-10.0, 10.0)

        for param in self.model_actor.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        
        for param in self.model_forward.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        

        self.optimizer_features.step()
        self.optimizer_actor.step()
        self.optimizer_forward.step()

        '''
        log some stats, using exponential smoothing
        '''
        k = 0.02

        self.loss_forward   = (1.0 - k)*self.loss_forward   + k*loss_forward.detach().to("cpu").numpy()
        self.loss_actor     = (1.0 - k)*self.loss_actor     + k*loss_actor.detach().to("cpu").numpy()
        self.entropy        = (1.0 - k)*self.entropy        + k*entropy_t.mean().detach().to("cpu").numpy()
        self.curiosity      = (1.0 - k)*self.curiosity      + k*curiosity_t.mean().detach().to("cpu").numpy()

        print(self.loss_forward, self.loss_actor, self.entropy, self.curiosity, "\n\n")

    def _sample_action(self, features_t, epsilon):

        batch_size = features_t.shape[0]

        q_values_t          = self.model_actor(features_t).to("cpu")

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
        action_one_hot_t = action_one_hot_t.to(self.model_actor.device)

        return action_one_hot_t

    def _intrinsic_motivation(self, features_imagined_seq_t, state_target_seq_t):

        #reshape state to one huge batch
        shape   = (self.batch_size*self.trajectory_length,  ) + self.state_shape
        x       = torch.reshape(state_target_seq_t, (self.batch_size*self.trajectory_length,  ) + self.state_shape)

        #obtain target features
        features_target_seq_t = self.model_features(x)

        features_target_seq_t = torch.reshape(features_target_seq_t, (self.batch_size, self.trajectory_length,  ) + self.features_shape)

        #difference : target and predicted
        dif     = features_target_seq_t - features_imagined_seq_t

        #compute entropy, variance
        if self.trajectory_length > 1:
            entropy_t   = torch.std(dif, dim = 1).view(dif.size(0), -1).mean(dim = 1)
        else:
            entropy_t   = torch.zeros(dif.shape).to(dif.device)

        #compute curiosity, mean square
        curiosity_t = (dif**2).view(dif.size(0), -1).mean(dim=1)

        #squashing and scaling
        entropy_t       = torch.tanh(self.entropy_beta*entropy_t).detach()
        curiosity_t     = torch.tanh(self.curiosity_beta*curiosity_t).detach()

        return entropy_t, curiosity_t


    def _process_imagination(self, state_initial_t, action_seq_t):
        action_seq_t_           = action_seq_t.transpose(0, 1)

        features_imagined_seq_t = torch.zeros((self.trajectory_length, self.batch_size) + self.features_shape).to(state_initial_t.device)

        features_imagined_seq_t[0] = self.model_features(state_initial_t).detach()

        for n in range(self.trajectory_length - 1):
            action_one_hot_t              = self._action_one_hot(action_seq_t_[n])
            features_imagined_seq_t[n+1]  = self.model_forward(features_imagined_seq_t[n], action_one_hot_t)

        features_imagined_seq_t = features_imagined_seq_t.transpose(0, 1)

        return features_imagined_seq_t

    def save(self, save_path):
        self.model_features.save(save_path)
        self.model_forward.save(save_path)
        self.model_actor.save(save_path)

    def load(self, load_path):
        self.model_features.load(load_path)
        self.model_forward.load(load_path)
        self.model_actor.load(load_path)

    def get_log(self):
        result = "" 
        result+= str(round(self.loss_forward, 7)) + " "
        result+= str(round(self.loss_actor, 7)) + " "
        result+= str(round(self.entropy, 7)) + " "
        result+= str(round(self.curiosity, 7)) + " "
        result+= str(round(self.entropy, 7)) + " "
        result+= str(round(self.curiosity, 7)) + " "

        return result