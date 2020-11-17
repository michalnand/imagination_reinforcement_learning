import libs_common.decay

class Config():

    def __init__(self):        
        self.batch_size         = 64
        self.gamma              = 0.99
        self.update_frequency   = 1
        self.tau                = 0.001

        self.exploration            = libs_common.decay.Exponential(0.99999, 1.0, 0.2, 0.2)
        self.experience_replay_size = 16384

        self.imagination_rollouts   = 16
        self.imagination_steps      = 1

        self.entropy_beta           = 1.0
        self.curiosity_beta         = 1.0
        
        self.critic_learning_rate   = 0.0002
        self.actor_learning_rate    = 0.0001
        
        self.env_learning_rate      = 0.0002
        


