import libs_common.decay

class Config(): 

    def __init__(self):
        self.gamma                  = 0.99
        self.update_frequency       = 4
        self.target_update          = 10000

        self.batch_size             = 32 
                 
        self.exploration            = libs_common.decay.Const(0.05, 0.05)        
        self.experience_replay_size = 32768
 
        self.learning_rate_forward  = 0.0005
        self.learning_rate_dqn      = 0.0001

        
        self.rollouts               = 8
        self.entropy_beta           = 10.0
        self.curiosity_beta         = 10.0

