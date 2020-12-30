import libs_common.decay

class Config(): 

    def __init__(self):
        self.gamma                  = 0.99
        self.update_frequency       = 4
        self.target_update          = 1000

        self.batch_size             = 64 
        self.learning_rate_dqn      = 0.0002
        self.learning_rate_forward  = 0.001

        self.beta                   = 0.1
                 
        self.exploration            = libs_common.decay.Const(0.05, 0.05)        
        self.experience_replay_size = 16384
 
