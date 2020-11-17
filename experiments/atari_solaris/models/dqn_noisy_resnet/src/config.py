import libs_common.decay

class Config(): 

    def __init__(self):
        self.gamma                  = 0.99
        self.update_frequency       = 4
        self.target_update          = 10000

        self.batch_size             = 32 
        self.learning_rate          = 0.0001
        self.bellman_steps          = 1
                 
        self.exploration            = libs_common.decay.Const(0.05, 0.05)        
        self.experience_replay_size = 32768
 
