import libs_common.decay

class Config(): 

    def __init__(self):
        self.gamma                  = 0.995
        self.update_frequency       = 4
        self.tau                    = 0.001

        self.batch_size             = 64 
        self.learning_rate          = 0.0001
        self.bellman_steps          = 1
                 
        self.exploration            = libs_common.decay.Const(0.02, 0.02)        
        self.experience_replay_size = 256 #500000
 
