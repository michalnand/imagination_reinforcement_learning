import libs_common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.99
        self.learning_rate          = 0.0002
        self.tau                    = 0.001
        self.bellman_steps          = 3

        self.batch_size             = 32
        self.update_frequency       = 4

        self.exploration   = libs_common.decay.Const(0.1, 0.1)

        self.experience_replay_size = 16384