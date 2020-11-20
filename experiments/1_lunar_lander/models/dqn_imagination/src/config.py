import libs_common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.99
        self.target_update          = 1000

        self.batch_size             = 32
        self.update_frequency       = 4


        self.learning_rate_features = 0.0002
        self.learning_rate_forward  = 0.0002
        self.learning_rate_reward   = 0.0002
        self.learning_rate_actor    = 0.0002

        self.exploration                    = libs_common.decay.Const(0.02, 0.02)
        self.experience_replay_size         = 16384

