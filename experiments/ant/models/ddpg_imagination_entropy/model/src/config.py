import libs_common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.99
        self.tau                    = 0.001

        self.batch_size          = 64
        self.update_frequency    = 4

        self.exploration   = libs_common.decay.Const(0.05, 0.05)

        self.experience_replay_size = 200000


        self.learning_rate_features = 0.0002
        self.learning_rate_forward  = 0.0002
        self.learning_rate_critic   = 0.0002
        self.learning_rate_actor    = 0.0001 
      

        self.imagination_rollouts   = 16
        self.imagination_steps      = 1
        self.entropy_beta           = 10.0
        self.curiosity_beta         = 10.0
