import libs_common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.99
        self.tau                    = 0.001

        self.batch_size          = 64
        self.update_frequency    = 4

        #self.exploration   = libs_common.decay.Const(0.05, 0.05)
        self.exploration   = libs_common.decay.Linear(2000000, 0.3, 0.05, 0.05)

        self.experience_replay_size = 200000


        self.learning_rate_forward  = 0.0002
        self.learning_rate_critic   = 0.0002
        self.learning_rate_actor    = 0.0001 
      

        self.trajectory_length      = 16
        self.entropy_beta           = 1.0
        self.curiosity_beta         = 1.0
