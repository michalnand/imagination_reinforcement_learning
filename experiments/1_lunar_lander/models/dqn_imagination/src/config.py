import libs_common.decay

class Config():

    def __init__(self):        
        self.gamma                          = 0.99

        self.model_actor_learning_rate      = 0.0001
        self.model_features_learning_rate   = 0.0002
        self.model_forward_learning_rate    = 0.0002
        self.model_reward_learning_rate     = 0.0002


        self.tau                            = 0.001

        self.batch_size                     = 64
        self.trajectory_length              = 16

        self.exploration                    = libs_common.decay.Const(0.02, 0.02)

        self.experience_replay_size         = 16384

