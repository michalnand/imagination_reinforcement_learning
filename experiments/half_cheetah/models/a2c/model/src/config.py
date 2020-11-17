import libs_common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.99
        self.learning_rate          = 0.0002
        self.entropy_beta           = 0.01

        self.batch_size             = 1024
        self.episodes_to_train      = 1

