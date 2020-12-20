import numpy
from scipy import stats

class RLStatsCompute:
    def __init__(self, files_list, destination_file = "result_stats.log", load_extended = False):
        self.load_extended = load_extended

        data = self.load_files(files_list)
        self.process_stats(data, destination_file)


  

    def load_files(self, files_list):
        data      = []
      
        for f in files_list:
            print("loading ", f)
            data_ = numpy.loadtxt(f, unpack = True)
            data.append(data_)

        data      = numpy.array(data)
      
        return data
        

    def compute_stats(self, data, confidence = 0.95):

        '''
        count   = data.shape[0]
        alpha   = 1.0 - confidence
        t       = stats.t.ppf(1.0 - alpha/2.0, count - 1)  
    
        mean = numpy.mean(data, axis = 0)
        std  = numpy.std(data, ddof=1, axis = 0)

        lower = mean - (t * std/ numpy.sqrt(count))
        upper = mean + (t * std/ numpy.sqrt(count))
        '''

        n       = data.shape[0]

        mean    = numpy.mean(data, axis = 0)
        std     = numpy.std(data, axis = 0)
        se      = stats.sem(data, axis=0)
        h       = se * stats.t.ppf((1 + confidence) / 2., n-1)

        lower = mean - h
        upper = mean + h

        return mean, std, lower, upper


    def process_stats(self, data, file_name):

        data = numpy.rollaxis(data, 1, 0)
    
        self.iterations     = data[0][0]
        games               = data[1]
        total_score         = data[3]
        episode_score       = data[4]


        if data.shape[0] == 14:
            self.forward_loss_mean, self.forward_loss_std, self.forward_loss_lower, self.forward_loss_upper = self.compute_stats(data[9])
            self.entropy_mean, self.entropy_std, self.entropy_lower, self.entropy_upper                     = self.compute_stats(data[12])
            self.curiosity_mean, self.curiosity_std, self.curiosity_lower, self.curiosity_upper             = self.compute_stats(data[13])

        if data.shape[0] == 13:
            self.forward_loss_mean, self.forward_loss_std, self.forward_loss_lower, self.forward_loss_upper = self.compute_stats(data[9])
            self.entropy_mean, self.entropy_std, self.entropy_lower, self.entropy_upper                     = self.compute_stats(data[11])
            self.curiosity_mean, self.curiosity_std, self.curiosity_lower, self.curiosity_upper             = self.compute_stats(data[12])

        self.per_iteration_score = total_score/self.iterations

        
        self.games_mean, games_std, games_lower, games_upper         = self.compute_stats(games)

        self.total_mean, self.total_std, self.total_lower, self.total_upper         = self.compute_stats(total_score)
        self.per_iteration_mean, self.per_iteration_std, self.per_iteration_lower, self.per_iteration_upper         = self.compute_stats(self.per_iteration_score)
        self.episode_mean, self.episode_std, self.episode_lower, self.episode_upper = self.compute_stats(episode_score)

        decimal_places = 4
        f = open(file_name, "w")
        for i in range(len(self.iterations)):
            result_str = ""
            result_str+= str(self.iterations[i]) + " "
            result_str+= str(self.games_mean[i])      + " "
            
            result_str+= str(round(self.total_mean[i], decimal_places)) + " "
            result_str+= str(round(self.total_lower[i], decimal_places)) + " "
            result_str+= str(round(self.total_upper[i], decimal_places)) + " "

            result_str+= str(round(self.per_iteration_mean[i], decimal_places)) + " "
            result_str+= str(round(self.per_iteration_lower[i], decimal_places)) + " "
            result_str+= str(round(self.per_iteration_upper[i], decimal_places)) + " "
    
            result_str+= str(round(self.episode_mean[i], decimal_places)) + " "
            result_str+= str(round(self.episode_lower[i], decimal_places)) + " "
            result_str+= str(round(self.episode_upper[i], decimal_places)) + " "
            result_str+= "\n" 

            f.write(result_str)

        f.close()
