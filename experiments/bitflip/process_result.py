import sys
sys.path.insert(0, '../../')

from libs_common.RLStatsCompute import *

import matplotlib.pyplot as plt

result_path = "./results/"

files = []
files.append("./models/dqn_baseline/result/result.log")
rl_stats_compute_dqn = RLStatsCompute(files, result_path + "dqn_baseline.log")

files = []
files.append("./models/dqn_hindsight/result/result.log")
rl_stats_compute_dqn_hindsight = RLStatsCompute(files, result_path + "dqn_hindsight.log") 

files = []
files.append("./models/dqn_curious_goals/result/result.log")
rl_stats_compute_dqn_curious_goals = RLStatsCompute(files, result_path + "dqn_curious_goals.log") 



plt.cla()
plt.ylabel("score")
plt.xlabel("episode")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_dqn.games_mean, rl_stats_compute_dqn.episode_mean, label="dqn baseline", color='blue')
plt.fill_between(rl_stats_compute_dqn.games_mean, rl_stats_compute_dqn.episode_lower, rl_stats_compute_dqn.episode_upper, color='blue', alpha=0.2)


plt.plot(rl_stats_compute_dqn_hindsight.games_mean, rl_stats_compute_dqn_hindsight.episode_mean, label="dqn hindsight", color='red')
plt.fill_between(rl_stats_compute_dqn_hindsight.games_mean, rl_stats_compute_dqn_hindsight.episode_lower, rl_stats_compute_dqn_hindsight.episode_upper, color='red', alpha=0.2)

plt.plot(rl_stats_compute_dqn_curious_goals.games_mean, rl_stats_compute_dqn_curious_goals.episode_mean, label="dqn curious goals", color='orange')
plt.fill_between(rl_stats_compute_dqn_curious_goals.games_mean, rl_stats_compute_dqn_curious_goals.episode_lower, rl_stats_compute_dqn_curious_goals.episode_upper, color='orange', alpha=0.2)


plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "score_per_episode.png", dpi = 300)

 


plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_dqn.iterations, rl_stats_compute_dqn.episode_mean, label="dqn baseline", color='blue')
plt.fill_between(rl_stats_compute_dqn.iterations, rl_stats_compute_dqn.episode_lower, rl_stats_compute_dqn.episode_upper, color='blue', alpha=0.2)

plt.plot(rl_stats_compute_dqn_hindsight.iterations, rl_stats_compute_dqn_hindsight.episode_mean, label="dqn hindsight", color='red')
plt.fill_between(rl_stats_compute_dqn_hindsight.iterations, rl_stats_compute_dqn_hindsight.episode_lower, rl_stats_compute_dqn_hindsight.episode_upper, color='red', alpha=0.2)

plt.plot(rl_stats_compute_dqn_curious_goals.iterations, rl_stats_compute_dqn_curious_goals.episode_mean, label="dqn curious goals", color='orange')
plt.fill_between(rl_stats_compute_dqn_curious_goals.iterations, rl_stats_compute_dqn_curious_goals.episode_lower, rl_stats_compute_dqn_curious_goals.episode_upper, color='orange', alpha=0.2)


plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration.png", dpi = 300)

