import sys
sys.path.insert(0, '../../')

from libs_common.RLStatsCompute import *

import matplotlib.pyplot as plt

result_path = "./results/"

files = []
files.append("./models/ddpg_baseline/run_0/result/result.log")
files.append("./models/ddpg_baseline/run_1/result/result.log")
files.append("./models/ddpg_baseline/run_2/result/result.log")
files.append("./models/ddpg_baseline/run_3/result/result.log")
files.append("./models/ddpg_baseline/run_4/result/result.log")
files.append("./models/ddpg_baseline/run_5/result/result.log")
files.append("./models/ddpg_baseline/run_6/result/result.log")
files.append("./models/ddpg_baseline/run_7/result/result.log")
rl_stats_compute_ddpg = RLStatsCompute(files, result_path + "ddpg_baseline.log")



files = []
files.append("./models/ddpg_entropy_trajectory/run_0/result/result.log")
files.append("./models/ddpg_entropy_trajectory/run_1/result/result.log")
rl_stats_compute_imagination = RLStatsCompute(files, result_path + "ddpg_entropy_trajectory.log") 


plt.cla()
plt.ylabel("score")
plt.xlabel("episode")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_ddpg.games_mean, rl_stats_compute_ddpg.episode_mean, label="ddpg baseline", color='blue')
plt.fill_between(rl_stats_compute_ddpg.games_mean, rl_stats_compute_ddpg.episode_lower, rl_stats_compute_ddpg.episode_upper, color='blue', alpha=0.2)


plt.plot(rl_stats_compute_imagination.games_mean, rl_stats_compute_imagination.episode_mean, label="ddpg entropy trajectory", color='red')
plt.fill_between(rl_stats_compute_imagination.games_mean, rl_stats_compute_imagination.episode_lower, rl_stats_compute_imagination.episode_upper, color='red', alpha=0.2)


plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "score_per_episode.png", dpi = 300)

 
plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_ddpg.iterations, rl_stats_compute_ddpg.episode_mean, label="ddpg baseline", color='blue')
plt.fill_between(rl_stats_compute_ddpg.iterations, rl_stats_compute_ddpg.episode_lower, rl_stats_compute_ddpg.episode_upper, color='blue', alpha=0.2)

plt.plot(rl_stats_compute_imagination.iterations, rl_stats_compute_imagination.episode_mean, label="ddpg entropy trajectory", color='red')
plt.fill_between(rl_stats_compute_imagination.iterations, rl_stats_compute_imagination.episode_lower, rl_stats_compute_imagination.episode_upper, color='red', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration.png", dpi = 300)
