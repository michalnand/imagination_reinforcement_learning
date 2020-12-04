import sys
sys.path.insert(0, '../../')

from libs_common.RLStatsCompute import *

import matplotlib.pyplot as plt


files = []
files.append("./models/dqn_noisy_resnet/result/result.log")
files.append("./models/dqn_noisy_resnet/result/result.log")
rl_stats_compute_dqn = RLStatsCompute(files, "./results/dqn_baseline.log")


files = []
files.append("./models/dqn_entropy_trajectory_imagination/result/result.log")
files.append("./models/dqn_entropy_trajectory_imagination/result/result.log")
rl_stats_compute_dqn_noisy = RLStatsCompute(files, "./results/dqn_imagination.log")
 

plt.cla()
plt.ylabel("score per episode")
plt.xlabel("episode")

plt.plot(rl_stats_compute_dqn.games_mean, rl_stats_compute_dqn.episode_mean, label="baseline", color='blue')
plt.fill_between(rl_stats_compute_dqn.games_mean, rl_stats_compute_dqn.episode_lower, rl_stats_compute_dqn.episode_upper, color='blue', alpha=0.2)

plt.plot(rl_stats_compute_dqn_noisy.games_mean, rl_stats_compute_dqn_noisy.episode_mean, label="imagination", color='red')
plt.fill_between(rl_stats_compute_dqn_noisy.games_mean, rl_stats_compute_dqn_noisy.episode_lower, rl_stats_compute_dqn_noisy.episode_upper, color='red', alpha=0.2)


plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig("./results/score_per_episode.png", dpi = 300)

 
plt.cla()
plt.ylabel("score per iteration")
plt.xlabel("iteration")

plt.plot(rl_stats_compute_dqn.iterations, rl_stats_compute_dqn.per_iteration_mean, label="baseline", color='blue')
plt.fill_between(rl_stats_compute_dqn.iterations, rl_stats_compute_dqn.per_iteration_lower, rl_stats_compute_dqn.per_iteration_upper, color='blue', alpha=0.2)

plt.plot(rl_stats_compute_dqn_noisy.iterations, rl_stats_compute_dqn_noisy.per_iteration_mean, label="imagination", color='red')
plt.fill_between(rl_stats_compute_dqn_noisy.iterations, rl_stats_compute_dqn_noisy.per_iteration_lower, rl_stats_compute_dqn_noisy.per_iteration_upper, color='red', alpha=0.2)
 

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig("./results/score_per_iteration.png", dpi = 300)
