import sys
sys.path.insert(0, '..')
import gym
import numpy
import time

import libs_agents

import libs_common.atari_wrapper

from PIL import Image


def make_graph(self, state, tiles_count = 8):
    channels    = state.shape[0]
    height      = state.shape[1]
    width       = state.shape[2]

    tile_height = height//tiles_count
    tile_width  = width//tiles_count

    total_tiles_count = tiles_count*tiles_count

    points = numpy.zeros(total_tiles_count, channels, tile_height, tile_width)
    edges  = numpy.zeros((2, total_tiles_count*8) dtype=int) 

    idx = 0
    for y in range(tiles_count):
        for x in range(tiles_count):
            ys = y*tile_height
            ye = (y+1)*tile_height

            xs = x*tile_width
            xe = (x+1)*tile_width

            points[idx] = state[:, ys:ye, xs:xe].copy()
            idx+= 1


    


#env = gym.make("PongNoFrameskip-v4")
#env = gym.make("BreakoutNoFrameskip-v4")
env = gym.make("MsPacmanNoFrameskip-v4")
env = libs_common.atari_wrapper.AtariWrapper(env)
env.reset()

agent = libs_agents.AgentRandom(env)


obs, _, _, _ = env.step(0)


print(obs)
print(obs.shape)


im = Image.fromarray(obs[0]*255.0)
im.show()

k = 0.02
fps = 0
while True:
    time_start = time.time()
    reward, done = agent.main()
    time_stop  = time.time()
    env.render()

    fps = (1.0-k)*fps + k*1.0/(time_stop - time_start)


    if reward != 0:
        print("reward = ", reward)
    
    if done:
        print("FPS = ", round(fps, 1))
        print("DONE \n\n")
    
    time.sleep(0.01)
