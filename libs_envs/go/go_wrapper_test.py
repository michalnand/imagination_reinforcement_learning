import numpy
import go_wrapper
import time

size    = 9
env     = go_wrapper.GoEnv(size)


game_id = 0

while True:
    q_values = numpy.random.rand(size*size+1)*2.0 - 1.0

    move = env.step_e_greedy(q_values, 0.3)

    observation, reward, done, info = env.step(move)


    if done:
        observation = env.reset()
        game_id+= 1
        print("games played = ", game_id, ", fps = ", round(env.fps, 2))

