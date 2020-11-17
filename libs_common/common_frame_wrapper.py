import gym
import numpy
from PIL import Image



class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip = 4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            state, reward, done, info = self.env.step(action)
            total_reward+= reward
            if done:
                break

        return state, total_reward, done, info


class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, height = 96, width = 96, frame_stacking = 4):
        super(ResizeEnv, self).__init__(env)
        self.height = height
        self.width  = width
        self.frame_stacking = frame_stacking

        state_shape = (self.frame_stacking, self.height, self.width)
        self.dtype = numpy.float32

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=state_shape, dtype=self.dtype)
        self.state = numpy.zeros(state_shape, dtype=self.dtype)

    def observation(self, state):
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize((self.height, self.width))

        for i in reversed(range(self.frame_stacking-1)):
            self.state[i+1] = self.state[i].copy()
        self.state[0] = (numpy.array(img).astype(self.dtype)/255.0).copy()

        return self.state




class ClipRewardEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.raw_score_per_episode   = 0.0
        self.raw_score_per_iteration = 0.0
        self.raw_reward              = 0.0
        self.raw_reward_episode_sum  = 0.0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.raw_reward_episode_sum+= reward

        self.raw_reward = reward


        k = 0.01
        self.raw_score_per_iteration = (1.0 - k)*self.raw_score_per_iteration + k*reward

        if done:
            k = 0.05
            self.raw_score_per_episode   = (1.0 - k)*self.raw_score_per_episode + k*self.raw_reward_episode_sum
            self.raw_reward_episode_sum  = 0.0

        reward = numpy.clip(reward, -1.0, 1.0)
        return obs, reward, done, info


def CommonFrameWrapper(env, height = 96, width = 96, frame_stacking=4, frame_skipping=4, Preprocessing = None):
    env = SkipEnv(env, frame_skipping)
    env = ResizeEnv(env, height, width, frame_stacking)
    env = ClipRewardEnv(env)

    if Preprocessing is not None:
        env = Preprocessing(env)

    env.observation_space.shape = (frame_stacking, height, width)

    return env