import numpy as np
import gymnasium as gym

from sokoban_gym.utils.game_utils import room_to_tiny_world_rgb


class ImageObservationWrapper(gym.ObservationWrapper):
    """
    A sokoban térképet átalakítja egy RGB képpé.
    """

    def __init__(self, env, scale=8):
        super().__init__(env)
        self.scale = scale
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.padded_size[0] * scale, self.padded_size[1] * scale, 3),
            dtype=np.uint8
        )

    def observation(self, observation):
        return room_to_tiny_world_rgb(observation, scale=self.scale)