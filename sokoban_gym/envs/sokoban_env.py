import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sokoban_gym.utils.game_utils import generate_and_convert_room, room_to_tiny_world_rgb
import random

EMPTY = 0
WALL = 1
BOX = 2
TARGET = 3
PLAYER = 4

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

colors = {
    0: '\033[30m',  # Black
    1: '\033[0m',  # Reset color
    2: '\033[34m',  # Blue
    3: '\033[32m',  # Green
    4: '\033[31m'  # Red
}

bg_colors = {
    0: '\033[40m',  # Black
    1: '\033[47m',  # Reset color
    2: '\033[44m',  # Blue
    3: '\033[42m',  # Green
    4: '\033[41m'  # Red
}


class SokobanEnv(gym.Env):
    metadata = {'render_modes': ['text', 'rgb_array']}

    def __init__(self, render_mode=None, size=(5, 5), padded_size=(8, 8), num_boxes=1, time_limit=50):
        self.size = size
        self.padded_size = padded_size
        self.num_boxes = num_boxes

        self.time_limit = time_limit
        self.num_step = 0
        self.prev_num_correct_boxes = 0

        self.grid = None
        self._player_location = None
        self._target_locations = None

        # 0: empty, 1: wall, 2: box, 3: target, 4: player
        self.observation_space = spaces.Box(low=0, high=4, shape=self.padded_size, dtype=int)

        # 4 actions, corresponding to 'up', 'down', 'left', 'right'
        self.action_space = spaces.Discrete(5, start=1)

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

    def _get_obs(self):
        # if self.render_mode == 'rgb_array':
        #     return self.render()
        return self.grid

    def _get_info(self):
        return {'grid': self.grid, 'target_locations': self._target_locations}

    def _calc_reward(self):
        num_correct_boxes = 0
        num_targets = len(self._target_locations)
        for target in self._target_locations:
            if self.grid[target] == BOX:
                num_correct_boxes += 1

        if num_correct_boxes == num_targets:
            return 1.0
        else:
            return 0.0

    def _check_win(self):
        for target in self._target_locations:
            if self.grid[target] != BOX:
                return False
        return True

    def _update_grid(self, action):
        _new_player_location = (self._player_location[0] + action[0], self._player_location[1] + action[1])

        # if new_loc is box
        if self.grid[_new_player_location] == BOX:
            _new_box_location = (_new_player_location[0] + action[0], _new_player_location[1] + action[1])

            # if new_box_loc is empty or target
            if self.grid[_new_box_location] == EMPTY or self.grid[_new_box_location] == TARGET:
                self.grid[_new_player_location] = EMPTY  # delete box from prev loc
                self.grid[_new_box_location] = BOX  # add box to new loc

        # if new_loc is empty or target
        if self.grid[_new_player_location] == EMPTY or self.grid[_new_player_location] == TARGET:
            if self._player_location in self._target_locations:
                self.grid[self._player_location] = TARGET
            else:
                self.grid[self._player_location] = EMPTY

            self.grid[_new_player_location] = PLAYER  # add player to new loc

            self._player_location = _new_player_location

    def _move(self, action):
        if action == 0:
            pass
        elif action == 1:
            self._update_grid(UP)
        elif action == 2:
            self._update_grid(DOWN)
        elif action == 3:
            self._update_grid(LEFT)
        elif action == 4:
            self._update_grid(RIGHT)

    def set_task(self):
        room_structure = generate_and_convert_room(dim=self.size, padded_size=self.padded_size, num_boxes=self.num_boxes)
        self.grid = room_structure
        self._target_locations = [(t[0], t[1]) for t in np.argwhere(self.grid == TARGET)]
        player_loc = np.argwhere(self.grid == PLAYER)
        self._player_location = (player_loc[0][0], player_loc[0][1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_step = 0
        self.prev_num_correct_boxes = 0

        self.set_task()

        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return observation, info

    def step(self, action):

        self._move(action)
        self.render()

        terminated = False
        reward = 0
        observation = self._get_obs()
        info = self._get_info()

        self.num_step += 1
        if self.num_step >= self.time_limit:
            terminated = True
        else:
            reward = self._calc_reward()
            if self._check_win():
                terminated = True

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return room_to_tiny_world_rgb(self.grid, 32)

        if self.render_mode == 'text':
            for row in self.grid:
                for element in row:
                    print(bg_colors[element] + str(element), end=' ')
                print('\033[0m')


gym.envs.register(
    id='Sokoban-v1',
    entry_point='sokoban_gym.envs.sokoban_env:SokobanEnv',
    kwargs={}
)
