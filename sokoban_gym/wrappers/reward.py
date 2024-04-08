import gymnasium as gym

class CustomRewardWrapper(gym.Wrapper):
    """
    Egyedi jutalomfüggvény, amiben több minden egyes doboz célterületre tolása után
    jutalommal tér vissza.
    """

    def __init__(self, env):
        super().__init__(env)
        self.prev_num_correct_boxes = 0
        self.num_targets = 0
        self.reset_flag = False

    def calc_reward(self, info):

        num_correct_boxes = 0

        self.num_targets = len(info['target_locations'])
        grid = info['grid']
        for target in info['target_locations']:
            if grid[target] == 2:
                num_correct_boxes += 1


        if num_correct_boxes == self.prev_num_correct_boxes:
            return 0
        if num_correct_boxes > self.prev_num_correct_boxes:
            self.prev_num_correct_boxes = num_correct_boxes
            return num_correct_boxes / self.num_targets
        else:
            self.prev_num_correct_boxes = num_correct_boxes
            return -(num_correct_boxes / self.num_targets)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = self.calc_reward(info)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.reset_flag = True
        return super().reset(**kwargs)
