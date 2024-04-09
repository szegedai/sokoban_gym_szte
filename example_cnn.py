from stable_baselines3 import A2C
from sokoban_gym.envs.sokoban_env import SokobanEnv
import gymnasium as gym
from sokoban_gym.wrappers.observation import ImageObservationWrapper
from sokoban_gym.wrappers.reward import CustomRewardWrapper
from utils.eval_utils import evaluate, create_videos

# Környezet létrehozása
env = gym.make('Sokoban-v1', size=(5, 5), padded_size=(8, 8), num_boxes=2, render_mode='rgb_array')

# A megfigyelések kiterjesztése a tábla alapján számolt új jellemzők segítségével.
env = ImageObservationWrapper(env)

# Jutalmak módosítása
env = CustomRewardWrapper(env)

# Modell létrehozása
model = A2C('CnnPolicy',  env, verbose=1, seed=42)

print(model.policy)

# Tanulás
model.learn(total_timesteps=100000)

# Model kimentése
model.save("models/Sokoban-v1_5_8_1box_A2C_CNN")

# Kiértékelés 10 véletlen környezetben
score = evaluate(env, model, 10)
print("Score: {}".format(score))

# Videók készítése
create_videos(env, model)