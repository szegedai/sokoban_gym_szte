from stable_baselines3 import A2C
from sokoban_gym.envs.sokoban_env import SokobanEnv
import gymnasium as gym
#from tetris_gym.wrappers.observation import ExtendedObservationWrapper
from utils.eval_utils import evaluate, create_videos

# Környezet létrehozása
env = gym.make('Sokoban-v1', size=(5, 5), padded_size=(8, 8), num_boxes=1, render_mode='rgb_array')

# Modell létrehozása
model = A2C('MlpPolicy',  env, verbose=1, seed=42)

print(model.policy)

# Tanulás
model.learn(total_timesteps=100000)

# Model kimentése
model.save("models/Sokoban-v1_5_8_1box_A2C")

# Kiértékelés 10 véletlen környezetben
score = evaluate(env, model, 100)
print("Score: {}".format(score))

# Videók készítése
create_videos(env, model)