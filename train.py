from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from sokoban_gym.envs.sokoban_env import SokobanEnv
import gymnasium as gym
from utils.eval_utils import evaluate, create_videos
from sokoban_gym.wrappers.observation import ImageObservationWrapper

# Környezet létrehozása
env = gym.make('Sokoban-v1', size=[(5, 5), (6, 6), (7, 7)], padded_size=(10, 10), num_boxes=[1, 2], render_mode='rgb_array')

env.reset(seed=42)

# A megfigyelések kiterjesztése a tábla alapján számolt új jellemzők segítségével.
env = ImageObservationWrapper(env, scale=8)

env.reset(seed=42)

# Modell létrehozása
model = A2C('CnnPolicy', env, verbose=1, seed=42)

print(model.policy)

# Tanulás
model.learn(total_timesteps=40000)

# Model kimentése
model.save("agent/Sokoban-A2C-CNN")

# Kiértékelés 100 véletlen környezetben
score = evaluate(env, model, 100)
print("Score: {}".format(score))

# Videók készítése
create_videos(env, model, folder="cnn_videos")
