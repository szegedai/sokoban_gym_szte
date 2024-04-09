from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from sokoban_gym.envs.sokoban_env import SokobanEnv
import gymnasium as gym
from sokoban_gym.wrappers.observation import ImageObservationWrapper
from sokoban_gym.wrappers.reward import CustomRewardWrapper
from utils.eval_utils import evaluate, create_videos

# Környezet létrehozása
env = gym.make('Sokoban-v1', size=(5, 5), padded_size=(7, 7), num_boxes=[1, 2], render_mode='rgb_array')

# A megfigyelések kiterjesztése a tábla alapján számolt új jellemzők segítségével.
env = ImageObservationWrapper(env, scale=8)

env.reset(seed=42)

# Modell létrehozása
model = A2C('CnnPolicy', env, policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)), verbose=1, seed=42)

print(model.policy)

# Tanulás
model.learn(total_timesteps=40000)

# Model kimentése
model.save("models/Sokoban-v1_5_8_1box_A2C_CNN")

# Kiértékelés 10 véletlen környezetben
score = evaluate(env, model, 100)
print("Score: {}".format(score))

# Videók készítése
create_videos(env, model)