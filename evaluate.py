from agent.agent import Agent
from sokoban_gym.envs.sokoban_env import SokobanEnv
import gymnasium as gym
from utils.eval_utils import evaluate_agent

# Környezet létrehozása
env = gym.make('Sokoban-v1', size=(5, 5), padded_size=(8, 8), num_boxes=2, render_mode='rgb_array')

agent = Agent(env)

print(evaluate_agent(env, agent, 100))
