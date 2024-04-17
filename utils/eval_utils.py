import gymnasium as gym
from tqdm import tqdm
from sokoban_gym.envs.sokoban_env import SokobanEnv
from agent.agent import Agent


def evaluate(env, model, ep_num=100, seed=0):
    env_test = env
    env_test.reset(seed=seed)

    sum_reward = 0

    # Hat epizódot futtatunk a teszt környezetben.
    for num_episode in tqdm(range(ep_num)):

        # Inicializáljuk a környezetet az epizód kezdetén.
        obs, info = env_test.reset()
        score = 0
        terminated = False
        truncated = False

        # Az epizód addig tart, amíg a terminated vagy a truncated értéke nem lesz igaz.
        while not terminated and not truncated:

            # A megfigyelések alapján (obs) kiválasztjuk a következő akciót.
            # A deterministic=True paraméter azért szükséges, hogy mindig a lehető
            # legjobb paramétert válasszuk.
            action, _states = model.predict(obs, deterministic=True)

            # Végrehajtjuk a kiválasztott akciót.
            obs, reward, terminated, truncated, info = env_test.step(action)

            score += reward

        sum_reward += score

    return sum_reward / ep_num

def evaluate_agent(env, agent, ep_num=100, seed=0, disable_progress_bar=False):
    env_test = env
    env_test.reset(seed=seed)

    sum_reward = 0

    for _ in tqdm(range(ep_num), disable=disable_progress_bar): 
      
        obs, info = env_test.reset()
        score = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, _ = agent.act(obs)

            obs, reward, terminated, truncated, info = env_test.step(action)

            score += reward

        sum_reward += score
        
    env_test.close()

    return sum_reward

def evaluate_agent_competition(max_task_num=400, seed=0, disable_progress_bar=True):
    ep_num_multiplier = int(max_task_num / 40)
    
    box_size_combinations = [
        (5, 1, 5),
        (5, 2, 5),
        (6, 1, 5),
        (6, 2, 5),
        (6, 3, 4),
        (7, 2, 4),
        (7, 3, 3),
        (8, 3, 2),
        (8, 4, 2),
        (9, 4, 2),
        (10, 4, 2),
        (10, 5, 1)
    ]

    # Környezet létrehozása
    correct_results = 0

    for size, num_boxes, num_episodes in tqdm(box_size_combinations):
        env = SokobanEnv(size=(size, size), padded_size=(10, 10), num_boxes=num_boxes, render_mode='rgb_array')

        agent = Agent(env)

        correct_results += evaluate_agent(env, agent, num_episodes*ep_num_multiplier, seed=seed, disable_progress_bar=disable_progress_bar)
    
    print()
    print(f"Result: {int(correct_results)} correct tasks out of {ep_num_multiplier*40}")
    print()

    return int(correct_results)


def create_videos(env, model, ep_num=2, seed=0, folder="videos"):
    env = gym.wrappers.RecordVideo(env, folder, episode_trigger=lambda x: True)
    env_test = env

    env_test.metadata["render_fps"] = 4

    env_test.reset(seed=seed)

    sum_reward = 0

    # Hat epizódot futtatunk a teszt környezetben.
    for num_episode in tqdm(range(ep_num)):

        # Inicializáljuk a környezetet az epizód kezdetén.
        obs, info = env_test.reset()
        score = 0
        terminated = False
        truncated = False

        # Az epizód addig tart, amíg a terminated vagy a truncated értéke nem lesz igaz.
        while not terminated and not truncated:

            # A megfigyelések alapján (obs) kiválasztjuk a következő akciót.
            # A deterministic=True paraméter azért szükséges, hogy mindig a lehető
            # legjobb paramétert válasszuk.
            action, _states = model.predict(obs, deterministic=True)

            # Végrehajtjuk a kiválasztott akciót.
            obs, reward, terminated, truncated, info = env_test.step(action)

            score += reward

        sum_reward += score

    return sum_reward / ep_num
