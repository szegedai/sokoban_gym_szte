import gymnasium as gym
from tqdm import tqdm

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

def evaluate_agent(env, agent, ep_num=100, seed=0):
    env_test = env
    env_test.reset(seed=seed)

    sum_reward = 0

    for _ in range(ep_num): 
      
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

    return sum_reward / ep_num

def create_videos(env, model, ep_num=2, seed=0, folder="videos"):
    env = gym.wrappers.RecordVideo(env, folder)
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
