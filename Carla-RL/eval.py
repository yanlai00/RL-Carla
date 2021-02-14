# TODO: Out-dated. Refer to train_sac.py to revise the code. 

import gym
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac import CnnPolicy
from stable_baselines3.common.noise import NormalActionNoise
from carla_env import CarlaEnv

def main(model_name):
    env = CarlaEnv()

    try:
        model = SAC.load(model_name)

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            print(reward, info)
            if done:
                obs = env.reset()
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', help='name of model when saving')

    args = parser.parse_args()
    model_name = args.model_name

    main(model_name)