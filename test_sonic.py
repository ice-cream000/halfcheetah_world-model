from retro_contest.local import make
import numpy as np


def main():
    env = make(game='HalfCheetahBulletEnv-v0')
    obs = env.reset()
    total = 0
    while True:
        x = np.zeros((6,), dtype=np.float32)
        obs, rew, done, info = env.step(x)
        env.render()
        if done:
            obs = env.reset()
        total += 1


if __name__ == '__main__':
    main()