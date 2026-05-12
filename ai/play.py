import argparse
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from ai.constants import FPS, FRAME_SKIP
from ai.mario_env import MarioEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Play Mario with a trained PPO model.")
    parser.add_argument("--model-path", default="checkpoints/mario_ppo_v1")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=FPS // FRAME_SKIP)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    env = MarioEnv(headless=args.headless)
    model = PPO.load(args.model_path)

    obs, _ = env.reset()
    clock = env.pg.time.Clock()
    running = True
    started_at = time.time()
    while running:
        for event in env.pg.event.get():
            if event.type == env.pg.QUIT:
                running = False

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs, _ = env.reset()

        if args.seconds > 0 and time.time() - started_at >= args.seconds:
            running = False

        clock.tick(args.fps)

    env.close()


if __name__ == "__main__":
    main()
