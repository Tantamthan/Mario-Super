import argparse
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.mario_env import MarioEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check MarioEnv.")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    return parser.parse_args()


def print_observation(obs):
    labels = [
        "mario_x",
        "mario_y",
        "x_vel",
        "y_vel",
        "is_jumping",
        "dist_enemy",
        "dist_obstacle",
        "score",
        "time_remaining",
        "is_on_ground",
        "ground_1_ahead",
        "ground_2_ahead",
        "ground_3_ahead",
        "gap_distance",
    ]
    for name, val in zip(labels, obs):
        print(f"{name:20s} = {val:.4f}")


def run_random_steps(env, total_steps):
    obs, _ = env.reset()
    for _ in range(total_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


def main():
    args = parse_args()
    env = MarioEnv(headless=True)
    obs, _ = env.reset()
    print_observation(obs)

    if args.benchmark:
        start = time.time()
        run_random_steps(env, args.steps)
        elapsed = time.time() - start
        print(f"{args.steps} steps trong {elapsed:.2f}s = {args.steps / elapsed:.0f} steps/giay")
    else:
        run_random_steps(env, 200)

    env.close()
    print("Environment OK")


if __name__ == "__main__":
    main()
