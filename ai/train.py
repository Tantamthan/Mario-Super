import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from ai.mario_env import MarioEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for MarioEnv.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-path", default="checkpoints/mario_ppo_v1")
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-prefix", default="mario_ppo_checkpoint")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    env = make_vec_env(lambda: MarioEnv(headless=True), n_envs=args.n_envs)
    if args.resume:
        print(f"Loading model from {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
        )

    callback = None
    if args.checkpoint_freq > 0:
        save_freq = max(args.checkpoint_freq // args.n_envs, 1)
        callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=args.checkpoint_dir,
            name_prefix=args.checkpoint_prefix,
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        reset_num_timesteps=not bool(args.resume),
    )
    model.save(args.save_path)
    env.close()


if __name__ == "__main__":
    main()
