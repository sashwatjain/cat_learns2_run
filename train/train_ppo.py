"""
train/train_ppo.py

Usage examples:

# Train for 200k timesteps with 4 parallel envs (default)
python train/train_ppo.py

# Train for 1M timesteps and save to a specific folder
python train/train_ppo.py --timesteps 1000000 --n-envs 4 --save-dir ./models

# Evaluate a saved model
python train/train_ppo.py --eval --model-path ./models/ppo_cat.zip --episodes 5
"""

import os
import argparse
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# Import your environment factory
from env.cat_env import CatEnv


def make_env_fn(render: bool = False):
    """Return a callable that creates a single env instance (for vectorized envs)."""
    def _init():
        env = CatEnv(render=render)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="./models")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------
    # EVALUATION MODE
    # -------------------------
    if args.eval:
        if not args.model_path:
            raise ValueError("Evaluation requires --model-path.")
        model = PPO.load(args.model_path, device=device)
        mean_reward, std_reward = evaluate_policy(
            model,
            DummyVecEnv([make_env_fn(render=args.render)]),
            n_eval_episodes=args.episodes,
            deterministic=True,
            render=args.render,
        )
        print(f"[EVAL] mean_reward={mean_reward:.3f} +/- {std_reward:.3f}")
        return

    # -------------------------
    # TRAINING ENVIRONMENT
    # -------------------------
    env_fns = [make_env_fn(render=False) for _ in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # -------------------------
    # MODEL
    # -------------------------
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        tensorboard_log=args.log_dir,
        n_steps=2048 if args.n_envs == 1 else 1024,
        batch_size=256,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
    )

    # Load weights if continuing training
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}")
        model = PPO.load(args.model_path, env=vec_env, device=device)

    # -------------------------
    # CALLBACKS
    # -------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // max(1, args.n_envs),
        save_path=args.save_dir,
        name_prefix="ppo_cat"
    )

    # EVAL CALLBACK — FIXED WRAPPERS
    eval_env = DummyVecEnv([make_env_fn(render=False)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.log_dir,
        eval_freq=50_000 // max(1, args.n_envs),
        n_eval_episodes=5,
        deterministic=True
    )

    # -------------------------
    # TRAIN
    # -------------------------
    print("Starting training...")
    model.learn(total_timesteps=args.timesteps, callback=[checkpoint_cb, eval_cb])

    # -------------------------
    # SAVE FINAL
    # -------------------------
    final_model_path = os.path.join(args.save_dir, "ppo_cat_final.zip")
    model.save(final_model_path)

    vec_norm_path = os.path.join(args.save_dir, "vecnormalize.pkl")
    vec_env.save(vec_norm_path)

    print(f"Training complete. Saved model: {final_model_path}")
    print(f"Saved VecNormalize stats: {vec_norm_path}")

    # -------------------------
    # FINAL EVALUATION — FIXED WRAPPERS
    # -------------------------
    print("Running deterministic final evaluation...")

    eval_env = DummyVecEnv([make_env_fn(render=args.render)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize.load(vec_norm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=5,
        deterministic=True,
        render=args.render,
    )

    print(f"[Final Evaluation] mean_reward={mean_reward:.3f} +/- {std_reward:.3f}")


if __name__ == "__main__":
    main()

