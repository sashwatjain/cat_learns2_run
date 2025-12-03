"""
visualize/visualize_model.py

Visualize a trained cat model with PyBullet GUI and auto camera tracking.

Usage:
python -m visualize.visualize_model --model-path ./models/ppo_cat_final.zip --vecnorm-path ./models/vecnormalize.pkl
"""

import argparse
import time
import torch
import pybullet as p
import numpy as np
import pickle
import sys, os

from stable_baselines3 import PPO
from env.cat_env import CatEnv


def track_camera(body_id, distance=1.5, height=0.5):
    pos, orn = p.getBasePositionAndOrientation(body_id)
    euler = p.getEulerFromQuaternion(orn)
    yaw = np.degrees(euler[2])
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=-15,
        cameraTargetPosition=[pos[0], pos[1], height]
    )


def _extract_obs_rms_from_pickle(data):
    def _rms_to_mean_var(rms_obj):
        mean = getattr(rms_obj, "mean", None)
        var = getattr(rms_obj, "var", None)
        if mean is None or var is None:
            try:
                mean = rms_obj["mean"]
                var = rms_obj["var"]
            except Exception:
                mean, var = None, None
        if mean is None or var is None:
            raise ValueError("Could not read RunningMeanStd mean/var from object.")
        return np.array(mean, dtype=np.float32), np.array(var, dtype=np.float32)

    if isinstance(data, dict):
        if "obs_rms" in data:
            return _rms_to_mean_var(data["obs_rms"])
        for k in ("vecnormalize", "vec_norm", "wrapped"):
            if k in data and hasattr(data[k], "obs_rms"):
                return _rms_to_mean_var(data[k].obs_rms)
        raise ValueError("Pickle dictionary did not contain 'obs_rms' key.")

    obs_rms = getattr(data, "obs_rms", None)
    if obs_rms is not None:
        return _rms_to_mean_var(obs_rms)

    try:
        return _rms_to_mean_var(data)
    except Exception:
        pass

    raise ValueError("Could not extract obs_rms from the pickle file.")


def load_vecnorm_stats_safe(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[VecNorm] Failed to open {path}: {e}")
        return None

    try:
        obs_mean, obs_var = _extract_obs_rms_from_pickle(data)
    except Exception as e:
        print(f"[VecNorm] Failed to parse vecnormalize data: {e}")
        return None

    def normalize_obs(obs):
        obs = np.array(obs, dtype=np.float32)
        return (obs - obs_mean) / np.sqrt(obs_var + 1e-8)

    print("[VecNorm] loaded obs mean/var successfully.")
    return normalize_obs


def _safe_disconnect_any():
    # Try to disconnect any existing connection (best-effort).
    try:
        info = p.getConnectionInfo()
        # if connected, disconnect
        p.disconnect()
        # short pause for PyBullet to cleanup
        time.sleep(0.05)
    except Exception:
        # no connection or already disconnected — ignore
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained PPO model (.zip)")
    parser.add_argument("--vecnorm-path", type=str, required=False,
                        help="Path to saved VecNormalize stats (.pkl)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to visualize")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Clean stale connections that could block GUI creation
    _safe_disconnect_any()

    # Create direct environment; CatEnv will create the GUI client when render=True
    env = CatEnv(render=True)

    # Load normalization stats safely
    vecnorm = None
    if args.vecnorm_path:
        print(f"Loading VecNormalize stats from {args.vecnorm_path} (robust loader)...")
        vecnorm = load_vecnorm_stats_safe(args.vecnorm_path)
        if vecnorm is None:
            print("[VecNorm] Warning: running without obs normalization.")

    print(f"Loading trained model: {args.model_path}")
    model = PPO.load(args.model_path, device=device)

    try:
        for ep in range(args.episodes):
            print(f"\n--- Episode {ep + 1} ---")
            obs, _ = env.reset()
            done = False
            truncated = False

            while not (done or truncated):
                obs_in = vecnorm(obs) if vecnorm else obs
                action, _ = model.predict(obs_in, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

                # camera tracking (safe)
                try:
                    if hasattr(env, "cat") and env.cat and env.cat.body_id is not None:
                        track_camera(env.cat.body_id)
                except Exception:
                    pass

                time.sleep(1 / 240.0)

            print(f"Episode {ep + 1} finished.")
    finally:
        # cleanup
        try:
            env.close()
        except Exception:
            pass
        try:
            p.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
