import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

from .cat_model import CatModel


class CatEnv(gym.Env):
    """
    Gymnasium environment for the Cat. This version is safe for vectorized envs
    because it passes physicsClientId explicitly to PyBullet calls and creates
    the CatModel inside the correct client.
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 240}

    def __init__(self, render: bool = False):
        super().__init__()
        self.render_mode = "human" if render else "none"

        # physics
        self.physics_client = None

        # model
        self.cat = None

        # action / observation
        self.action_dim = 4
        # obs: pitch, roll, vx, vy, vz, joint_angles(4), joint_vels(4) => 5 + 4 + 4 = 13
        self.obs_dim = 13

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # timing
        self.dt = 1.0 / 240.0
        self.max_steps = int(15.0 / self.dt)  # 15 seconds
        self.steps = 0

    # ----------------------------
    # reset
    # ----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # If existing client exists, disconnect it to avoid exceeding connection limits
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = None

        # connect
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        if self.physics_client < 0:
            raise RuntimeError("Failed to connect to PyBullet")

        # basic world
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        # create cat model inside this physics client
        self.cat = CatModel()
        self.cat.load(self.physics_client)

        # reset counters
        self.steps = 0

        obs = self._get_obs()
        return obs, {}

    # ----------------------------
    # step
    # ----------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # apply actions (muscle torques) using the correct client
        self.cat.apply_muscle_forces(action)

        # step the physics for this client
        p.stepSimulation(physicsClientId=self.physics_client)
        if self.render_mode == "human":
            time.sleep(self.dt)

        obs = self._get_obs()

        # reward: forward velocity (x) - energy penalty - tilt penalty
        base_pos, base_orn = self.cat.get_base_position()
        base_vel = self.cat.get_base_velocity()[0]  # (linear_vel, angular_vel)
        forward_vel = base_vel[0]

        reward = 0.1 * forward_vel
        reward -= 0.01 * np.sum(np.abs(action))

        pitch, roll, yaw = p.getEulerFromQuaternion(base_orn)
        reward -= 0.05 * (abs(pitch) + abs(roll))

        # done conditions
        self.steps += 1
        terminated = False
        truncated = False

        # fall detection
        if base_pos[2] < 0.15:
            terminated = True
            reward -= 2.0

        if abs(pitch) > 1.2 or abs(roll) > 1.2:
            terminated = True
            reward -= 1.0

        if self.steps >= self.max_steps:
            truncated = True

        return obs, float(reward), terminated, truncated, {}

    # ----------------------------
    # observation
    # ----------------------------
    def _get_obs(self):
        # get base state using stored client
        base_pos, base_orn = self.cat.get_base_position()
        base_vel, _ = self.cat.get_base_velocity()

        pitch, roll, yaw = p.getEulerFromQuaternion(base_orn)

        torso_obs = np.array([pitch, roll, base_vel[0], base_vel[1], base_vel[2]], dtype=np.float32)

        # joints: real readings
        joint_angles = []
        joint_vels = []
        for j in self.cat.joint_ids:
            js = p.getJointState(self.cat.body_id, j, physicsClientId=self.physics_client)
            joint_angles.append(js[0])
            joint_vels.append(js[1])

        joint_angles = np.array(joint_angles, dtype=np.float32)
        joint_vels = np.array(joint_vels, dtype=np.float32)

        obs = np.concatenate([torso_obs, joint_angles, joint_vels]).astype(np.float32)
        return obs

    # ----------------------------
    # close
    # ----------------------------
    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = None
