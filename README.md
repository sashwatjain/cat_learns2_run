
# 🐾 **cat_learns2_run — Version 1**

### *An AI-powered quadruped that learns to run using Reinforcement Learning + PyBullet Physics*

---

## 🎥 Demo

```
![Cat Running Demo](./assets/cat_running.gif)
```

---

## 🧠 Project Inspiration

I created **cat_learns2_run** because I wanted to learn:

* How **robots learn movement** from scratch
* How **Reinforcement Learning** interacts with **real-time physics**
* How to build a **robotic simulation** that evolves behavior through trial and error
* How muscle-like torques and multi-body physics work

I’ve always been fascinated by how animals learn to walk, run, or balance. Instead of writing fixed rules, I wanted the robot to **figure it out on its own** — like a newborn animal learning to stand and take its first steps.

This project is a combination of:

* **Creativity**
* **Physics simulation**
* **AI & RL theory**
* **Software engineering**

Version 1 focuses on creating a **simple but functional quadruped robot** and teaching it to run forward. Future versions will add biological realism, more joints, better reward shaping, and more expressive motion.

---

# 🚀 Features (Version 1)

### 🐈 Simple quadruped robot

* 1 torso
* 4 legs
* 4 revolute hip joints
* torque-based actuation (muscles)

### ⚙️ Physics with PyBullet

* Real gravity
* Multi-body dynamics
* Torque-controlled motors
* Accurate collision detection

### 🧩 Reinforcement Learning with PPO

* Stable-Baselines3
* VecNormalize (observation + reward normalization)
* Vectorized environments
* Checkpointing & evaluation callbacks

### 👀 Visualization

* PyBullet GUI
* Auto-follow camera
* Deterministic playback of trained policies
* Support for multiple episodes

---

# 📦 Project Structure

        ```
        cat_learns2_run/
        │
        ├── env/
        │   ├── cat_env.py        # RL environment (Gym-style)
        │   ├── cat_model.py      # Robot body, joints, and muscle physics
        │
        ├── train/
        │   ├── train_ppo.py      # PPO training script
        │
        ├── visualize/
        │   ├── visualize_model.py # GUI visualization of trained model
        │
        ├── models/               # Saved PPO models + normalization stats
        ├── logs/                 # TensorBoard logs
        └── .venv/                # Python virtual environment
        ```

---

# 📚 Technical Explanation

## 1️⃣ **Robot Model (cat_model.py)**

The robot is generated using `createMultiBody()`:

* Torso = simple box
* Each leg = box link
* Joints = **revolute (hinge)** joints with torque control
* Joint axis = `[0, 1, 0]` → legs swing forward/back
* Collision frames moved down so legs pivot from the top (like real anatomy)

Torques come directly from RL actions:

```python
torque = action[i] * max_force
p.setJointMotorControl2(... TORQUE_CONTROL ...)
```

Each RL step applies new torques → PyBullet simulates motion → next state returned.

---

## 2️⃣ **Environment (cat_env.py)**

The environment wraps the physics into an RL-compatible interface:

* `reset()`
* `step(action)`
* `observation_space`
* `action_space`

Observations include:

* torso orientation
* linear velocity
* joint angles
* joint velocities

Rewards are based on:

* forward speed
* energy usage
* balance
* avoiding falling

This drives the cat to **run faster** and **stay upright**.

---

## 3️⃣ **Reinforcement Learning (train_ppo.py)**

Uses **Proximal Policy Optimization (PPO)**.

### Key components:

* **VecNormalize** — normalizes observations & rewards
* **DummyVecEnv** — runs multiple envs in parallel
* **Checkpoints** — saves model every N steps
* **EvalCallback** — runs periodic evaluations

Training call:

```python
model.learn(total_timesteps=200_000)
```

Over time, PPO discovers a **running gait** by optimizing reward.

---

## 4️⃣ **Visualization (visualize_model.py)**

A dedicated script shows the learned motion:

* Opens PyBullet GUI
* Loads model + normalization stats
* Runs deterministic policy (no randomness)
* Auto-follow camera tracks the cat

You can visually see:

* leg torques
* gait rhythm
* speed and stability improving over episodes

---

# 🧪 How to Train

```
python -m train.train_ppo --timesteps 200000 --n-envs 4
```

Model + VecNormalize stats will be saved in:

```
./models/
```

---

# 🎬 How to Visualize

```
python -m visualize.visualize_model --model-path ./models/ppo_cat_final.zip --vecnorm-path ./models/vecnormalize.pkl
```

A PyBullet window will open showing your running cat.

---

# 📈 Version 1 Limitations

This is an early version, intentionally simple:

* Legs have 1 joint each (hip joint only)
* No knees
* No spine movement
* No ground reaction force sensing
* No energy-efficient gaits
* No URDF model

Even with these limitations, PPO discovers **functional running behavior**.

---

# 🗺️ Roadmap (Planned for Version 2+)

### 🦴 More realistic anatomy

* 2-segment legs (hip + knee)
* Joint limits
* URDF skeleton

### ⚙️ Better physics

* Muscle activation modeling
* Tendon-like dampers
* Friction optimization

### 🤖 Smarter learning

* Curriculum learning
* Energy-efficient reward shaping
* Terrain adaptation

### 🎥 Better visualization

* Torque bars
* Foot contact indicators
* Camera shake/zoom
* Record MP4 animation

---

# 🙏 Acknowledgments

This project is inspired by:

* Reinforcement learning research in robotics
* OpenAI locomotion work
* PyBullet examples
* Biological movement in quadrupeds

---

# 📣 Author’s Note

This project was built not just as a technical exercise but as a journey into:

* AI
* biomechanics
* physics simulation
* creative problem solving

I hope this inspires others to build their own learning agents.
**Version 1 is just the beginning.**

---

