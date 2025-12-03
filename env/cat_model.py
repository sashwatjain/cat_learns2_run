import pybullet as p
import numpy as np


class CatModel:
    """
    Physics-client-aware cat model.
    Call load(physics_client_id) to create the body in a specific pybullet client.
    """

    def __init__(self, max_force: float = 20.0):
        self.max_force = max_force
        self.body_id = None
        self.joint_ids = []
        self._pc = None  # physics client id used to create this body

    def load(self, physics_client_id: int):
        """
        Create a simple 4-joint quadruped inside the specified PyBullet client.
        Returns the body id.
        """
        self._pc = physics_client_id

        # Torso
        torso_half = [0.25, 0.10, 0.08]
        torso_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=torso_half, physicsClientId=self._pc
        )
        torso_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=torso_half, physicsClientId=self._pc
        )

        # Leg
        leg_half = [0.05, 0.05, 0.15]
        leg_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=leg_half, physicsClientId=self._pc
        )
        leg_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=leg_half, physicsClientId=self._pc
        )

        # leg offsets relative to torso base frame
        leg_offsets = [
            [0.2, 0.12, 0.0],   # front-right
            [0.2, -0.12, 0.0],  # front-left
            [-0.2, 0.12, 0.0],  # back-right
            [-0.2, -0.12, 0.0], # back-left
        ]

        num_legs = len(leg_offsets)
        linkMasses = [0.2] * num_legs
        linkCollisionShapeIndices = [leg_collision] * num_legs
        linkVisualShapeIndices = [leg_visual] * num_legs
        linkPositions = leg_offsets
        linkOrientations = [[0, 0, 0, 1]] * num_legs
        linkInertialFramePositions = [[0, 0, 0]] * num_legs
        linkInertialFrameOrientations = [[0, 0, 0, 1]] * num_legs
        linkParentIndices = [0] * num_legs
        linkJointTypes = [p.JOINT_REVOLUTE] * num_legs
        linkJointAxis = [[0, 1, 0]] * num_legs  # rotate around local Y axis

        self.body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=torso_collision,
            baseVisualShapeIndex=torso_visual,
            basePosition=[0, 0, 0.3],
            baseOrientation=[0, 0, 0, 1],
            linkMasses=linkMasses,
            linkCollisionShapeIndices=linkCollisionShapeIndices,
            linkVisualShapeIndices=linkVisualShapeIndices,
            linkPositions=linkPositions,
            linkOrientations=linkOrientations,
            linkInertialFramePositions=linkInertialFramePositions,
            linkInertialFrameOrientations=linkInertialFrameOrientations,
            linkParentIndices=linkParentIndices,
            linkJointTypes=linkJointTypes,
            linkJointAxis=linkJointAxis,
            physicsClientId=self._pc,
        )

        # store joint indices (guaranteed to exist after createMultiBody)
        n = p.getNumJoints(self.body_id, physicsClientId=self._pc)
        self.joint_ids = list(range(n))

        # disable default velocity motors so torque control works cleanly
        for j in self.joint_ids:
            p.setJointMotorControl2(
                bodyIndex=self.body_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self._pc,
            )

        return self.body_id

    def apply_muscle_forces(self, actions):
        """
        actions: iterable-like of length >= len(self.joint_ids)
        Each action in [-1,1] scaled to torque = action * max_force
        All PyBullet calls use the stored physics client id.
        """
        if self._pc is None:
            raise RuntimeError("CatModel: physics client id not set. Call load(pc) first.")

        actions = np.array(actions, dtype=float).flatten()
        # If fewer actions provided than joints, only apply available actions
        n = min(len(self.joint_ids), len(actions))
        for idx in range(n):
            joint_idx = self.joint_ids[idx]
            torque = float(actions[idx]) * float(self.max_force)
            p.setJointMotorControl2(
                bodyIndex=self.body_id,
                jointIndex=joint_idx,
                controlMode=p.TORQUE_CONTROL,
                force=torque,
                physicsClientId=self._pc,
            )

    # helper read methods that take physics client id into account
    def get_joint_states(self):
        states = []
        for j in self.joint_ids:
            js = p.getJointState(self.body_id, j, physicsClientId=self._pc)
            states.append(js)
        return states

    def get_base_position(self):
        return p.getBasePositionAndOrientation(self.body_id, physicsClientId=self._pc)

    def get_base_velocity(self):
        return p.getBaseVelocity(self.body_id, physicsClientId=self._pc)
