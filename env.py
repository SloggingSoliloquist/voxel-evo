# env.py
# Simulation environment wrapping pymunk for PPO training.

import math
import pymunk
from config import (
    WIDTH, HEIGHT, VOXEL_SIZE, GRAVITY, SUBSTEPS, FPS,
    ROWS, COLS
)
from world import create_space, init_ground, extend_ground, terrain_end_x, terrain_points
from grid import build_grid
from voxel import MUSCLE_A, MUSCLE_B

# Fixed obs/action sizes based on max grid — always pad to this
MAX_NODES   = (ROWS + 1) * (COLS + 1)
OBS_SIZE    = MAX_NODES * 4          # (rel_x, rel_y, vx, vy) per node
MAX_MUSCLES = ROWS * COLS            # upper bound on muscle count
ACTION_SIZE = MAX_MUSCLES

GROUND_Y_FRAC = 0.75
DT = 1.0 / FPS


def _get_com(bodies):
    """Center of mass position."""
    x = sum(b.position.x for b in bodies) / len(bodies)
    y = sum(b.position.y for b in bodies) / len(bodies)
    return x, y


class VoxelEnv:
    """
    A single-episode pymunk environment for one morphology.

    reset() → obs
    step(action) → obs, reward, done
    """

    def __init__(self, morphology):
        self.morphology = morphology
        self.rows = len(morphology)
        self.cols = len(morphology[0])

        # Identify muscle voxels once — order is fixed per morphology
        self.muscle_positions = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if morphology[r][c] in (MUSCLE_A, MUSCLE_B)
        ]
        self.n_muscles = len(self.muscle_positions)

        self.space   = None
        self.voxels  = None
        self.nodes   = None   # flat list of all active bodies
        self.prev_x  = None
        self.t       = 0.0

    # ------------------------------------------------------------------

    def reset(self):
        self.space = create_space(GRAVITY)
        ground_y   = int(HEIGHT * GROUND_Y_FRAC)
        init_ground(self.space, start_x=0, start_y=ground_y, length=WIDTH * 6)

        robot_height = self.rows * VOXEL_SIZE
        spawn_y      = ground_y - robot_height - 5

        self.voxels = build_grid(
            self.space, start_x=200, start_y=spawn_y,
            morphology=self.morphology
        )

        # Collect all unique node bodies
        seen = set()
        self.nodes = []
        for row in self.voxels:
            for v in row:
                if v is None:
                    continue
                for b in v.bodies:
                    if id(b) not in seen:
                        seen.add(id(b))
                        self.nodes.append(b)

        com_x, _ = _get_com(self.nodes)
        self.prev_x = com_x
        self.t = 0.0

        return self._observe()

    # ------------------------------------------------------------------

    def step(self, action):
        """
        action: list/array of length ACTION_SIZE in [0, 1]
                mapped to spring scale [0.6, 1.6]
        """
        # Apply actions to muscle voxels
        for idx, (r, c) in enumerate(self.muscle_positions):
            if idx >= len(action):
                break
            scale = 0.6 + action[idx] * 1.0   # [0.6, 1.6]
            voxel = self.voxels[r][c]
            if voxel is not None:
                voxel.apply_scale(scale)

        # Step physics
        for _ in range(SUBSTEPS):
            self.space.step(DT / SUBSTEPS)
        self.t += DT

        # Extend ground if needed
        if self.nodes:
            head_x = self.nodes[0].position.x
            if terrain_end_x() - head_x < WIDTH * 2:
                extend_ground(self.space, count=50)

        obs    = self._observe()
        com_x, _ = _get_com(self.nodes)
        reward = (com_x - self.prev_x) / VOXEL_SIZE   # normalised by voxel size
        self.prev_x = com_x

        return obs, reward, False   # no terminal condition — PPO runs fixed steps

    # ------------------------------------------------------------------

    def _observe(self):
        """
        Build observation vector of length OBS_SIZE.
        Each node contributes (rel_x, rel_y, vx, vy) relative to CoM.
        Inactive slots are zero-padded.
        """
        obs = [0.0] * OBS_SIZE

        if not self.nodes:
            return obs

        com_x, com_y = _get_com(self.nodes)

        for i, body in enumerate(self.nodes):
            if i >= MAX_NODES:
                break
            base = i * 4
            obs[base]     = (body.position.x - com_x) / VOXEL_SIZE
            obs[base + 1] = (body.position.y - com_y) / VOXEL_SIZE
            obs[base + 2] = body.velocity.x / 100.0
            obs[base + 3] = body.velocity.y / 100.0

        return obs