# voxel.py

import pymunk
import math
from config import SPRING_STIFFNESS, SPRING_DAMPING, SHAPE_FRICTION

# --- Voxel types ---
EMPTY      = 0
MUSCLE_A   = 1   # contracts first:  sin(t)
MUSCLE_B   = 2   # expands first:    sin(t + pi)
SOFT       = 3   # passive, compliant
RIGID      = 4   # passive, stiff

SOFT_STIFFNESS  = SPRING_STIFFNESS * 0.25
RIGID_STIFFNESS = SPRING_STIFFNESS * 8.0

# Base colors per type
VOXEL_COLORS = {
    MUSCLE_A: (120,  60, 200),  # purple
    MUSCLE_B: ( 60, 180, 200),  # teal
    SOFT:     ( 80, 160,  80),  # green
    RIGID:    (160, 120,  60),  # tan
}


class Voxel:
    def __init__(self, space, tl, tr, bl, br, size, voxel_type=MUSCLE_A):
        self.space = space
        self.size = size
        self.voxel_type = voxel_type

        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br
        self.bodies = [tl, tr, br, bl]

        self.actuated_springs = []
        self.base_lengths = []
        self.structural_springs = []

        self._connect_internal_springs()

    def _stiffness(self):
        if self.voxel_type == SOFT:
            return SOFT_STIFFNESS
        if self.voxel_type == RIGID:
            return RIGID_STIFFNESS
        return SPRING_STIFFNESS

    def _connect_internal_springs(self):
        stiffness = self._stiffness()

        edge_pairs = [
            (self.tl, self.tr),
            (self.bl, self.br),
            (self.tl, self.bl),
            (self.tr, self.br),
        ]
        diag_pairs = [
            (self.tl, self.br),
            (self.tr, self.bl),
        ]

        for a, b in edge_pairs:
            dist = (a.position - b.position).length
            spring = pymunk.DampedSpring(
                a, b, (0, 0), (0, 0), dist, stiffness, SPRING_DAMPING
            )
            self.space.add(spring)
            if self.voxel_type in (MUSCLE_A, MUSCLE_B):
                self.actuated_springs.append(spring)
                self.base_lengths.append(dist)
            else:
                self.structural_springs.append(spring)

        for a, b in diag_pairs:
            dist = (a.position - b.position).length
            spring = pymunk.DampedSpring(
                a, b, (0, 0), (0, 0), dist, stiffness, SPRING_DAMPING
            )
            self.space.add(spring)
            self.structural_springs.append(spring)

    def get_compression(self):
        edge_pairs = [
            (self.tl, self.tr),
            (self.bl, self.br),
            (self.tl, self.bl),
            (self.tr, self.br),
        ]
        total = 0.0
        for a, b in edge_pairs:
            actual = (a.position - b.position).length
            total += (actual - self.size) / self.size
        return max(-1.0, min(1.0, total / 4))

    def get_color(self):
        c = self.get_compression()
        base = VOXEL_COLORS.get(self.voxel_type, (120, 60, 200))

        if self.voxel_type == RIGID:
            t = abs(c) * 0.3
            return (
                int(base[0] * (1 - t) + 255 * t),
                int(base[1] * (1 - t)),
                int(base[2] * (1 - t)),
            )

        if c < 0:
            t = -c
            return (
                int(base[0] * (1 - t)),
                int(base[1] * (1 - t) + 100 * t),
                int(base[2] * (1 - t) + 255 * t),
            )
        else:
            t = c
            return (
                int(base[0] * (1 - t) + 255 * t),
                int(base[1] * (1 - t) + 60 * t),
                int(base[2] * (1 - t) + 80 * t),
            )

    def apply_scale(self, scale):
        """Only called on MUSCLE_A and MUSCLE_B voxels by the controller."""
        for spring, base in zip(self.actuated_springs, self.base_lengths):
            spring.rest_length = max(5, base * scale)