# voxel.py

import pymunk
import math
from config import SPRING_STIFFNESS, SPRING_DAMPING, SHAPE_FRICTION

VOXEL_BASE_COLOR = (120, 60, 200)


class Voxel:
    def __init__(self, space, tl, tr, bl, br, size):
        """
        Voxel defined by four shared corner bodies.
        tl = top-left, tr = top-right
        bl = bottom-left, br = bottom-right
        """

        self.space = space
        self.size = size

        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br

        # compatibility with previous code
        self.bodies = [tl, tr, br, bl]

        # springs that will be actuated
        self.actuated_springs = []
        self.base_lengths = []

        # structural springs
        self.structural_springs = []

        self._connect_internal_springs()

    def _connect_internal_springs(self):

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

        # actuated springs
        for a, b in edge_pairs:
            dist = (a.position - b.position).length

            spring = pymunk.DampedSpring(
                a,
                b,
                (0, 0),
                (0, 0),
                dist,
                SPRING_STIFFNESS,
                SPRING_DAMPING,
            )

            self.space.add(spring)

            self.actuated_springs.append(spring)
            self.base_lengths.append(dist)

        # structural springs
        for a, b in diag_pairs:
            dist = (a.position - b.position).length

            spring = pymunk.DampedSpring(
                a,
                b,
                (0, 0),
                (0, 0),
                dist,
                SPRING_STIFFNESS,
                SPRING_DAMPING,
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
        base = VOXEL_BASE_COLOR

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
        for spring, base in zip(self.actuated_springs, self.base_lengths):
            spring.rest_length = max(5, base * scale)