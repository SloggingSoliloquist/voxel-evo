# voxel.py

import pymunk
import math
from config import SPRING_STIFFNESS, SPRING_DAMPING, SHAPE_FRICTION

VOXEL_BASE_COLOR = (120, 60, 200)

class Voxel:
    def __init__(self, space, tl, tr, bl, br, size):
        """
        Takes 4 pre-built pymunk.Body objects as corners.
        tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right
        """
        self.space = space
        self.size = size
        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br
        self.bodies = [tl, tr, br, bl]  # kept for compatibility
        self.springs = []
        self.base_lengths = []

        self._connect_internal_springs()

    def _connect_internal_springs(self):
        # Edges (actuated)
        edge_pairs = [(self.tl, self.tr), (self.bl, self.br),
                      (self.tl, self.bl), (self.tr, self.br)]
        # Diagonals (structural only)
        diag_pairs = [(self.tl, self.br), (self.tr, self.bl)]

        for a, b in edge_pairs:
            dist = (a.position - b.position).length
            spring = pymunk.DampedSpring(a, b, (0,0), (0,0), dist, SPRING_STIFFNESS, SPRING_DAMPING)
            self.space.add(spring)
            self.springs.append(spring)
            self.base_lengths.append(dist)

        for a, b in diag_pairs:
            dist = (a.position - b.position).length
            spring = pymunk.DampedSpring(a, b, (0,0), (0,0), dist, SPRING_STIFFNESS, SPRING_DAMPING)
            self.space.add(spring)
            # diagonals not added to self.springs — not actuated

    def get_compression(self):
        edge_pairs = [(self.tl, self.tr), (self.bl, self.br),
                      (self.tl, self.bl), (self.tr, self.br)]
        total = 0.0
        for a, b in edge_pairs:
            ax, ay = a.position
            bx, by = b.position
            actual = math.sqrt((ax-bx)**2 + (ay-by)**2)
            total += (actual - self.size) / self.size
        return max(-1.0, min(1.0, total / 4))

    def get_color(self):
        c = self.get_compression()
        base = VOXEL_BASE_COLOR
        if c < 0:
            t = -c
            return (int(base[0]*(1-t)), int(base[1]*(1-t) + 100*t), int(base[2]*(1-t) + 255*t))
        else:
            t = c
            return (int(base[0]*(1-t) + 255*t), int(base[1]*(1-t) + 60*t), int(base[2]*(1-t) + 80*t))

    def apply_scale(self, scale):
        for spring, base in zip(self.springs, self.base_lengths):
            spring.rest_length = max(5, base * scale)