# voxel.py

import pymunk
import math
from config import SPRING_STIFFNESS, SPRING_DAMPING

class Voxel:
    def __init__(self, space, x, y, size):
        self.space = space
        self.size = size
        self.bodies = []
        self.springs = []
        self.base_lengths = []

        offsets = [
            (-size/2, -size/2),
            ( size/2, -size/2),
            ( size/2,  size/2),
            (-size/2,  size/2),
        ]

        for dx, dy in offsets:
            body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
            body.position = x + dx, y + dy
            shape = pymunk.Circle(body, 5)
            shape.friction = 1.0
            space.add(body, shape)
            self.bodies.append(body)

        self._connect_internal_springs()

    def _connect_internal_springs(self):
        pairs = [
            (0,1),(1,2),(2,3),(3,0),
            (0,2),(1,3)
        ]

        for i,j in pairs:
            if abs(i-j)==1 or abs(i-j)==3:
                length = self.size
            else:
                length = self.size * math.sqrt(2)

            spring = pymunk.DampedSpring(
                self.bodies[i],
                self.bodies[j],
                (0,0),(0,0),
                length,
                SPRING_STIFFNESS,
                SPRING_DAMPING
            )

            self.space.add(spring)
            self.springs.append(spring)
            self.base_lengths.append(length)

    def modulate(self, t, amplitude=3, frequency=2):
        for spring, base in zip(self.springs, self.base_lengths):
            new_length = base + amplitude * math.sin(frequency * t)
            spring.rest_length = max(5, new_length)

    def apply_scale(self, scale):
        for spring, base in zip(self.springs, self.base_lengths):
            spring.rest_length = max(5, base * scale)

