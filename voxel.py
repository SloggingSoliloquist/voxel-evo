# voxel.py

import pymunk
import math
import random
from config import SPRING_STIFFNESS, SPRING_DAMPING, SHAPE_FRICTION

class Voxel:
    def __init__(self, space, x, y, size):
        self.space = space
        self.size = size
        self.bodies = []
        self.springs = []
        self.base_lengths = []
        # self.theta      # current phase
        # self.omega      # natural frequency
        # self.amplitude  # contraction strength

        self.theta = random.uniform(0, 2*math.pi)
        self.omega = 2.0
        self.amplitude = 3.0

        offsets = [
            (-size/2, -size/2),
            ( size/2, -size/2),
            ( size/2,  size/2),
            (-size/2,  size/2),
        ]

        for dx, dy in offsets:
            body = pymunk.Body(1, pymunk.moment_for_box(10, (5,5)))
            body.position = x + dx, y + dy
            shape = pymunk.Poly.create_box(body, (5, 5))

            shape.friction = SHAPE_FRICTION
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

    def modulate(self, t, amplitude=8, frequency=10):
        for spring, base in zip(self.springs, self.base_lengths):
            new_length = base + amplitude * math.sin(frequency * t)
            spring.rest_length = max(5, new_length)

    def apply_scale(self, scale):
        for spring, base in zip(self.springs, self.base_lengths):
            spring.rest_length = max(5, base * scale)

