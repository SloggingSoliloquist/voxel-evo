# world.py

import pymunk
from config import WIDTH

def create_space(gravity):
    space = pymunk.Space()
    space.gravity = gravity
    space.iterations = 30
    return space


def create_ground(space):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (WIDTH // 2, 20)

    shape = pymunk.Poly.create_box(body, (WIDTH, 40))
    shape.friction = 1.0
    space.add(body, shape)
