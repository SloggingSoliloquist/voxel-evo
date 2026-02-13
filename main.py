# main.py

import pygame
import pymunk.pygame_util
from config import *
from world import create_space, create_ground
from grid import build_grid

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

space = create_space(GRAVITY)
create_ground(space)

draw_options = pymunk.pygame_util.DrawOptions(screen)

voxels = build_grid(space, start_x=200, start_y=300)

running = True
t = 0

while running:
    dt = 1 / FPS
    t += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for row in voxels:
        for voxel in row:
            voxel.modulate(t, amplitude=10, frequency=3)

    for _ in range(SUBSTEPS):
        space.step(dt / SUBSTEPS)

    screen.fill((30,30,30))
    space.debug_draw(draw_options)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
