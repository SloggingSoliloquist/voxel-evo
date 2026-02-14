# main.py

import pygame
import pymunk.pygame_util
from config import *
from world import create_space, create_ground
from grid import build_grid
from random_controller import RandomController
controller = RandomController(ROWS, COLS, amplitude=0.2)

#controller = CPGController(ROWS, COLS, omega=2.0, K=1.0, amplitude=0.2)

#controller = WaveController(ROWS, COLS, amplitude=0.2, frequency=3, phase_offset=0.6)


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

    controller.update(dt)

    for r, row in enumerate(voxels):
        for c, voxel in enumerate(row):
            scale = controller.get_scale(r, c)
            voxel.apply_scale(scale)



    for _ in range(SUBSTEPS):
        space.step(dt / SUBSTEPS)

    screen.fill((30,30,30))
    space.debug_draw(draw_options)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
