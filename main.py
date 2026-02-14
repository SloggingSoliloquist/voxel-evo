import pygame
import pymunk
import pymunk.pygame_util
from config import *
from world import create_space, init_ground, extend_ground, terrain_end_x, terrain_points
from grid import build_grid
from simple_wave_controller import WaveController

controller = WaveController(ROWS, COLS, amplitude=0.4, frequency=3, phase_offset=0.6)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# --- Physics space ---
space = create_space(GRAVITY)

# --- Initialize flat ground ---
GROUND_Y = init_ground(space, start_x=0, start_y=200, length=800)

# --- Draw options for Pygame ---
draw_options = pymunk.pygame_util.DrawOptions(screen)

# --- Spawn the worm slightly above the top of the ground ---
voxels = build_grid(space, start_x=200, start_y=0)

running = True
t = 0

camera_x = 0
camera_y = 0

# --- Camera helper: use head of worm ---
def get_head_body(voxels):
    # First voxel, first body as reference
    return voxels[0][0].bodies[0]

while running:
    dt = 1 / FPS
    t += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Update wave controller ---
    for r, row in enumerate(voxels):
        for c, voxel in enumerate(row):
            scale = controller.get_scale(r, c, t)
            voxel.apply_scale(scale)

    # --- Step physics ---
    for _ in range(SUBSTEPS):
        space.step(dt / SUBSTEPS)

    # --- Camera follow ---
    head = get_head_body(voxels)
    target_x = head.position.x + 100 - WIDTH / 3  # look ahead
    target_y = head.position.y - HEIGHT / 2

    # Smooth follow
    camera_x += (target_x - camera_x) * 0.1
    camera_y += (target_y - camera_y) * 0.1

    # --- Extend ground if needed ---
    if terrain_end_x() - head.position.x < 600:
        extend_ground(space, count=20)

    # --- Draw ---
    screen.fill((30, 30, 30))
    draw_options.transform = pymunk.Transform.translation(-camera_x, -camera_y)
    space.debug_draw(draw_options)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
