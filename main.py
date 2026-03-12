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
font = pygame.font.SysFont("monospace", 18)

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

# --- Fitness tracking ---
def get_head_body(voxels):
    return voxels[0][0].bodies[0]

def get_robot_x(voxels):
    """Average x position across all voxel bodies."""
    all_bodies = [b for row in voxels for v in row for b in v.bodies]
    return sum(b.position.x for b in all_bodies) / len(all_bodies)

start_x = None
best_fitness = 0.0
fitness = 0.0
EVAL_DURATION = 10.0  # seconds per evaluation window

while running:
    dt = 1 / FPS
    t += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Record start position ---
    if start_x is None:
        start_x = get_robot_x(voxels)

    # --- Update wave controller ---
    for r, row in enumerate(voxels):
        for c, voxel in enumerate(row):
            scale = controller.get_scale(r, c, t)
            voxel.apply_scale(scale)

    # --- Step physics ---
    for _ in range(SUBSTEPS):
        space.step(dt / SUBSTEPS)

    # --- Fitness: distance traveled from start ---
    current_x = get_robot_x(voxels)
    fitness = current_x - start_x
    best_fitness = max(best_fitness, fitness)

    # --- Reset eval window every EVAL_DURATION seconds ---
    if t >= EVAL_DURATION:
        print(f"[Eval complete] Fitness: {fitness:.2f} px | Best: {best_fitness:.2f} px")
        t = 0
        start_x = current_x  # reset from current position

    # --- Camera follow ---
    head = get_head_body(voxels)
    target_x = head.position.x + 100 - WIDTH / 3
    target_y = head.position.y - HEIGHT / 2
    camera_x += (target_x - camera_x) * 0.1
    camera_y += (target_y - camera_y) * 0.1

    # --- Extend ground if needed ---
    if terrain_end_x() - head.position.x < 600:
        extend_ground(space, count=20)

    # --- Draw ---
    screen.fill((30, 30, 30))
    draw_options.transform = pymunk.Transform.translation(-camera_x, -camera_y)
    space.debug_draw(draw_options)

    # --- HUD overlay ---
    elapsed_pct = min(t / EVAL_DURATION, 1.0)
    hud_lines = [
        f"Fitness (distance): {fitness:.1f} px",
        f"Best this session:  {best_fitness:.1f} px",
        f"Eval time: {t:.1f}s / {EVAL_DURATION:.0f}s  [{int(elapsed_pct*100)}%]",
    ]
    for i, line in enumerate(hud_lines):
        surf = font.render(line, True, (220, 220, 100))
        screen.blit(surf, (12, 12 + i * 22))

    # --- Eval progress bar ---
    bar_w = 200
    bar_h = 10
    pygame.draw.rect(screen, (80, 80, 80), (12, 12 + len(hud_lines) * 22, bar_w, bar_h))
    pygame.draw.rect(screen, (100, 220, 100), (12, 12 + len(hud_lines) * 22, int(bar_w * elapsed_pct), bar_h))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()