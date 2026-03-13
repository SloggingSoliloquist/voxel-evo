# evaluator.py

import pygame
import pymunk
import math
from config import (
    WIDTH, HEIGHT, VOXEL_SIZE, GRAVITY, SUBSTEPS, FPS,
    GROUND_FRICTION, SHAPE_FRICTION, SPRING_STIFFNESS, SPRING_DAMPING
)
from world import create_space, init_ground, extend_ground, terrain_end_x
from world import terrain_points
from grid import build_grid
from simple_wave_controller import WaveController
from voxel import MUSCLE_A, MUSCLE_B

EVAL_DURATION = 10.0   # seconds per genome evaluation
SPAWN_X       = 200
SPAWN_Y       = 0


def get_robot_x(voxels):
    all_bodies = [b for row in voxels for v in row if v is not None for b in v.bodies]
    if not all_bodies:
        return 0.0
    return sum(b.position.x for b in all_bodies) / len(all_bodies)


def get_head_body(voxels):
    for row in voxels:
        for v in row:
            if v is not None:
                return v.bodies[0]


def draw_ground(screen, terrain_pts, camera_x, camera_y):
    if len(terrain_pts) < 2:
        return
    screen_pts = [(x - camera_x, y - camera_y) for x, y in terrain_pts]
    bottom = HEIGHT + 100
    filled_pts = screen_pts + [(screen_pts[-1][0], bottom), (screen_pts[0][0], bottom)]
    pygame.draw.polygon(screen, (80, 80, 80), filled_pts)
    pygame.draw.lines(screen, (180, 180, 180), False, screen_pts, 2)


def draw_robot(screen, voxels, camera_x, camera_y):
    for row in voxels:
        for voxel in row:
            if voxel is None:
                continue
            def to_screen(body):
                return (body.position.x - camera_x, body.position.y - camera_y)
            pts = [
                to_screen(voxel.tl),
                to_screen(voxel.tr),
                to_screen(voxel.br),
                to_screen(voxel.bl),
            ]
            pygame.draw.polygon(screen, voxel.get_color(), pts)
            pygame.draw.polygon(screen, (200, 200, 200), pts, 1)


def evaluate(genome, rows, cols, screen, font,
             generation=0, individual=0, population_size=0):
    """
    Run one genome for EVAL_DURATION seconds.
    Returns fitness (distance travelled in pixels).

    genome: flat list of ints, length rows*cols
    screen/font: shared pygame surface and font (created once by the caller)
    """
    # Reshape genome into 2D morphology
    morphology = [
        genome[r * cols:(r + 1) * cols]
        for r in range(rows)
    ]

    # Fresh physics space
    space = create_space(GRAVITY)
    init_ground(space, start_x=0, start_y=200, length=800)

    voxels = build_grid(space, start_x=SPAWN_X, start_y=SPAWN_Y,
                        morphology=morphology)

    active_voxels = sum(1 for row in voxels for v in row if v is not None)

    controller = WaveController(rows, cols, amplitude=0.6, frequency=3, phase_offset=0.5)

    clock = pygame.time.Clock()
    t = 0.0
    eval_t = 0.0
    camera_x = 0.0
    camera_y = 0.0
    start_x = None
    fitness = 0.0

    while eval_t < EVAL_DURATION:
        dt = 1.0 / FPS
        t += dt
        eval_t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        if start_x is None:
            start_x = get_robot_x(voxels)

        # Actuate muscle voxels
        for r, row in enumerate(voxels):
            for c, voxel in enumerate(row):
                if voxel is None:
                    continue
                if voxel.voxel_type not in (MUSCLE_A, MUSCLE_B):
                    continue
                scale = controller.get_scale(r, c, t, voxel_type=voxel.voxel_type)
                voxel.apply_scale(scale)

        for _ in range(SUBSTEPS):
            space.step(dt / SUBSTEPS)

        current_x = get_robot_x(voxels)
        raw_fitness = current_x - start_x
        fitness = raw_fitness / active_voxels if active_voxels > 0 else 0.0

        # Camera
        head = get_head_body(voxels)
        if head:
            target_x = head.position.x + 100 - WIDTH / 3
            target_y = head.position.y - HEIGHT / 2
            camera_x += (target_x - camera_x) * 0.1
            camera_y += (target_y - camera_y) * 0.1

        # Extend ground
        if terrain_end_x() - (head.position.x if head else 0) < 600:
            extend_ground(space, count=20)

        # Draw
        screen.fill((20, 20, 20))
        draw_ground(screen, terrain_points, camera_x, camera_y)
        draw_robot(screen, voxels, camera_x, camera_y)

        # HUD
        elapsed_pct = min(eval_t / EVAL_DURATION, 1.0)
        hud_lines = [
            f"Generation {generation}  |  Individual {individual}/{population_size}",
            f"Fitness (norm): {fitness:.2f} px/voxel  [{active_voxels} voxels]",
            f"Eval: {eval_t:.1f}s / {EVAL_DURATION:.0f}s  [{int(elapsed_pct*100)}%]",
        ]
        for i, line in enumerate(hud_lines):
            surf = font.render(line, True, (220, 220, 100))
            screen.blit(surf, (12, 12 + i * 22))

        bar_w = 200
        bar_h = 10
        pygame.draw.rect(screen, (80, 80, 80),
                         (12, 12 + len(hud_lines) * 22, bar_w, bar_h))
        pygame.draw.rect(screen, (100, 220, 100),
                         (12, 12 + len(hud_lines) * 22, int(bar_w * elapsed_pct), bar_h))

        pygame.display.flip()
        clock.tick(FPS)

    return fitness