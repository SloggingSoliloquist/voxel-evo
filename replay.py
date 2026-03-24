# replay.py
# Load the best genome from a generation log and replay it visually with a trained PPO policy.
# Usage: python replay.py evo_logs/gen_0050.json

import sys
import json
import math
import pygame
import pymunk
import torch
from config import WIDTH, HEIGHT, VOXEL_SIZE, GRAVITY, SUBSTEPS, FPS, ROWS, COLS
from world import create_space, init_ground, extend_ground, terrain_end_x, terrain_points
from grid import build_grid
from cppn_morphology import CPPN
from ppo import ActorCritic, train_ppo
from env import VoxelEnv, OBS_SIZE, ACTION_SIZE
from voxel import MUSCLE_A, MUSCLE_B

GROUND_Y_FRAC = 0.75
REPLAY_DURATION = 20.0   # seconds to replay


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


def replay(log_path):
    # --- Load log ---
    with open(log_path) as f:
        data = json.load(f)

    genome     = data["best_genome"]
    morphology = data["best_morphology"]
    gen        = data["generation"]
    fitness    = data["best_fitness"]

    print(f"Replaying gen {gen} | fitness {fitness:.4f}")
    policy_path = data.get("policy_path")
    if not policy_path:
        print("No policy path in log file. Run evolution again to generate saved weights.")
        sys.exit(1)

    policy = ActorCritic(OBS_SIZE, ACTION_SIZE)
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()
    print(f"Loaded policy from {policy_path}")

    # --- Build visual sim ---
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Replay — Gen {gen} | Fitness {fitness:.4f}")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 18)

    rows = len(morphology)
    cols = len(morphology[0])

    space    = create_space(GRAVITY)
    ground_y = int(HEIGHT * GROUND_Y_FRAC)
    init_ground(space, start_x=0, start_y=ground_y, length=WIDTH * 6)

    robot_height = rows * VOXEL_SIZE
    spawn_y      = ground_y - robot_height - 5
    voxels       = build_grid(space, start_x=200, start_y=spawn_y, morphology=morphology)

    # Collect nodes for observation
    seen  = set()
    nodes = []
    for row in voxels:
        for v in row:
            if v is None:
                continue
            for b in v.bodies:
                if id(b) not in seen:
                    seen.add(id(b))
                    nodes.append(b)

    muscle_positions = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if morphology[r][c] in (MUSCLE_A, MUSCLE_B)
    ]

    from env import MAX_NODES, OBS_SIZE as OBS
    def observe():
        obs = [0.0] * OBS
        if not nodes:
            return obs
        com_x = sum(b.position.x for b in nodes) / len(nodes)
        com_y = sum(b.position.y for b in nodes) / len(nodes)
        for i, body in enumerate(nodes):
            if i >= MAX_NODES:
                break
            base = i * 4
            obs[base]     = (body.position.x - com_x) / VOXEL_SIZE
            obs[base + 1] = (body.position.y - com_y) / VOXEL_SIZE
            obs[base + 2] = body.velocity.x / 100.0
            obs[base + 3] = body.velocity.y / 100.0
        return obs

    running    = True
    t          = 0.0
    camera_x   = 0.0
    camera_y   = 0.0
    start_x    = get_robot_x(voxels)
    dt         = 1.0 / FPS

    while running and t < REPLAY_DURATION:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        obs_t  = torch.FloatTensor(observe()).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = policy.get_action(obs_t)
        action_np = action.squeeze(0).cpu().numpy()

        for idx, (r, c) in enumerate(muscle_positions):
            if idx >= len(action_np):
                break
            scale = 0.6 + float(action_np[idx]) * 1.0
            v = voxels[r][c]
            if v is not None:
                v.apply_scale(scale)

        for _ in range(SUBSTEPS):
            space.step(dt / SUBSTEPS)
        t += dt

        head = get_head_body(voxels)
        if head:
            target_x = head.position.x - WIDTH / 2
            target_y = head.position.y - int(HEIGHT * 0.4)
            camera_x += (target_x - camera_x) * 0.1
            camera_y += (target_y - camera_y) * 0.1
            if terrain_end_x() - head.position.x < WIDTH * 2:
                extend_ground(space, count=50)

        current_x = get_robot_x(voxels)
        distance  = current_x - start_x

        screen.fill((20, 20, 20))
        draw_ground(screen, terrain_points, camera_x, camera_y)
        draw_robot(screen, voxels, camera_x, camera_y)

        hud = [
            f"Gen {gen}  |  PPO fitness: {fitness:.4f}",
            f"Distance: {distance:.1f} px",
            f"Time: {t:.1f}s / {REPLAY_DURATION:.0f}s",
        ]
        for i, line in enumerate(hud):
            surf = font.render(line, True, (220, 220, 100))
            screen.blit(surf, (12, 12 + i * 22))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replay.py <path_to_gen_XXXX.json>")
        sys.exit(1)
    replay(sys.argv[1])