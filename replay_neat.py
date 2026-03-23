# replay_neat.py
# Watch any saved genome from a NEAT run.
# Usage: python replay_neat.py neat_logs/winner.pkl

import sys
import pickle
import pygame
import neat
from config import ROWS, COLS, WIDTH, HEIGHT, FPS, GRAVITY, VOXEL_SIZE, SUBSTEPS
from world import create_space, init_ground, extend_ground, terrain_end_x, terrain_points
from grid import build_grid
from cppn_neat_decoder import build_network, decode_morphology, get_scale
from voxel import MUSCLE_A, MUSCLE_B

CONFIG_PATH = "neat_config.cfg"
GROUND_Y_FRAC = 0.75


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


def replay(genome_path):
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    net = build_network(genome, config)
    morphology = decode_morphology(net, ROWS, COLS)

    print("Morphology:")
    for row in morphology:
        print(row)
    print(f"Nodes: {len(genome.nodes)} | Connections: {len(genome.connections)}")
    print(f"Fitness: {genome.fitness}")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Replay: {genome_path}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    space = create_space(GRAVITY)
    ground_y = int(HEIGHT * GROUND_Y_FRAC)
    init_ground(space, start_x=0, start_y=ground_y, length=WIDTH * 3)

    robot_height = ROWS * VOXEL_SIZE
    spawn_y = ground_y - robot_height - 5
    voxels = build_grid(space, start_x=200, start_y=spawn_y, morphology=morphology)

    t = 0.0
    camera_x, camera_y = 0.0, 0.0
    start_x = None
    running = True

    while running:
        dt = 1.0 / FPS
        t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if start_x is None:
            start_x = get_robot_x(voxels)

        for r, row in enumerate(voxels):
            for c, voxel in enumerate(row):
                if voxel is None:
                    continue
                if voxel.voxel_type not in (MUSCLE_A, MUSCLE_B):
                    continue
                scale = get_scale(net, r, c, t, rows=ROWS, cols=COLS)
                voxel.apply_scale(scale)

        for _ in range(SUBSTEPS):
            space.step(dt / SUBSTEPS)

        fitness = get_robot_x(voxels) - start_x

        head = get_head_body(voxels)
        if head:
            target_x = head.position.x - WIDTH / 2
            target_y = head.position.y - int(HEIGHT * 0.4)
            camera_x += (target_x - camera_x) * 0.1
            camera_y += (target_y - camera_y) * 0.1

        if terrain_end_x() - (head.position.x if head else 0) < WIDTH:
            extend_ground(space, count=30)

        screen.fill((20, 20, 20))

        # Draw ground
        if len(terrain_points) >= 2:
            pts = [(x - camera_x, y - camera_y) for x, y in terrain_points]
            filled = pts + [(pts[-1][0], HEIGHT+100), (pts[0][0], HEIGHT+100)]
            pygame.draw.polygon(screen, (80, 80, 80), filled)
            pygame.draw.lines(screen, (180, 180, 180), False, pts, 2)

        # Draw robot
        for row in voxels:
            for voxel in row:
                if voxel is None:
                    continue
                def to_screen(body):
                    return (body.position.x - camera_x, body.position.y - camera_y)
                pts = [to_screen(voxel.tl), to_screen(voxel.tr),
                       to_screen(voxel.br), to_screen(voxel.bl)]
                pygame.draw.polygon(screen, voxel.get_color(), pts)
                pygame.draw.polygon(screen, (200, 200, 200), pts, 1)

        hud = [
            f"Fitness: {fitness:.1f}px",
            f"Nodes: {len(genome.nodes)} | Conns: {len(genome.connections)}",
            f"t={t:.1f}s  |  ESC to quit",
        ]
        for i, line in enumerate(hud):
            surf = font.render(line, True, (220, 220, 100))
            screen.blit(surf, (12, 12 + i * 22))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "neat_logs/winner.pkl"
    replay(path)