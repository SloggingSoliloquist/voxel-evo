# grid.py

import pymunk
from voxel import Voxel, EMPTY, MUSCLE_A, MUSCLE_B, SOFT, RIGID
from config import VOXEL_SIZE, SPRING_STIFFNESS, SPRING_DAMPING, SHAPE_FRICTION, MORPHOLOGY


def _make_node(space, x, y):
    mass = (VOXEL_SIZE / 40) ** 2  # 0.25 at VOXEL_SIZE=20
    moment = pymunk.moment_for_box(mass, (VOXEL_SIZE / 10, VOXEL_SIZE / 10))
    body = pymunk.Body(mass, moment)
    body.position = x, y
    shape = pymunk.Poly.create_box(body, (VOXEL_SIZE / 10, VOXEL_SIZE / 10))
    shape.friction = SHAPE_FRICTION
    space.add(body, shape)
    return body


def build_grid(space, start_x, start_y, morphology=None):
    if morphology is None:
        morphology = MORPHOLOGY

    rows = len(morphology)
    cols = len(morphology[0])

    # --- Shared node grid ---
    used = set()
    for r in range(rows):
        for c in range(cols):
            if morphology[r][c] != EMPTY:
                used.add((r,     c))
                used.add((r,     c + 1))
                used.add((r + 1, c))
                used.add((r + 1, c + 1))

    nodes = {}
    for (r, c) in used:
        x = start_x + c * VOXEL_SIZE
        y = start_y + r * VOXEL_SIZE
        nodes[(r, c)] = _make_node(space, x, y)

    # --- Build voxels ---
    voxels = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cell = morphology[r][c]
            if cell == EMPTY:
                row.append(None)
                continue
            voxel = Voxel(
                space,
                tl=nodes[(r,     c)],
                tr=nodes[(r,     c + 1)],
                bl=nodes[(r + 1, c)],
                br=nodes[(r + 1, c + 1)],
                size=VOXEL_SIZE,
                voxel_type=cell,
            )
            row.append(voxel)
        voxels.append(row)

    return voxels