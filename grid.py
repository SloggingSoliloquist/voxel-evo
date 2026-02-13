# grid.py

from voxel import Voxel
from config import VOXEL_SIZE, ROWS, COLS, SPRING_STIFFNESS, SPRING_DAMPING
import pymunk


def build_grid(space, start_x, start_y):
    voxels = []

    for r in range(ROWS):
        row = []
        for c in range(COLS):
            x = start_x + c * VOXEL_SIZE
            y = start_y - r * VOXEL_SIZE
            voxel = Voxel(space, x, y, VOXEL_SIZE)
            row.append(voxel)
        voxels.append(row)

    _connect_neighbors(space, voxels)

    return voxels


def _connect_neighbors(space, voxels):
    for r in range(len(voxels)):
        for c in range(len(voxels[0])):
            if c < len(voxels[0]) - 1:
                connect_voxels(space, voxels[r][c], voxels[r][c+1])
            if r < len(voxels) - 1:
                connect_voxels(space, voxels[r][c], voxels[r+1][c])


def connect_voxels(space, v1, v2):
    for b1 in v1.bodies:
        for b2 in v2.bodies:
            dist = (b1.position - b2.position).length
            if dist < 10:
                spring = pymunk.DampedSpring(
                    b1, b2,
                    (0,0),(0,0),
                    dist,
                    SPRING_STIFFNESS,
                    SPRING_DAMPING
                )
                space.add(spring)
