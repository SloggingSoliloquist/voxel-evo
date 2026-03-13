# config.py

WIDTH  = 900
HEIGHT = 600

VOXEL_SIZE = 20

ROWS = 8
COLS = 8
VOXEL_PD = [0.2, 0.3, 0.3, 0.1, 0.1]

# Voxel types
EMPTY    = 0
MUSCLE_A = 1
MUSCLE_B = 2
SOFT     = 3
RIGID    = 4

MORPHOLOGY = [
    [4, 4, 4, 4, 4, 4, 4, 4],
    [1, 2, 1, 2, 1, 2, 1, 2],
    [1, 2, 1, 2, 0, 2, 1, 2],
    [1, 2, 1, 2, 0, 2, 1, 2],
    [1, 2, 1, 0, 0, 2, 1, 2],
    [1, 2, 0, 0, 0, 2, 1, 2]
]

_S = VOXEL_SIZE / 40  # 0.5

SPRING_STIFFNESS = int(500 * _S * 4)   # 4x multiplier to support more rows
SPRING_DAMPING   = 20 * _S * 2         # higher damping to prevent oscillation
GRAVITY          = (0, 400)            # reduced from 900 — less crushing force

SUBSTEPS = 10                          # up from 3 — more stable with tall structures
FPS      = 60

GROUND_FRICTION = 2.0
SHAPE_FRICTION  = 1.5