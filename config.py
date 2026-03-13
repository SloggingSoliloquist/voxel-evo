# config.py

WIDTH  = 900
HEIGHT = 600

VOXEL_SIZE = 20  # scaled down from 40

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
    [1, 2, 1, 2, 1, 2, 1, 2]
]

# Scale physics relative to original VOXEL_SIZE of 40
_S = VOXEL_SIZE / 40  # 0.5 at VOXEL_SIZE=20

SPRING_STIFFNESS = int(500 * _S)        # 250
SPRING_DAMPING   = 20 * _S              # 10
GRAVITY          = (0, 900)             # unchanged

SUBSTEPS = 3
FPS      = 60

GROUND_FRICTION = 2.0
SHAPE_FRICTION  = 1.5