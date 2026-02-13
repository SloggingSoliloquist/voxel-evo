import pygame
import pymunk
import pymunk.pygame_util
import math

WIDTH, HEIGHT = 900, 600
VOXEL_SIZE = 40
ROWS = 2
COLS = 8

SPRING_STIFFNESS = 300
SPRING_DAMPING = 50

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = (0, -900)
space.iterations = 30

draw_options = pymunk.pygame_util.DrawOptions(screen)


# ---------------- Ground ----------------

def create_ground():
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (WIDTH // 2, 20)


    shape = pymunk.Poly.create_box(body, (WIDTH, 40))
    shape.friction = 1.0
    space.add(body, shape)

create_ground()


# ---------------- Voxel ----------------

class Voxel:
    def __init__(self, x, y, size):
        self.size = size
        self.bodies = []
        self.springs = []
        self.base_lengths = []

        offsets = [
            (-size/2, -size/2),
            ( size/2, -size/2),
            ( size/2,  size/2),
            (-size/2,  size/2),
        ]

        for dx, dy in offsets:
            body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
            body.position = x + dx, y + dy
            shape = pymunk.Circle(body, 5)
            shape.friction = 1.0
            space.add(body, shape)
            self.bodies.append(body)

        self.connect_internal_springs()

    def connect_internal_springs(self):
        pairs = [
            (0,1),(1,2),(2,3),(3,0),  # edges
            (0,2),(1,3)               # diagonals
        ]

        for i,j in pairs:
            if abs(i-j)==1 or abs(i-j)==3:
                length = self.size
            else:
                length = self.size * math.sqrt(2)

            spring = pymunk.DampedSpring(
                self.bodies[i],
                self.bodies[j],
                (0,0),(0,0),
                length,
                SPRING_STIFFNESS,
                SPRING_DAMPING
            )

            space.add(spring)
            self.springs.append(spring)
            self.base_lengths.append(length)

    def modulate(self, t, amplitude=3, frequency=2):
        for spring, base in zip(self.springs, self.base_lengths):
            new_length = base + amplitude * math.sin(frequency * t)
            spring.rest_length = max(5, new_length)


# ---------------- Build Grid ----------------

voxels = []

start_x = 200
start_y = 300

for r in range(ROWS):
    row = []
    for c in range(COLS):
        x = start_x + c * VOXEL_SIZE
        y = start_y - r * VOXEL_SIZE
        voxel = Voxel(x, y, VOXEL_SIZE)
        row.append(voxel)
    voxels.append(row)


# ---------------- Connect Neighbor Voxels ----------------

def connect_voxels(v1, v2):
    for b1 in v1.bodies:
        for b2 in v2.bodies:
            dist = (b1.position - b2.position).length
            if dist < 10:
                spring = pymunk.DampedSpring(
                    b1, b2,
                    (0,0),(0,0),
                    dist,  # <-- use actual distance
                    SPRING_STIFFNESS,
                    SPRING_DAMPING
                )
                space.add(spring)

for r in range(ROWS):
    for c in range(COLS):
        if c < COLS-1:
            connect_voxels(voxels[r][c], voxels[r][c+1])
        if r < ROWS-1:
            connect_voxels(voxels[r][c], voxels[r+1][c])


# ---------------- Main Loop ----------------

running = True
t = 0

while running:
    dt = 1/60
    t += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Modulate voxels
    for row in voxels:
        for voxel in row:
            voxel.modulate(t, amplitude=10, frequency=3)

    # Substepping for stability
    for _ in range(3):
        space.step(dt / 3)

    screen.fill((30,30,30))
    space.debug_draw(draw_options)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
