import pymunk
from config import WIDTH, HEIGHT, GROUND_FRICTION

terrain_points = []

def create_space(gravity):
    """Create a Pymunk space with given gravity."""
    space = pymunk.Space()
    space.gravity = gravity
    space.iterations = 30
    return space

def init_ground(space, start_x=0, start_y=100, length=600, step=40):
    """Initialize a flat ground for scrolling."""
    global terrain_points
    terrain_points.clear()

    x = start_x
    y = start_y
    terrain_points.append((x, y))

    for _ in range(length // step):
        x_new = x + step
        seg = pymunk.Segment(space.static_body, (x, y), (x_new, y), 3)
        seg.friction = GROUND_FRICTION
        space.add(seg)
        terrain_points.append((x_new, y))
        x = x_new

    return y  # Return the top y-coordinate for spawning objects above

def extend_ground(space, count=20, step=40):
    """Add more segments to the ground procedurally."""
    global terrain_points
    x, y = terrain_points[-1]
    segments = []

    for _ in range(count):
        x_new = x + step
        seg = pymunk.Segment(space.static_body, (x, y), (x_new, y), 3)
        seg.friction = GROUND_FRICTION
        space.add(seg)
        terrain_points.append((x_new, y))
        segments.append(seg)
        x = x_new

    return segments

def terrain_end_x():
    """Return the x-coordinate of the last terrain point."""
    return terrain_points[-1][0] if terrain_points else 0
