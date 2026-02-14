import math, random
class RandomController:
    def __init__(self, rows, cols, amplitude=0.2):
        self.rows = rows
        self.cols = cols
        self.amplitude = amplitude

        self.thetas = [
            [random.uniform(0, 2*math.pi) for _ in range(cols)]
            for _ in range(rows)
        ]

        self.omegas = [
            [random.uniform(1.0, 3.0) for _ in range(cols)]
            for _ in range(rows)
        ]

    def update(self, dt):
        for r in range(self.rows):
            for c in range(self.cols):
                self.thetas[r][c] += self.omegas[r][c] * dt

    def get_scale(self, r, c):
        return 1 + self.amplitude * math.sin(self.thetas[r][c])
