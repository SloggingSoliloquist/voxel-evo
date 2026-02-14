import math
import random

class CPGController:
    def __init__(self, rows, cols, omega=2.0, K=1.0, amplitude=3.0):
        self.rows = rows
        self.cols = cols
        self.omega = omega
        self.K = K
        self.amplitude = amplitude

        # phase grid
        self.thetas = [
            [random.uniform(0, 2*math.pi) for _ in range(cols)]
            for _ in range(rows)
        ]

    def get_neighbors(self, r, c):
        neighbors = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors

    def update(self, dt):
        new_thetas = [
            row[:] for row in self.thetas
        ]

        for r in range(self.rows):
            for c in range(self.cols):
                theta = self.thetas[r][c]

                coupling_sum = 0
                delta=math.pi
                for nr, nc in self.get_neighbors(r, c):
                    neighbor_theta = self.thetas[nr][nc]
                    coupling_sum += math.sin(neighbor_theta - theta-delta)

                dtheta = self.omega + self.K * coupling_sum
                new_thetas[r][c] += dtheta * dt

        self.thetas = new_thetas

    def get_scale(self, r, c):
        theta = self.thetas[r][c]
        return 1 + self.amplitude * math.sin(theta)
