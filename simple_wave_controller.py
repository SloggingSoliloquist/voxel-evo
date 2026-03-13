# controller.py

import math
from voxel import MUSCLE_A, MUSCLE_B

class WaveController:
    def __init__(self, rows, cols, amplitude=0.6, frequency=3, phase_offset=0.5):
        self.rows = rows
        self.cols = cols
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_offset = phase_offset

    def get_scale(self, r, c, t, voxel_type=MUSCLE_A):
        phase = c * self.phase_offset

        # MUSCLE_B is always pi out of phase with MUSCLE_A
        if voxel_type == MUSCLE_B:
            phase += math.pi

        return 1 + self.amplitude * math.sin(self.frequency * t + phase)