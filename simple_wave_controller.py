# controller.py

import math

class WaveController:
    def __init__(self, rows, cols, amplitude=0.6, frequency=12, phase_offset=1.2):
        self.rows = rows
        self.cols = cols
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_offset = phase_offset

    def get_scale(self, r, c, t):
        phase = c * self.phase_offset
        return 1 + self.amplitude * math.sin(self.frequency * t + phase)
