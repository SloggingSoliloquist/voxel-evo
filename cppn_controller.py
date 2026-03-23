# cppn_controller.py

import math
import random
from config import ROWS, COLS

# --- Network architecture ---
INPUT_SIZE  = 5   # (x, y, d, theta, bias)
HIDDEN_SIZE = 16
OUTPUT_SIZE = 4   # (muscle_probability, frequency, phase, amplitude)

# --- Activation functions (same set as morphology CPPN) ---
def sigmoid(x):
    x = max(-20, min(20, x))
    return 1.0 / (1.0 + math.exp(-x))

def tanh_act(x):
    return math.tanh(x)

def sin_act(x):
    return math.sin(x)

def cos_act(x):
    return math.cos(x)

def gauss_act(x):
    return math.exp(-x * x)

def abs_act(x):
    return abs(x)

def linear(x):
    return x

ACTIVATION_FUNCTIONS = [tanh_act, sin_act, cos_act, gauss_act, abs_act, linear, sigmoid]
N_ACTIVATIONS = len(ACTIVATION_FUNCTIONS)

# Threshold above which a muscle voxel is actually activated
MUSCLE_PROB_THRESHOLD = 0.5


class CPPNController:
    """
    Fixed-topology CPPN controller: 5 -> 16 -> 16 -> 4

    Inputs: (x, y, d, theta, bias)

    Outputs:
        muscle_probability : sigmoid → [0,1], thresholded to decide if voxel fires
        frequency          : 0.5 + sigmoid * 5.5 → [0.5, 6.0] Hz
        phase              : tanh * π → [-π, π]
        amplitude          : 0.2 + sigmoid * 0.6 → [0.2, 0.8]

    Genome layout (identical structure to morphology CPPN):
        W1, b1, acts1 / W2, b2, acts2 / W3, b3
    """

    def __init__(self, weights=None):
        self.genome_length = (
            INPUT_SIZE  * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE +
            HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE +
            HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE
        )

        if weights is not None:
            assert len(weights) == self.genome_length, \
                f"Expected {self.genome_length} weights, got {len(weights)}"
            self.weights = list(weights)
        else:
            self.weights = self._random_weights()

        self._unpack()

    def _random_weights(self):
        w = []
        for _ in range(INPUT_SIZE * HIDDEN_SIZE):
            w.append(random.gauss(0, 1.0))
        for _ in range(HIDDEN_SIZE):
            w.append(random.gauss(0, 1.0))
        for _ in range(HIDDEN_SIZE):
            w.append(random.uniform(0, N_ACTIVATIONS))
        for _ in range(HIDDEN_SIZE * HIDDEN_SIZE):
            w.append(random.gauss(0, 1.0))
        for _ in range(HIDDEN_SIZE):
            w.append(random.gauss(0, 1.0))
        for _ in range(HIDDEN_SIZE):
            w.append(random.uniform(0, N_ACTIVATIONS))
        for _ in range(HIDDEN_SIZE * OUTPUT_SIZE):
            w.append(random.gauss(0, 1.0))
        for _ in range(OUTPUT_SIZE):
            w.append(random.gauss(0, 1.0))
        return w

    def _unpack(self):
        w = self.weights
        idx = 0

        def take(n):
            nonlocal idx
            chunk = w[idx:idx + n]
            idx += n
            return chunk

        self.W1    = take(INPUT_SIZE  * HIDDEN_SIZE)
        self.b1    = take(HIDDEN_SIZE)
        self.acts1 = [ACTIVATION_FUNCTIONS[int(a) % N_ACTIVATIONS] for a in take(HIDDEN_SIZE)]
        self.W2    = take(HIDDEN_SIZE * HIDDEN_SIZE)
        self.b2    = take(HIDDEN_SIZE)
        self.acts2 = [ACTIVATION_FUNCTIONS[int(a) % N_ACTIVATIONS] for a in take(HIDDEN_SIZE)]
        self.W3    = take(HIDDEN_SIZE * OUTPUT_SIZE)
        self.b3    = take(OUTPUT_SIZE)

    def _matmul_add(self, W, b, x, in_size, out_size):
        result = []
        for i in range(out_size):
            val = b[i]
            for j in range(in_size):
                val += W[i * in_size + j] * x[j]
            result.append(val)
        return result

    def forward(self, x, y):
        """
        Query controller at normalized (x, y).
        Returns (active, frequency, phase, amplitude).
        active is bool — whether this voxel should fire at all.
        """
        d     = math.sqrt(x * x + y * y) / math.sqrt(2)
        theta = math.atan2(y, x)
        inp   = [x, y, d, theta, 1.0]

        h1_pre = self._matmul_add(self.W1, self.b1, inp, INPUT_SIZE, HIDDEN_SIZE)
        h1 = [self.acts1[i](v) for i, v in enumerate(h1_pre)]

        h2_pre = self._matmul_add(self.W2, self.b2, h1, HIDDEN_SIZE, HIDDEN_SIZE)
        h2 = [self.acts2[i](v) for i, v in enumerate(h2_pre)]

        out = self._matmul_add(self.W3, self.b3, h2, HIDDEN_SIZE, OUTPUT_SIZE)

        muscle_prob = sigmoid(out[0])
        active      = muscle_prob >= MUSCLE_PROB_THRESHOLD
        frequency   = 0.5 + sigmoid(out[1]) * 5.5   # [0.5, 6.0]
        phase       = math.tanh(out[2]) * math.pi    # [-π, π]
        amplitude   = 0.2 + sigmoid(out[3]) * 0.6    # [0.2, 0.8]

        return active, frequency, phase, amplitude

    def get_scale(self, r, c, t, rows=ROWS, cols=COLS):
        """
        Returns spring scale for muscle voxel at (r, c) at time t.
        Returns 1.0 (rest length) if controller silences this voxel.
        """
        x = (c / (cols - 1)) * 2 - 1 if cols > 1 else 0.0
        y = (r / (rows - 1)) * 2 - 1 if rows > 1 else 0.0

        active, frequency, phase, amplitude = self.forward(x, y)

        if not active:
            return 1.0

        return 1.0 + amplitude * math.sin(frequency * t + phase)

    def set_weights(self, weights):
        self.weights = list(weights)
        self._unpack()