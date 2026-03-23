# cppn.py

import math
import random
from config import ROWS, COLS
from voxel import EMPTY, MUSCLE_A, MUSCLE_B, SOFT, RIGID

# --- Network architecture ---
INPUT_SIZE  = 5   # (x, y, d, theta, bias)
HIDDEN_SIZE = 16
OUTPUT_SIZE = 5   # (presence, muscle_a, muscle_b, soft, rigid)

VOXEL_TYPES_ACTIVE = [MUSCLE_A, MUSCLE_B, SOFT, RIGID]

# --- Activation functions ---
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


class CPPN:
    """
    Fixed-topology CPPN: 5 -> HIDDEN_SIZE -> HIDDEN_SIZE -> 5

    Inputs: (x, y, d, theta, bias)
        x, y   : normalized to [-1, 1]
        d      : distance from center, normalized to [0, 1]
        theta  : atan2(y, x), angle in [-π, π]
        bias   : constant 1.0

    Outputs: (presence, muscle_a, muscle_b, soft, rigid) logits
        presence < 0.0  → EMPTY
        otherwise       → argmax of type logits

    Genome layout:
        W1:    INPUT_SIZE  x HIDDEN_SIZE floats
        b1:    HIDDEN_SIZE floats
        acts1: HIDDEN_SIZE floats (activation indices, floored when used)
        W2:    HIDDEN_SIZE x HIDDEN_SIZE floats
        b2:    HIDDEN_SIZE floats
        acts2: HIDDEN_SIZE floats
        W3:    HIDDEN_SIZE x OUTPUT_SIZE floats
        b3:    OUTPUT_SIZE floats
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
        d     = math.sqrt(x * x + y * y) / math.sqrt(2)
        theta = math.atan2(y, x)
        inp   = [x, y, d, theta, 1.0]

        h1_pre = self._matmul_add(self.W1, self.b1, inp, INPUT_SIZE, HIDDEN_SIZE)
        h1 = [self.acts1[i](v) for i, v in enumerate(h1_pre)]

        h2_pre = self._matmul_add(self.W2, self.b2, h1, HIDDEN_SIZE, HIDDEN_SIZE)
        h2 = [self.acts2[i](v) for i, v in enumerate(h2_pre)]

        out = self._matmul_add(self.W3, self.b3, h2, HIDDEN_SIZE, OUTPUT_SIZE)
        return out

    def decode(self, rows=ROWS, cols=COLS, presence_threshold=0.0):
        morphology = []
        for r in range(rows):
            row = []
            for c in range(cols):
                x = (c / (cols - 1)) * 2 - 1 if cols > 1 else 0.0
                y = (r / (rows - 1)) * 2 - 1 if rows > 1 else 0.0

                out = self.forward(x, y)

                presence_logit = out[0]
                type_logits    = out[1:]

                if presence_logit < presence_threshold:
                    row.append(EMPTY)
                else:
                    best = type_logits.index(max(type_logits))
                    row.append(VOXEL_TYPES_ACTIVE[best])
            morphology.append(row)
        return morphology

    def set_weights(self, weights):
        self.weights = list(weights)
        self._unpack()