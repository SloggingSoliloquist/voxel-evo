# cppn.py

import math
import random
from config import ROWS, COLS
from voxel import EMPTY, MUSCLE_A, MUSCLE_B, SOFT, RIGID

# --- Network architecture ---
INPUT_SIZE  = 3   # (x, y, d) — normalized position + distance from center
HIDDEN_SIZE = 16
OUTPUT_SIZE = 5   # (presence, muscle_a, muscle_b, soft, rigid) logits

VOXEL_TYPES_ACTIVE = [MUSCLE_A, MUSCLE_B, SOFT, RIGID]

# --- Activation functions ---
def sigmoid(x):
    x = max(-20, min(20, x))
    return 1.0 / (1.0 + math.exp(-x))

def tanh_act(x):
    return math.tanh(x)

def sin_act(x):
    return math.sin(x)

def gauss_act(x):
    return math.exp(-x * x)

def abs_act(x):
    return abs(x)

def linear(x):
    return x

ACTIVATION_FUNCTIONS = [tanh_act, sin_act, gauss_act, abs_act, linear]
N_ACTIVATIONS = len(ACTIVATION_FUNCTIONS)


class CPPN:
    """
    Fixed-topology CPPN: 3 -> HIDDEN_SIZE -> HIDDEN_SIZE -> 5

    Genome is a flat list:
        W1:     INPUT_SIZE  x HIDDEN_SIZE floats
        b1:     HIDDEN_SIZE floats
        acts1:  HIDDEN_SIZE ints (activation index per node, layer 1)
        W2:     HIDDEN_SIZE x HIDDEN_SIZE floats
        b2:     HIDDEN_SIZE floats
        acts2:  HIDDEN_SIZE ints (activation index per node, layer 2)
        W3:     HIDDEN_SIZE x OUTPUT_SIZE floats
        b3:     OUTPUT_SIZE floats

    acts are stored as floats in [0, N_ACTIVATIONS) and floored when used,
    so they participate in crossover naturally without special handling.
    """

    def __init__(self, weights=None):
        self.genome_length = (
            INPUT_SIZE  * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE +  # W1, b1, acts1
            HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE +  # W2, b2, acts2
            HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE                   # W3, b3
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
        idx = 0
        total = self.genome_length

        # W1
        for _ in range(INPUT_SIZE * HIDDEN_SIZE):
            w.append(random.gauss(0, 1.0))
        # b1
        for _ in range(HIDDEN_SIZE):
            w.append(random.gauss(0, 1.0))
        # acts1 — random float in [0, N_ACTIVATIONS)
        for _ in range(HIDDEN_SIZE):
            w.append(random.uniform(0, N_ACTIVATIONS))
        # W2
        for _ in range(HIDDEN_SIZE * HIDDEN_SIZE):
            w.append(random.gauss(0, 1.0))
        # b2
        for _ in range(HIDDEN_SIZE):
            w.append(random.gauss(0, 1.0))
        # acts2
        for _ in range(HIDDEN_SIZE):
            w.append(random.uniform(0, N_ACTIVATIONS))
        # W3
        for _ in range(HIDDEN_SIZE * OUTPUT_SIZE):
            w.append(random.gauss(0, 1.0))
        # b3
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
        Query CPPN at normalized (x, y) in [-1, 1].
        d = distance from center, also in [0, ~1.414], normalized to [0, 1].
        """
        d = math.sqrt(x * x + y * y) / math.sqrt(2)  # normalize to [0, 1]
        inp = [x, y, d]

        # Layer 1 — per-node activation functions
        h1_pre = self._matmul_add(self.W1, self.b1, inp, INPUT_SIZE, HIDDEN_SIZE)
        h1 = [self.acts1[i](v) for i, v in enumerate(h1_pre)]

        # Layer 2 — per-node activation functions
        h2_pre = self._matmul_add(self.W2, self.b2, h1, HIDDEN_SIZE, HIDDEN_SIZE)
        h2 = [self.acts2[i](v) for i, v in enumerate(h2_pre)]

        # Output layer — linear
        out = self._matmul_add(self.W3, self.b3, h2, HIDDEN_SIZE, OUTPUT_SIZE)

        return out  # [presence_logit, muscle_a_logit, muscle_b_logit, soft_logit, rigid_logit]

    def decode(self, rows=ROWS, cols=COLS, presence_threshold=0.0):
        """
        Query CPPN at every grid position and return a 2D morphology list.
        presence_threshold: logit threshold (0.0 = sigmoid 0.5 boundary)
        """
        morphology = []
        for r in range(rows):
            row = []
            for c in range(cols):
                x = (c / (cols - 1)) * 2 - 1 if cols > 1 else 0.0
                y = (r / (rows - 1)) * 2 - 1 if rows > 1 else 0.0

                out = self.forward(x, y)

                presence_logit = out[0]
                type_logits    = out[1:]  # [muscle_a, muscle_b, soft, rigid]

                if presence_logit < presence_threshold:
                    row.append(EMPTY)
                else:
                    # argmax over type logits — no saturation bias
                    best = type_logits.index(max(type_logits))
                    row.append(VOXEL_TYPES_ACTIVE[best])

            morphology.append(row)
        return morphology

    def set_weights(self, weights):
        self.weights = list(weights)
        self._unpack()