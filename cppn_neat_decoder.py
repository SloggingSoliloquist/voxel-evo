# cppn_neat_decoder.py
#
# Decodes a NEAT genome into morphology + controller outputs.
# The single CPPN takes (x, y, d, theta, bias) and produces 8 outputs:
#
#   [0] presence_logit      — raw, threshold at 0 → EMPTY or not
#   [1] muscle_a_logit      |
#   [2] muscle_b_logit      | argmax of [1..4] → voxel type
#   [3] soft_logit          |
#   [4] rigid_logit         |
#   [5] frequency_raw       → 0.5 + sigmoid * 5.5 → [0.5, 6.0] Hz
#   [6] phase_raw           → tanh * π            → [-π, π]
#   [7] muscle_prob_raw     → sigmoid              → [0, 1], threshold 0.5

import math
import neat
from config import ROWS, COLS
from voxel import EMPTY, MUSCLE_A, MUSCLE_B, SOFT, RIGID

VOXEL_TYPES_ACTIVE = [MUSCLE_A, MUSCLE_B, SOFT, RIGID]
PRESENCE_THRESHOLD    = 0.0   # presence_logit threshold
MUSCLE_PROB_THRESHOLD = 0.5   # below this the voxel is silent

def build_network(genome, config):
    """Create a feedforward NEAT network from a genome."""
    return neat.nn.FeedForwardNetwork.create(genome, config)

def _sigmoid(x):
    x = max(-20, min(20, x))
    return 1.0 / (1.0 + math.exp(-x))


def build_network(genome, config):
    """Create a feedforward NEAT network from a genome."""
    return neat.nn.FeedForwardNetwork.create(genome, config)


def query(net, x, y):
    """
    Query the CPPN network at normalized position (x, y).
    Returns raw output list of length 8.
    """
    d     = math.sqrt(x * x + y * y) / math.sqrt(2)
    theta = math.atan2(y, x)
    return net.activate([x, y, d, theta, 1.0])


def decode_morphology(net, rows=ROWS, cols=COLS):
    """
    Query the CPPN at every grid position.
    Returns 2D morphology list.
    """
    morphology = []
    for r in range(rows):
        row = []
        for c in range(cols):
            x = (c / (cols - 1)) * 2 - 1 if cols > 1 else 0.0
            y = (r / (rows - 1)) * 2 - 1 if rows > 1 else 0.0

            out = query(net, x, y)

            presence_logit = out[0]
            type_logits    = out[1:5]

            if presence_logit < PRESENCE_THRESHOLD:
                row.append(EMPTY)
            else:
                best = type_logits.index(max(type_logits))
                row.append(VOXEL_TYPES_ACTIVE[best])

        morphology.append(row)
    return morphology


def get_scale(net, r, c, t, rows=ROWS, cols=COLS):
    """
    Query the CPPN for controller output at voxel (r, c) at time t.
    Returns spring scale value (1.0 = rest, silent voxel).
    """
    x = (c / (cols - 1)) * 2 - 1 if cols > 1 else 0.0
    y = (r / (rows - 1)) * 2 - 1 if rows > 1 else 0.0

    out = query(net, x, y)

    muscle_prob = _sigmoid(out[7])
    if muscle_prob < MUSCLE_PROB_THRESHOLD:
        return 1.0

    frequency = 0.5 + _sigmoid(out[5]) * 5.5
    phase     = math.tanh(out[6]) * math.pi
    amplitude = 0.2 + _sigmoid(out[7]) * 0.6

    return 1.0 + amplitude * math.sin(frequency * t + phase)