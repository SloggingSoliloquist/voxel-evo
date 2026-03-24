# evolution.py

import random
import json
import os
import math
import torch
from config import ROWS, COLS
from voxel import EMPTY, MUSCLE_A, MUSCLE_B, SOFT, RIGID
from cppn_morphology import CPPN
from ppo import train_ppo, continue_ppo, DEVICE

# --- GA parameters ---
POPULATION_SIZE   = 20
GENERATIONS       = 50
MUTATION_RATE     = 0.1
MUTATION_STRENGTH = 0.8
MAX_RETRIES       = 100

SURVIVAL_START    = 0.5
SURVIVAL_END      = 0.2

_morph_ref    = CPPN()
GENOME_LENGTH = _morph_ref.genome_length
print(f"Morphology CPPN genome length: {GENOME_LENGTH}")
print(f"Running on device: {DEVICE}")


# ------------------------------------------------------------------
# Survival schedule
# ------------------------------------------------------------------

def get_survival_rate(generation, total_generations):
    t = generation / max(total_generations - 1, 1)
    return SURVIVAL_START + t * (SURVIVAL_END - SURVIVAL_START)


# ------------------------------------------------------------------
# Morphology validity
# ------------------------------------------------------------------

def is_connected(morphology):
    rows = len(morphology)
    cols = len(morphology[0])
    active = [(r, c) for r in range(rows) for c in range(cols)
              if morphology[r][c] != EMPTY]
    if not active:
        return False
    visited = set()
    stack = [active[0]]
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if morphology[nr][nc] != EMPTY and (nr,nc) not in visited:
                    stack.append((nr, nc))
    return len(visited) == len(active)


def has_actuator(morphology):
    return any(cell in (MUSCLE_A, MUSCLE_B)
               for row in morphology for cell in row)


def morphology_hash(morphology):
    return tuple(cell for row in morphology for cell in row)


def is_valid(genome):
    m = CPPN(genome).decode()
    return is_connected(m) and has_actuator(m)


# ------------------------------------------------------------------
# Genome utilities
# ------------------------------------------------------------------

def random_genome():
    while True:
        genome = [random.gauss(0, 1.0) for _ in range(GENOME_LENGTH)]
        if is_valid(genome):
            return genome


def mutate(genome):
    for _ in range(MAX_RETRIES):
        candidate = [
            w + random.gauss(0, MUTATION_STRENGTH) if random.random() < MUTATION_RATE else w
            for w in genome
        ]
        if is_valid(candidate):
            return candidate
    return None


# ------------------------------------------------------------------
# Structure
# ------------------------------------------------------------------

class Structure:
    def __init__(self, genome, label):
        self.genome      = genome
        self.label       = label
        self.morphology  = CPPN(genome).decode()
        self.fitness     = None
        self.policy      = None
        self.stats       = None
        self.is_survivor = False

    def __repr__(self):
        dist = self.stats.get("final_dist_px", 0) if self.stats else 0
        return f"Structure(label={self.label}, fitness={self.fitness:.4f}, dist={dist:.1f}px)"


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def log_generation(generation, structures, log_dir):
    sorted_s = sorted(structures, key=lambda s: s.fitness, reverse=True)
    best     = sorted_s[0]

    # Save best policy weights
    policy_path = os.path.join(log_dir, f"gen_{generation:04d}_policy.pt")
    torch.save(best.policy.state_dict(), policy_path)

    data = {
        "generation":      generation,
        "best_fitness":    best.fitness,
        "mean_fitness":    sum(s.fitness for s in structures) / len(structures),
        "best_genome":     best.genome,
        "best_morphology": best.morphology,
        "policy_path":     policy_path,
        "all_fitnesses":   [s.fitness for s in structures],

        # Full stats for every individual
        "all_stats": [
            {
                "label":         s.label,
                "fitness":       s.fitness,
                "final_dist_px": s.stats.get("final_dist_px", 0) if s.stats else 0,
                "max_dist_px":   s.stats.get("max_dist_px", 0)   if s.stats else 0,
                "final_reward":  s.stats.get("final_reward", 0)  if s.stats else 0,
            }
            for s in structures
        ],
    }

    path = os.path.join(log_dir, f"gen_{generation:04d}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[Gen {generation:>3}] best={data['best_fitness']:.4f}  "
          f"mean={data['mean_fitness']:.4f}  "
          f"best_dist={best.stats.get('max_dist_px', 0):.1f}px  "
          f"logged → {path}")


# ------------------------------------------------------------------
# Main evolution loop
# ------------------------------------------------------------------

def evolve():
    user_input = input("Log directory [evo_logs]: ").strip()
    log_dir = user_input if user_input else "evo_logs"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}/\n")

    evaluated_hashes = {}

    structures = []
    label      = 0
    while len(structures) < POPULATION_SIZE:
        genome = random_genome()
        morph  = CPPN(genome).decode()
        h      = morphology_hash(morph)
        if h not in evaluated_hashes:
            evaluated_hashes[h] = True
            structures.append(Structure(genome, label))
            label += 1

    for generation in range(1, GENERATIONS + 1):
        print(f"\n=== Generation {generation}/{GENERATIONS} ===")

        survival_rate = get_survival_rate(generation - 1, GENERATIONS)
        num_survivors = max(2, math.ceil(POPULATION_SIZE * survival_rate))
        print(f"Survival rate: {survival_rate:.0%}  →  {num_survivors} survivors\n")

        # --- Train ---
        for i, s in enumerate(structures):
            if s.is_survivor:
                print(f"  [{i+1}/{len(structures)}] label={s.label} SURVIVOR — continuing PPO")
                stats, policy = continue_ppo(s.morphology, s.policy)
            else:
                print(f"  [{i+1}/{len(structures)}] label={s.label} NEW — training PPO from scratch")
                stats, policy = train_ppo(s.morphology)

            s.fitness = stats["best_reward"]
            s.policy  = policy
            s.stats   = stats
            print(f"    → best_reward={stats['best_reward']:.4f}  "
                  f"final_reward={stats['final_reward']:.4f}  "
                  f"final_dist={stats['final_dist_px']:.1f}px  "
                  f"max_dist={stats['max_dist_px']:.1f}px")

        # --- Sort and log ---
        structures.sort(key=lambda s: s.fitness, reverse=True)
        log_generation(generation, structures, log_dir)

        print(f"\nTop {num_survivors} designs:")
        for s in structures[:num_survivors]:
            print(f"  {s}")

        # --- Truncation selection ---
        survivors = structures[:num_survivors]
        for s in survivors:
            s.is_survivor = True

        # --- Mutation ---
        next_gen = list(survivors)
        attempts = 0
        while len(next_gen) < POPULATION_SIZE and attempts < MAX_RETRIES * POPULATION_SIZE:
            parent       = random.choice(survivors)
            child_genome = mutate(parent.genome)
            attempts    += 1

            if child_genome is None:
                continue

            child_morph = CPPN(child_genome).decode()
            h = morphology_hash(child_morph)

            if h not in evaluated_hashes:
                evaluated_hashes[h] = True
                next_gen.append(Structure(child_genome, label))
                label += 1

        if len(next_gen) < POPULATION_SIZE:
            print(f"Warning: only {len(next_gen)}/{POPULATION_SIZE} unique children generated")

        for s in next_gen:
            if s not in survivors:
                s.is_survivor = False

        structures = next_gen

    print("\nEvolution complete.")


if __name__ == "__main__":
    evolve()