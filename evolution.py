# evolution.py

import pygame
import random
import json
import os
from config import ROWS, COLS, VOXEL_PD
from voxel import EMPTY, MUSCLE_A, MUSCLE_B, SOFT, RIGID
from evaluator import evaluate

# --- GA parameters ---
POPULATION_SIZE = 20
GENERATIONS     = 50
TOURNAMENT_K    = 3
MUTATION_RATE   = 0.1
ELITISM         = 1

GENOME_LENGTH = ROWS * COLS

VOXEL_TYPES = [EMPTY, MUSCLE_A, MUSCLE_B, SOFT, RIGID]


# ------------------------------------------------------------------
# Connectivity check
# ------------------------------------------------------------------

def genome_to_morphology(genome):
    return [genome[r * COLS:(r + 1) * COLS] for r in range(ROWS)]


def is_connected(genome):
    grid = genome_to_morphology(genome)

    active = [
        (r, c)
        for r in range(ROWS)
        for c in range(COLS)
        if grid[r][c] != EMPTY
    ]

    if len(active) == 0:
        return False

    visited = set()
    stack = [active[0]]
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                if grid[nr][nc] != EMPTY and (nr, nc) not in visited:
                    stack.append((nr, nc))

    return len(visited) == len(active)


# ------------------------------------------------------------------
# Genome utilities
# ------------------------------------------------------------------

def sample_voxel():
    return random.choices(VOXEL_TYPES, weights=VOXEL_PD, k=1)[0]


def random_genome():
    while True:
        genome = [sample_voxel() for _ in range(GENOME_LENGTH)]
        if is_connected(genome):
            return genome


def mutate(genome):
    while True:
        candidate = [
            sample_voxel() if random.random() < MUTATION_RATE else gene
            for gene in genome
        ]
        if is_connected(candidate):
            return candidate


def crossover(a, b):
    while True:
        point = random.randint(1, GENOME_LENGTH - 1)
        child = a[:point] + b[point:]
        if is_connected(child):
            return child


def tournament_select(population, fitnesses, k=TOURNAMENT_K):
    competitors = random.sample(range(len(population)), k)
    best = max(competitors, key=lambda i: fitnesses[i])
    return population[best]


def log_generation(generation, population, fitnesses, log_dir):
    data = {
        "generation": generation,
        "best_fitness": max(fitnesses),
        "mean_fitness": sum(fitnesses) / len(fitnesses),
        "best_genome": population[fitnesses.index(max(fitnesses))],
        "all_fitnesses": fitnesses,
    }
    path = os.path.join(log_dir, f"gen_{generation:04d}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Gen {generation:>3}] best={data['best_fitness']:.1f}px  "
          f"mean={data['mean_fitness']:.1f}px  "
          f"logged → {path}")


# ------------------------------------------------------------------
# Main evolution loop
# ------------------------------------------------------------------

def evolve():
    # --- Prompt for log directory ---
    user_input = input("Log directory [evo_logs]: ").strip()
    log_dir = user_input if user_input else "evo_logs"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}/")

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Morphology Evolution")
    font = pygame.font.SysFont("monospace", 18)

    population = [random_genome() for _ in range(POPULATION_SIZE)]

    for generation in range(1, GENERATIONS + 1):
        fitnesses = []

        for i, genome in enumerate(population):
            print(f"  Evaluating gen {generation}, individual {i+1}/{POPULATION_SIZE}...")
            fitness = evaluate(
                genome, ROWS, COLS, screen, font,
                generation=generation,
                individual=i + 1,
                population_size=POPULATION_SIZE,
            )
            fitnesses.append(fitness)
            print(f"    → fitness: {fitness:.1f} px")

        log_generation(generation, population, fitnesses, log_dir)

        ranked = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
        population_sorted = [g for _, g in ranked]

        new_population = []

        for i in range(ELITISM):
            new_population.append(population_sorted[i])

        while len(new_population) < POPULATION_SIZE:
            parent_a = tournament_select(population, fitnesses)
            parent_b = tournament_select(population, fitnesses)
            child = crossover(parent_a, parent_b)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    print("\nEvolution complete.")
    pygame.quit()


if __name__ == "__main__":
    evolve()