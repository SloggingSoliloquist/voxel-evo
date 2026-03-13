# evolution.py

import pygame
import random
import json
import os
from config import MORPHOLOGY
from voxel import EMPTY, MUSCLE_A, MUSCLE_B, SOFT, RIGID
from evaluator import evaluate

# --- GA parameters ---
POPULATION_SIZE = 20
GENERATIONS     = 50
TOURNAMENT_K    = 3       # competitors per tournament
MUTATION_RATE   = 0.1     # probability of each cell mutating
ELITISM         = 1       # number of top genomes carried over unchanged

ROWS = len(MORPHOLOGY)
COLS = len(MORPHOLOGY[0])
GENOME_LENGTH = ROWS * COLS

VOXEL_TYPES = [EMPTY, MUSCLE_A, MUSCLE_B, SOFT, RIGID]

LOG_DIR = "evo_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Genome utilities
# ------------------------------------------------------------------

def random_genome():
    return [random.choice(VOXEL_TYPES) for _ in range(GENOME_LENGTH)]


def mutate(genome):
    return [
        random.choice(VOXEL_TYPES) if random.random() < MUTATION_RATE else gene
        for gene in genome
    ]


def crossover(a, b):
    """Single-point crossover."""
    point = random.randint(1, GENOME_LENGTH - 1)
    return a[:point] + b[point:]


def tournament_select(population, fitnesses, k=TOURNAMENT_K):
    """Pick k random individuals, return the genome of the best."""
    competitors = random.sample(range(len(population)), k)
    best = max(competitors, key=lambda i: fitnesses[i])
    return population[best]


def genome_to_morphology(genome):
    return [genome[r * COLS:(r + 1) * COLS] for r in range(ROWS)]


def log_generation(generation, population, fitnesses):
    data = {
        "generation": generation,
        "best_fitness": max(fitnesses),
        "mean_fitness": sum(fitnesses) / len(fitnesses),
        "best_genome": population[fitnesses.index(max(fitnesses))],
        "all_fitnesses": fitnesses,
    }
    path = os.path.join(LOG_DIR, f"gen_{generation:04d}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Gen {generation:>3}] best={data['best_fitness']:.1f}px  "
          f"mean={data['mean_fitness']:.1f}px  "
          f"logged → {path}")


# ------------------------------------------------------------------
# Main evolution loop
# ------------------------------------------------------------------

def evolve():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Morphology Evolution")
    font = pygame.font.SysFont("monospace", 18)

    # Initial random population
    population = [random_genome() for _ in range(POPULATION_SIZE)]

    for generation in range(1, GENERATIONS + 1):
        fitnesses = []

        # Evaluate every individual sequentially
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

        log_generation(generation, population, fitnesses)

        # --- Selection + reproduction ---
        # Sort by fitness descending
        ranked = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
        fitnesses_sorted = [f for f, _ in ranked]
        population_sorted = [g for _, g in ranked]

        new_population = []

        # Elitism: carry top genomes unchanged
        for i in range(ELITISM):
            new_population.append(population_sorted[i])

        # Fill rest with tournament selection + crossover + mutation
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