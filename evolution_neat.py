# evolution_neat.py

import os
import json
import pickle
import pygame
import neat
from config import ROWS, COLS
from cppn_neat_decoder import build_network, decode_morphology
from evaluator_neat import evaluate_neat

# --- Parameters ---
GENERATIONS  = 200
CONFIG_PATH  = "neat_config.cfg"
HEADLESS     = True   # set True for fast runs, False to watch


def make_fitness_fn(rows, cols, screen, font, generation_counter):
    """
    Returns a fitness function closure compatible with NEAT-Python's
    population.run() interface.
    """
    def eval_genomes(genomes, config):
        generation = generation_counter[0]
        total = len(genomes)

        for i, (genome_id, genome) in enumerate(genomes):
            net = build_network(genome, config)
            fitness = evaluate_neat(
                net, rows, cols, screen, font,
                generation=generation,
                individual=i + 1,
                population_size=total,
                headless=HEADLESS,
            )
            genome.fitness = fitness
            print(f"  Gen {generation} | {i+1}/{total} | "
                  f"id={genome_id} | fitness={fitness:.1f}px")

        generation_counter[0] += 1

    return eval_genomes


def log_generation(population, log_dir, generation):
    """Log best genome of this generation."""
    best = max(population.population.values(), key=lambda g: g.fitness or 0)
    config = population.config
    net = build_network(best, config)
    morphology = decode_morphology(net, ROWS, COLS)

    data = {
        "generation":    generation,
        "best_fitness":  best.fitness,
        "best_genome_id": best.key,
        "num_nodes":     len(best.nodes),
        "num_connections": len(best.connections),
        "num_species":   len(population.species.species),
        "best_morphology": morphology,
    }
    path = os.path.join(log_dir, f"gen_{generation:04d}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    # Also save the best genome as a pickle for replay
    pkl_path = os.path.join(log_dir, f"best_genome_{generation:04d}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(best, f)


    species_dir = os.path.join(log_dir, f"gen_{generation:04d}_species")
    os.makedirs(species_dir, exist_ok=True)

    for sid, species in population.species.species.items():
        best_in_species = max(species.members.values(), 
                            key=lambda g: g.fitness or 0)
        spath = os.path.join(species_dir, f"species_{sid}.pkl")
        with open(spath, "wb") as f:
            pickle.dump(best_in_species, f)

    print(f"[Gen {generation:>3}] "
          f"best={best.fitness:.1f}px | "
          f"species={len(population.species.species)} | "
          f"nodes={len(best.nodes)} | "
          f"conns={len(best.connections)}")


class GenerationReporter(neat.reporting.BaseReporter):
    """Custom reporter that logs each generation to disk."""

    def __init__(self, population, log_dir):
        self.population = population
        self.log_dir    = log_dir
        self.generation  = 0

    def start_generation(self, generation):
        self.generation = generation
    
    def end_generation(self, config, population, species_set):
        log_generation(self.population, self.log_dir, self.generation)

        target_species = 10
        current_species = len(species_set.species)

        if current_species < target_species - 2:
            config.species_set_config.compatibility_threshold *= 0.95
        elif current_species > target_species + 2:
            config.species_set_config.compatibility_threshold *= 1.05

        config.species_set_config.compatibility_threshold = max(1.0, min(10.0, config.species_set_config.compatibility_threshold))

        print(f"  Species: {current_species} | Threshold: {config.species_set_config.compatibility_threshold:.2f}")

def evolve():
    user_input = input("Log directory [neat_logs]: ").strip()
    log_dir = user_input if user_input else "neat_logs"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}/")
    print(f"Grid: {ROWS}x{COLS}  |  Headless: {HEADLESS}")

    if not HEADLESS:
        pygame.init()
        screen = pygame.display.set_mode((900, 600))
        pygame.display.set_caption("CPPN-NEAT Co-Evolution")
        font = pygame.font.SysFont("monospace", 18)
    else:
        screen = None
        font   = None

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    population = neat.Population(config)

    # --- Reporters ---
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(GenerationReporter(population, log_dir))

    # --- Save full stats at end ---
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    generation_counter = [1]
    fitness_fn = make_fitness_fn(ROWS, COLS, screen, font, generation_counter)

    winner = population.run(fitness_fn, GENERATIONS)

    # Save winner
    winner_path = os.path.join(log_dir, "winner.pkl")
    import pickle
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"\nWinner saved to {winner_path}")
    print(f"Winner fitness: {winner.fitness:.1f}px")
    print(f"Winner nodes: {len(winner.nodes)}, connections: {len(winner.connections)}")

    if not HEADLESS:
        pygame.quit()


if __name__ == "__main__":
    evolve()