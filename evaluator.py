# evaluator.py

from ppo import train_ppo


def evaluate(morphology, rows, cols,
             generation=0, individual=0, population_size=0):
    """
    Train a PPO controller for the given morphology.
    Returns fitness = best mean reward over training.
    Fully headless — no pygame, no rendering.
    """
    print(f"    [PPO] gen {generation} individual {individual}/{population_size} "
          f"morphology {rows}x{cols} ...")

    best_reward, final_reward = train_ppo(morphology)

    print(f"    [PPO] best={best_reward:.4f}  final={final_reward:.4f}")
    return best_reward