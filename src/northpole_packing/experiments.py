import sys
import uuid
from northpole_packing.genetic_algorithm_angle_based import GeneticAlgorithm
from northpole_packing.simulated_annealing import SimulatedAnnealing
from northpole_packing.hill_climbing import HillClimbing
from northpole_packing.tabu_search import TabuSearch


def solve_ga():
    ga = GeneticAlgorithm(
        f"output_ga_{uuid.uuid4().hex}.log", num_trees=100, num_generations=10
    )
    ga.solve()


def solve_sa():
    sa = SimulatedAnnealing(
        f"data/high_temp_output_saa_{uuid.uuid4().hex}.log",
        num_trees=100,
        alpha=0.99986,
        start_temp=5000,
        min_t=3.8e-6,
    )
    sa.solve()

def solve_hc():
    hc = HillClimbing(
        f"output_hc_{uuid.uuid4().hex}.log",
        num_trees=100
    )
    hc.solve()


def solve_ts():
    ts = TabuSearch(
        f"output_ts_{uuid.uuid4().hex}.log",
        num_trees=100,
        iterations=150_000
    )
    ts.solve()

