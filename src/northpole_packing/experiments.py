import sys
import uuid
from northpole_packing.genetic_algorithm import GeneticAlgorithm
from northpole_packing.simulated_annealing import SimulatedAnnealing


def solve_ga():
    ga = GeneticAlgorithm(
        f"output_ga_{uuid.uuid4().hex}.log", num_trees=100, num_generations=sys.maxsize
    )
    ga.solve()


def solve_sa():
    sa = SimulatedAnnealing(
        f"output_saa_{uuid.uuid4().hex}.log",
        num_trees=100,
        alpha=0.999,
        start_temp=1000,
        min_t=-float('inf')
    )
    sa.solve()