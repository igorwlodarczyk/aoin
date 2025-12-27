import sys
from northpole_packing.genetic_algorithm import GeneticAlgorithm


ga = GeneticAlgorithm(
    "output.log",
    num_trees=100,
    num_generations=sys.maxsize
)
ga.solve()