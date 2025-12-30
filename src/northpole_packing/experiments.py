import sys
import uuid
from northpole_packing.genetic_algorithm import GeneticAlgorithm


ga = GeneticAlgorithm(
    f"output_{uuid.uuid4().hex}.log", num_trees=100, num_generations=sys.maxsize
)
ga.solve()
