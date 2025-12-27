import random
import statistics
import numpy as np
from northpole_packing.initialization import greedy_initialization
from northpole_packing.tree import calculate_side_length, convert_trees_to_string


class Individual:
    def __init__(self, tree_angles):
        self.__tree_angles = tree_angles
        self.__trees = None
        self.__score = None
        self.__num_trees = None

    @property
    def tree_angles(self):
        return self.__tree_angles

    @property
    def num_trees(self):
        if self.__num_trees is None:
            self.__num_trees = len(self.__tree_angles)

        return self.__num_trees

    @property
    def trees(self):
        if self.__trees is None:
            self.__trees = greedy_initialization(
                num_trees=self.num_trees, tree_angles=self.tree_angles
            )
        return self.__trees

    @property
    def score(self):
        if self.__score is None:
            self.__score = calculate_side_length(self.trees)
        return self.__score


class GeneticAlgorithm:
    def __init__(
        self,
        output_log_path,
        num_trees=100,
        num_generations=100,
        pop_size=100,
        population_initialisation="random",
        elite=0.1,
        px=0.75,
        pm=0.05,
        selection_type="tournament",
        tournament_size=5,
    ):
        self.output_log_path = output_log_path
        self.num_trees = num_trees
        self.num_generations = num_generations
        self.pop_size = pop_size
        self.population_initialisation = population_initialisation
        self.elite = elite
        self.px = px
        self.pm = pm
        self.selection_type = selection_type
        self.tournament_size = tournament_size

    @staticmethod
    def one_point_crossover(individual_1, individual_2):
        tree_angles_1 = individual_1.tree_angles
        tree_angles_2 = individual_2.tree_angles

        crossover_point = random.randint(1, len(tree_angles_1) - 1)
        child_tree_angles = (
            tree_angles_1[:crossover_point] + tree_angles_2[crossover_point:]
        )

        child = Individual(tree_angles=child_tree_angles)
        return child

    def initialize_population_random(self):
        population = [
            Individual(
                tree_angles=[random.uniform(0, 360) for _ in range(self.num_trees)]
            )
            for _ in range(self.pop_size)
        ]
        return population

    @staticmethod
    def selection(
        population: list,
        selection_type: str = "tournament",
        tournament_size: int = 5,
    ) -> Individual:
        if selection_type == "tournament":
            participants = random.sample(population, tournament_size)
            best_individual = min(participants, key=lambda s: s.score)
            return best_individual
        else:
            raise Exception

    @staticmethod
    def mutation_gaussian_step(individual, pm):
        tree_angles = individual.tree_angles
        new_tree_angles = []
        for tree_angle in tree_angles:
            if random.uniform(0, 1) < pm:
                new_tree_angle = (tree_angle + np.random.normal(0, 30)) % 360
                new_tree_angles.append(new_tree_angle)
            else:
                new_tree_angles.append(tree_angle)
        return Individual(tree_angles=new_tree_angles)

    def solve(self):
        with open(self.output_log_path, "w") as log_file:
            population = self.initialize_population_random()
            for _ in range(self.num_generations):
                new_population = []
                elite_count = int(self.elite * self.pop_size)
                if elite_count > 0:
                    sorted_pop = sorted(population, key=lambda s: s.score)
                    elites = sorted_pop[:elite_count]
                    new_population.extend(elites)

                while len(new_population) < self.pop_size:
                    parent1 = self.selection(
                        population, self.selection_type, self.tournament_size
                    )
                    if random.uniform(0, 1) < self.px:
                        parent2 = self.selection(
                            population, self.selection_type, self.tournament_size
                        )
                        child = self.one_point_crossover(parent1, parent2)
                    else:
                        child = parent1

                    child = self.mutation_gaussian_step(child, self.pm)
                    new_population.append(child)

                population = new_population
                population_scores = [individual.score for individual in population]
                best_score = min(population_scores)
                worst_score = max(population_scores)
                avg_score = statistics.mean(population_scores)

                best_individual = min(population, key=lambda s: s.score)
                best_individual_trees = best_individual.trees
                best_individual_trees_string = convert_trees_to_string(
                    best_individual_trees
                )

                log_file.write(
                    f"{_};{best_score};{avg_score};{worst_score};{best_individual_trees_string}\n"
                )
