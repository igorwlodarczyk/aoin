import math
import numpy as np
import copy
import time
import statistics
import random
from decimal import Decimal
from sklearn.cluster import KMeans

from northpole_packing.initialization import greedy_initialization
from northpole_packing.tree import (
    ChristmasTree,
    has_collision_with_candidate,
    calculate_side_length,
convert_trees_to_string,
)


class Individual:
    def __init__(self, trees):
        self.__trees = trees
        self.__num_trees = len(self.__trees)
        self.__score = None

    @property
    def trees(self):
        return self.__trees

    @property
    def num_trees(self):
        return self.__num_trees

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
        population_initialisation="greedy",
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

    def initialize_population_greedy(self):
        population = [
            Individual(trees=greedy_initialization(num_trees=self.num_trees))
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
    def mutation(individual, pm=0.1, max_attempts=1000):
        sigma = 0.1
        trees = individual.trees
        new_trees = []
        for tree in trees:
            tree_placed = False
            for attempt in range(max_attempts):
                x, y, angle = tree.get_params()

                if random.uniform(0, 1) < pm:
                    x += np.random.normal(0, sigma)
                if random.uniform(0, 1) < pm:
                    y += np.random.normal(0, sigma)
                if random.uniform(0, 1) < pm:
                    angle += (angle + np.random.normal(0, 30)) % 360

                new_tree = ChristmasTree(str(x), str(y), str(angle))
                if not has_collision_with_candidate(new_trees, new_tree):
                    new_trees.append(new_tree)
                    tree_placed = True
            if not tree_placed:
                new_trees.append(copy.deepcopy(tree))
        return Individual(new_trees)

    def mutation_boosted(self, individual, pm=0.1, multiplier=5):
        potential_mutations = [self.mutation(individual, pm) for _ in range(multiplier)]
        return min(potential_mutations, key=lambda s: s.score)

    @staticmethod
    def crossover(parent1, parent2, cluster_n_divisor: int = 10):
        def match_centroid(parent1, parent2):
            n1 = Decimal(len(parent1))
            n2 = Decimal(len(parent2))

            p1x = sum(t.center_x for t in parent1) / n1
            p1y = sum(t.center_y for t in parent1) / n1

            p2x = sum(t.center_x for t in parent2) / n2
            p2y = sum(t.center_y for t in parent2) / n2

            dx = p1x - p2x
            dy = p1y - p2y

            adjusted = []
            for t in parent2:
                adjusted.append(
                    ChristmasTree(
                        str(t.center_x + dx),
                        str(t.center_y + dy),
                        str(t.angle),
                    )
                )

            return parent1, adjusted

        def get_trees_from_cluster(trees, labels, cluster_id):
            trees_cluster = [
                tree for tree, label in zip(trees, labels) if label == cluster_id
            ]
            return trees_cluster

        parent1 = parent1.trees
        parent2 = parent2.trees
        parent1, parent2 = match_centroid(parent1, parent2)
        parent1 = copy.deepcopy(parent1)

        num_trees = len(parent1)
        n_clusters = max(2, math.ceil(num_trees / cluster_n_divisor))

        parent1_points = np.array(
            [[float(t.center_x), float(t.center_y)] for t in parent1]
        )
        parent2_points = np.array(
            [[float(t.center_x), float(t.center_y)] for t in parent2]
        )
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=42, n_init="auto", max_iter=30
        )

        labels_parent1 = kmeans.fit_predict(parent1_points)
        labels_parent2 = kmeans.fit_predict(parent2_points)

        parent1_clusters_sorted = np.unique(labels_parent1)[
            np.argsort(-np.unique(labels_parent1, return_counts=True)[1])
        ].tolist()
        parent2_clusters_sorted = np.unique(labels_parent2)[
            np.argsort(-np.unique(labels_parent2, return_counts=True)[1])
        ].tolist()

        child = []
        incompatible_trees = []

        for p1_cluster, p2_cluster in zip(
            parent1_clusters_sorted, parent2_clusters_sorted
        ):
            p1_cluster_trees = get_trees_from_cluster(
                parent1, labels_parent1, p1_cluster
            )
            p2_cluster_trees = get_trees_from_cluster(
                parent2, labels_parent2, p2_cluster
            )
            tree_candidates = p1_cluster_trees + p2_cluster_trees

            for tree_candidate in tree_candidates:
                if len(child) == num_trees:
                    break

                if not has_collision_with_candidate(child, tree_candidate):
                    child.append(tree_candidate)
                else:
                    incompatible_trees.append(tree_candidate)

            if len(child) == num_trees:
                break

        # greedy repair
        while len(child) < num_trees:
            tree_candidate = incompatible_trees.pop(0)
            *_, angle = tree_candidate.get_params()
            # greedy params
            starting_dist = 0.4
            step = 0.1
            search_depth: int = 3
            tree_placed = False
            dist = starting_dist

            while not tree_placed:
                best_tree = None
                min_side = float("inf")
                tmp_dist = dist
                for _ in range(search_depth):
                    points = [
                        (tmp_dist, 0),
                        (-tmp_dist, 0),
                        (0, tmp_dist),
                        (0, -tmp_dist),
                        (tmp_dist, tmp_dist),
                        (tmp_dist, -tmp_dist),
                        (-tmp_dist, tmp_dist),
                        (-tmp_dist, -tmp_dist),
                    ]
                    for tree in child:
                        x, y, _ = tree.get_params()
                        for point_x, point_y in points:
                            x_candidate = x + point_x
                            y_candidate = y + point_y
                            tree_candidate = ChristmasTree(
                                str(x_candidate), str(y_candidate), str(angle)
                            )

                            if not has_collision_with_candidate(child, tree_candidate):
                                child.append(tree_candidate)
                                side = calculate_side_length(child)
                                child.remove(tree_candidate)

                                if side < min_side:
                                    min_side = side
                                    best_tree = tree_candidate

                    tmp_dist += step
                if best_tree is not None:
                    child.append(best_tree)
                    tree_placed = True
                else:
                    dist += step * search_depth
        return Individual(child)

    def solve(self):
        with open(self.output_log_path, "w") as log_file:
            start_time = time.time()
            population = self.initialize_population_greedy()
            init_time = time.time() - start_time
            print(f"Population initialization finished in {init_time:.2f} seconds")
            for epoch in range(self.num_generations):
                print(f"Starting epoch {epoch + 1}/{self.num_generations}")
                new_population = []
                elite_count = int(self.elite * self.pop_size)
                if elite_count > 0:
                    sorted_pop = sorted(population, key=lambda s: s.score)
                    elites = sorted_pop[:elite_count]
                    new_population.extend(elites)

                while len(new_population) < self.pop_size:
                    start_time = time.time()
                    parent1 = self.selection(
                        population, self.selection_type, self.tournament_size
                    )
                    if random.uniform(0, 1) < self.px:
                        parent2 = self.selection(
                            population, self.selection_type, self.tournament_size
                        )
                        child = self.crossover(parent1, parent2)
                    else:
                        child = parent1

                    child = self.mutation(child, self.pm)
                    new_population.append(child)
                    print(f"(Epoch {epoch}) Population size: {len(new_population)}/{self.pop_size}")
                population = new_population
                end_time = time.time()
                print(f"New population creation finished in {end_time-start_time:.2f} seconds")
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
                    f"{epoch};{best_score};{avg_score};{worst_score};{best_individual_trees_string}\n"
                )
                log_file.flush()

ga = GeneticAlgorithm("ga_long.log", num_trees=100, num_generations=10000, pop_size=50)
ga.solve()