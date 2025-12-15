from northpole_packing.solver import SimulatedAnnealing


def simulated_annealing():
    sa = SimulatedAnnealing(100)
    best_solution, best_solution_cost = sa.solve("output.csv")
    print(best_solution_cost)


simulated_annealing()
