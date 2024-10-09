import optuna
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
from evoman.environment import Environment
from demo_controller import player_controller

# make sure directory exists function
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# runs simulation for each individual and returns their fitness score
def simulation(env, x):
    ''' takes in the vector of an individuals variables, plays them in the environment
    and returns their fitness value '''
    f, p, e, t = env.play(pcont=x)
    return f

# evaluation returns an array of fitness values for each individual in a population
def evaluate(env, x):
    ''' takes in the population matrix (npop x nvariables) and returns an
    array (length npop) with the fitness of each individual in the population '''
    return np.array(list(map(lambda y: simulation(env, y), x)))

# function to select parents (tournament selection)
def tournament_selection(population, pop_fitness, tournament_size):
    ''' implements the tournament selection algorithm. takes in the population and their fitnesses.
    randomly selects tournament_size members from the population and
    returns the fittest individual.
    '''
    # select indices based on tournament size from the population
    selected_indices = np.random.choice(population.shape[0], tournament_size, replace=False)

    # find the index of the individual with the best fitness in the tournament
    best_index = selected_indices[np.argmax(pop_fitness[selected_indices])]

    return population[best_index]

# intermediate recombination function to generate children #
def intermediate_recombination(population, pop_fitness, x_percent, alpha, tournament_size=2):
    ''' implements an intermediate recombination method. Each gene of a child is a weighted average
    between the corresponding genes of two parents. the weighting is controlled by the parameter alpha. '''

    children = []

    # loop through 100 - x% the number of the population
    for i in range(int(population.shape[0] * ((100 - x_percent) / 100))):

        # select parents using tournament selection with the tournament size
        parent_1 = tournament_selection(population, pop_fitness, tournament_size)
        parent_2 = tournament_selection(population, pop_fitness, tournament_size)

        # create child as an intermediate recombination of the parents
        child = alpha * parent_1 + (1 - alpha) * parent_2

        # add new child to children
        children.append(child)

    # return newly made children in array format
    return np.array(children)

# uniform mutation function
def uniform_mutation(population, mutation_rate, domain_lower_bound, domain_upper_bound):
    ''' this function takes in a population, a mutation rate and a lower and upper bound.
    It applies uniform random mutation to each individual. Each gene of each individual has
    mutation rate probability of being changed within the range of the lower and upper bound. '''
    # loop through individuals in population
    for individual in population:

        # loop through each gene in the individual
        for i in range(len(individual)):

            # get random value and see if it is under the mutation rate
            if np.random.rand() < mutation_rate:

                # mutate gene with uniform random value within the range
                individual[i] = np.random.uniform(domain_lower_bound, domain_upper_bound)

    return population

# define a function to make the next generation of individuals
def next_generation(population, pop_fitness, x_percent, alpha, tournament_size):
    ''' this function takes in the current population and outputs the next generation
    It does this by retaining the top x percent solutions of the current population to exploit current good solutions
    and then recombines parents via tournament selection to create children to make up the
    rest of the next generation (100 - x% of the population size) '''

    # number of best individuals to keep based on chosen percentage and population size
    n_best = int(population.shape[0] * (x_percent / 100))

    # best x percent indices from population
    best_indices = np.argsort(pop_fitness)[::-1][:n_best]

    # add n best individuals to next generation
    best_individuals = population[best_indices]

    # obtain children and add to next generation to make up the rest of the next generation
    new_children = intermediate_recombination(population, pop_fitness, x_percent, alpha, tournament_size)

    # combine best individuals and the newly created children to make the next generation
    next_generation = np.vstack((best_individuals, new_children))

    return next_generation

# function to train a population with given parameters
def evolve_population(env, population_size, generations, mutation_rate, x_percent, alpha, tournament_size):
    ''' evolves a population with the given parameters and returns the best individual'''
    num_variables = (env.get_num_sensors() + 1) * 10 + (10 + 1) * 5
    population = np.random.uniform(-1, 1, (population_size, num_variables))
    pop_fitness = evaluate(env, population)

    for gen in range(generations):
        population = next_generation(population, pop_fitness, x_percent, alpha, tournament_size)
        population = uniform_mutation(population, mutation_rate, -1, 1)
        pop_fitness = evaluate(env, population)

    best_individual = population[np.argmax(pop_fitness)]
    return best_individual

def main():
    # define the four different parameter configurations
    configurations = [
        {"population_size": 183, "mutation_rate": 0.053, "x_percent": 23, "tournament_size": 8, "alpha": 0.41},
        {"population_size": 111, "mutation_rate": 0.097, "x_percent": 34, "tournament_size": 4, "alpha": 0.76},
        {"population_size": 178, "mutation_rate": 0.05, "x_percent": 50, "tournament_size": 7, "alpha": 0.97},
        {"population_size": 147, "mutation_rate": 0.05, "x_percent": 45, "tournament_size": 8, "alpha": 0.305}
    ]

    enemy_combinations = list(itertools.combinations(range(1, 9), 3))

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'multi_parameter_tuning'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initialise environment with all 8 enemies
    n_hidden_neurons = 10
    all_enemies_env = Environment(experiment_name=experiment_name,
                                  enemies=list(range(1, 9)),
                                  multiplemode="yes",
                                  playermode="ai",
                                  player_controller=player_controller(n_hidden_neurons),
                                  enemymode="static",
                                  level=2,
                                  speed="fastest",
                                  visuals=False)

    # loop through parameter configurations and enemy combinations
    for config_idx, config in enumerate(configurations):
        print(f"Testing configuration {config_idx + 1}: {config}")

        for enemy_group in enemy_combinations:
            print(f"Training on enemies {enemy_group}")

            # initialise envirobment for this enemy group
            env = Environment(experiment_name=experiment_name,
                              enemies=list(enemy_group),
                              multiplemode="yes",
                              playermode="ai",
                              player_controller=player_controller(n_hidden_neurons),
                              enemymode="static",
                              level=2,
                              speed="fastest",
                              visuals=False)

            # evolve population with this group and parameter settings
            best_individual = evolve_population(env,
                                               config["population_size"],
                                               generations=30,
                                               mutation_rate=config["mutation_rate"],
                                               x_percent=config["x_percent"],
                                               alpha=config["alpha"],
                                               tournament_size=config["tournament_size"])

            # test the best found individual against all enemies
            best_fitness = simulation(all_enemies_env, best_individual)
            print(f"Best fitness for enemies {enemy_group} with config {config_idx + 1}: {best_fitness}")

if __name__ == '__main__':
    main()
