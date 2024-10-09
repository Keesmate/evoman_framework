import optuna
import numpy as np
import matplotlib.pyplot as plt
import os
from evoman.environment import Environment
from demo_controller import player_controller

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

# objective function for Optuna
def objective(trial):
    ''' this function will perform parameter tuning for various parameters. The enemy group to be
    trained on can be altered and the code re-run to get differently tuned parameters for
    differnt enemy groups.
    '''
    # define the hyperparameters to be tuned 
    population_size = trial.suggest_int("population_size", 50, 200)
    mutation_rate = trial.suggest_float("mutation_rate", 0.05, 0.5)
    x_percent = trial.suggest_int("x_percent", 5, 50)
    tournament_size = trial.suggest_int("tournament_size", 2, 8)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)

    # set up envirobment
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'final_tuning'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10
    env = Environment(experiment_name=experiment_name,
                      enemies=[2, 4, 5, 6, 8],
                      multiplemode="yes",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    num_variables = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # initialise pop
    population = np.random.uniform(-1, 1, (population_size, num_variables))
    pop_fitness = evaluate(env, population)
    generations = 30

    # loop through the generations and evolve
    for gen in range(generations):
        population = next_generation(population, pop_fitness, x_percent, alpha, tournament_size)
        population = uniform_mutation(population, mutation_rate, -1, 1)
        pop_fitness = evaluate(env, population)

    best_fitness = np.max(pop_fitness)
    return best_fitness


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best hyperparameters:", study.best_trial.params)
    print("Best fitness:", study.best_value)


if __name__ == '__main__':
    main()
