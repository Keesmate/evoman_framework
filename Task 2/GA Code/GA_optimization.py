###############################################################################
# EvoMan FrameWork - V1.0 2016                                                #
# GA optimization                #
###############################################################################


# imports framework
import sys
from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import matplotlib.pyplot as plt
import os

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
def next_generation(population, pop_fitness, x_percent, alpha, mutation_rate, domain_lower_bound, domain_upper_bound, tournament_size):
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

    # mutate the new children
    mutated_children = uniform_mutation(new_children, mutation_rate, domain_lower_bound, domain_upper_bound)

    # combine best individuals and the newly created children to make the next generation
    next_generation = np.vstack((best_individuals, mutated_children))

    return next_generation


def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'optimization_group_2_parameters_2_extra'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # choose the enemy group to train against
    enemy_list = [2, 5, 6]

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=enemy_list,
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

    num_variables = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # GA Parameters
    domain_upper_bound = 1
    domain_lower_bound = -1
    pop_number = 154
    generations = 30
    mutation_rate = 0.0517
    alpha = 0.115
    tournament_size = 5
    x_percent = 24
    num_experiments = 10

    mean_fitness_experiments = []
    best_fitness_experiments = []
    best_individuals = []

    # loop for n experiments
    for i in range(num_experiments):
        average_gains = []
        mean_fitness_generation = []
        best_fitness_generation = []
        population = np.random.uniform(domain_lower_bound, domain_upper_bound, (pop_number, num_variables))
        pop_fitness = evaluate(env, population)
        best_fitness_generation.append(np.max(pop_fitness))
        mean_fitness_generation.append(np.mean(pop_fitness))

        current_best_individual = population[np.argmax(pop_fitness)]
        current_best_individual_fitness = simulation(env, current_best_individual)

        # loop through number of generatoins
        for j in range(generations - 1):

            population = next_generation(population, pop_fitness, x_percent, alpha, mutation_rate, domain_lower_bound, domain_upper_bound, tournament_size)
            pop_fitness = evaluate(env, population)

            best_fitness_generation.append(np.max(pop_fitness))
            mean_fitness_generation.append(np.mean(pop_fitness))

            if simulation(env, population[np.argmax(pop_fitness)]) > current_best_individual_fitness:
                current_best_individual = population[np.argmax(pop_fitness)]

        mean_fitness_experiments.append(mean_fitness_generation)
        best_fitness_experiments.append(best_fitness_generation)
        best_individuals.append(list(current_best_individual))

        print(f"Run {i + 1} complete")

        individual_gains = []
        total_gains = []

        # test the best individual against all 8 enemies in separate test environments
        test_enemies = list(range(1, 9))
        for enemy in test_enemies:
            test_env = Environment(experiment_name=experiment_name,
                    enemies=[enemy],
                    multiplemode="no",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

            f, p, e, t = test_env.play(pcont=current_best_individual)

            player_energy = test_env.get_playerlife()
            enemy_energy = test_env.get_enemylife()

            print(f"Enemy {enemy}:")
            print(f"Player Energy: {player_energy}")
            print(f"Enemy Energy: {enemy_energy}")
            print(f"Fitness: {f}")

            individual_gain = player_energy - enemy_energy
            individual_gains.append((enemy, individual_gain))
            total_gains.append(individual_gain)

        for enemy, gain in individual_gains:
            print(f"Enemy {enemy}: Individual Gain = {gain}")

        average_gain = np.mean(total_gains)
        average_gains.append(average_gain)
        print(f"Average Individual Gain Over All Enemies: {average_gain}")

    # calculate final statistics for line plot
    mean_mean_fitness = np.mean(mean_fitness_experiments, axis=0)
    std_mean_fitness = np.std(mean_fitness_experiments, axis=0)
    mean_max_fitness = np.mean(best_fitness_experiments, axis=0)
    std_max_fitness = np.std(best_fitness_experiments, axis=0)

    print(individual_gains)
    print(average_gains)

    # save the best individuals' values to a file
    with open(f"{experiment_name}/best_individuals.txt", 'w') as file:
        for i, individual in enumerate(best_individuals):
            file.write(f"Best individual for experiment {i + 1}: {individual}\n")

    # save the average statistics for each generation over the 10 runs
    with open(f"{experiment_name}/average_generation_statistics.txt", 'w') as file:
        file.write("gen \t mean_fitness \t std_mean_fitness \t max_fitness \t std_max_fitness\n")
        for gen in range(len(mean_mean_fitness)):
            file.write(f"{gen} \t {round(mean_mean_fitness[gen], 6)} \t {round(std_mean_fitness[gen], 6)} \t\t {round(mean_max_fitness[gen], 6)} \t\t {round(std_max_fitness[gen], 6)}\n")

    # create line-plot: generations on the x axis, with the average mean and average maximum (with std) over n generations per run on the y axis
    generations = np.arange(len(mean_mean_fitness))
    plt.figure(figsize=(10, 6))
    plt.errorbar(generations, mean_mean_fitness, yerr=std_mean_fitness, fmt='-o', label='Mean Fitness', capsize=3)
    plt.errorbar(generations, mean_max_fitness, yerr=std_max_fitness, fmt='-s', label='Max Fitness', capsize=3)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Average Mean and Max Fitness Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

    # create a boxplot for the 10 best individuals average individual gains against this enemy
    plt.figure(figsize=(8, 6))
    plt.boxplot(average_gains, vert=False)

    # add labels and title
    plt.title('Boxplot of Individual Gains of Best Individual Across 10 Runs')
    plt.ylabel('Individual Gains')
    plt.show()


if __name__ == '__main__':
    main()
