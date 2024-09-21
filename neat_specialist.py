#run pip install neat

# imports framework
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Import NEAT library
import neat
from neat.nn import FeedForwardNetwork

from evoman.environment import Environment
from evoman.controller import Controller

from neat_controller import player_controller

# Define a NEAT-compatible controller
class NEATController(Controller):
    def __init__(self, network):
        self.network = network

    def control(self, inputs, controller):
        # Activate the network with the inputs from the environment
        outputs = self.network.activate(inputs)
        # The outputs should be in the correct format expected by the environment
        return outputs

def main():
    # Choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'neat_optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        enemies=[2],  # You can change the enemy number here
        playermode="ai",
        player_controller=None,  # We'll set the controller later
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False
    )

    num_inputs = env.get_num_sensors()
    num_outputs = 5  # Number of action outputs
    num_experiments = 10
    generations = 30

    # Initialize lists for mean and max fitness over generations
    mean_fitness_experiments = []
    best_fitness_experiments = []
    best_individuals = []

    for experiment in range(num_experiments):
        # Create NEAT configuration file dynamically
       config_content = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 100000
pop_size              = 100
reset_on_extinction   = False

[DefaultGenome]
# Node activation options
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

# Node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# Node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# Node response options
response_init_mean          = 1.0
response_init_stdev         = 0.0
response_max_value          = 30.0
response_min_value          = -30.0
response_mutate_power       = 0.0
response_mutate_rate        = 0.0
response_replace_rate       = 0.0

# Genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# Connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# Connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

# Connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

# Node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# Genome node gene parameters
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}
initial_connection      = full
feed_forward            = true
recursive               = false

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""


    # Write the configuration to a file
    config_file = f"config-feedforward-{experiment}"
    with open(config_file, 'w') as f:
        f.write(config_content)

    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Create the population
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Define the fitness function
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            # Create the network
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            controller = player_controller(net) #NEATController(net)

            # Set the controller in the environment
            env.player_controller = controller

            # Play the game using the genome's network
            fitness, player_life, enemy_life, time = env.play()

            # Assign fitness to the genome
            genome.fitness = fitness

    # Run NEAT algorithm
    winner = p.run(eval_genomes, n=generations)

    # Collect statistics
    generation_fitnesses = np.array(stats.get_fitness_mean())
    max_fitnesses = np.array([c.fitness for c in stats.most_fit_genomes])

    mean_fitness_experiments.append(generation_fitnesses)
    best_fitness_experiments.append(max_fitnesses)
    best_individuals.append(winner)

    print(f"Experiment {experiment + 1} complete")

    # Clean up the config file
    os.remove(config_file)

    # Calculate final statistics for plotting
    mean_mean_fitness = np.mean(mean_fitness_experiments, axis=0)
    std_mean_fitness = np.std(mean_fitness_experiments, axis=0)
    mean_max_fitness = np.mean(best_fitness_experiments, axis=0)
    std_max_fitness = np.std(best_fitness_experiments, axis=0)

    # Create line plot
    generation_axis = np.arange(1, generations + 1)
    plt.figure(figsize=(10, 6))
    plt.errorbar(generation_axis, mean_mean_fitness, yerr=std_mean_fitness, fmt='-o', label='Mean Fitness', capsize=3)
    plt.errorbar(generation_axis, mean_max_fitness, yerr=std_max_fitness, fmt='-s', label='Max Fitness', capsize=3)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('NEAT: Average Mean and Max Fitness Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
