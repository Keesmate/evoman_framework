# Import necessary libraries
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import csv  # Added to handle CSV writing
import neat
from neat.nn import FeedForwardNetwork
from evoman.environment import Environment
from evoman.controller import Controller
from neat_controller import player_controller
import pickle  # Added to handle saving the best networks


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

    enemy_number = [2, 6, 8]

    # Initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        enemies=enemy_number,
        playermode="ai",
        player_controller=None,
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
        multiplemode='yes'
    )

    num_inputs = env.get_num_sensors()
    num_outputs = 5  # Number of action outputs
    num_experiments = 10
    generations = 30

    # Initialize lists for mean and max fitness over generations
    mean_fitness_experiments = []
    best_fitness_experiments = []
    best_individuals = []
    individual_gains = []  # Added to store gains of best individuals

    experiment_numbers = []  # Added to store experiment numbers

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
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid

# Node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.15
aggregation_options     = sum

# Node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.250
bias_mutate_rate        = 0.822
bias_replace_rate       = 0.1

# Node response options
response_init_mean          = 1.0
response_init_stdev         = 0.0
response_max_value          = 30.0
response_min_value          = -30.0
response_mutate_power       = 0.0
response_mutate_rate        = 0.0 
response_replace_rate       = 0.0

# Genome compatibility options - species management
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.4

# Connection add/remove rates
conn_add_prob           = 0.090
conn_delete_prob        = 0.447

# Connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

# Connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.510
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.108
weight_mutate_rate      = 0.984
weight_replace_rate     = 0.1

# Node add/remove rates
node_add_prob           = 0.376
node_delete_prob        = 0.895

# Genome node gene parameters
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}
initial_connection      = full
feed_forward            = true
recursive               = false

[DefaultSpeciesSet]
compatibility_threshold = 1.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 5
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
                controller = player_controller(net)  # NEATController(net)

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

        # Save the best individual (winner) to a file
        winner_file = os.path.join(experiment_name, f'best_individual_experiment_{experiment + 1}.pkl')
        with open(winner_file, 'wb') as f:
            pickle.dump(winner, f)
        print(f"Best individual from experiment {experiment + 1} saved to {winner_file}")

        # Evaluate the best individual multiple times and calculate average gain
        gains = []
        for _ in range(5):
            # Create the network for the best individual
            net = neat.nn.FeedForwardNetwork.create(winner, config)
            controller = player_controller(net)  # NEATController(net)
            env.player_controller = controller

            # Play the game
            fitness, player_life, enemy_life, time = env.play()
            gain = player_life - enemy_life
            gains.append(gain)

        average_gain = np.mean(gains)
        individual_gains.append(average_gain)
        experiment_numbers.append(experiment + 1)  # Keep track of experiment numbers

        # Clean up the config file
        os.remove(config_file)

    # Save the individual gains to a CSV file
    output_csv = os.path.join(experiment_name, f'individual_gains_enemy_{enemy_number}.csv')
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['Experiment', 'Average_Gain']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for exp_num, gain in zip(experiment_numbers, individual_gains):
            writer.writerow({'Experiment': exp_num, 'Average_Gain': gain})

    print(f"Average gains saved to {output_csv}")

    # Calculate final statistics for plotting
    mean_mean_fitness = np.mean(mean_fitness_experiments, axis=0)
    std_mean_fitness = np.std(mean_fitness_experiments, axis=0)
    mean_max_fitness = np.mean(best_fitness_experiments, axis=0)
    std_max_fitness = np.std(best_fitness_experiments, axis=0)

    # Save the statistics to a new CSV file
    stats_csv = os.path.join(experiment_name, f'generation_stats_enemy_{enemy_number}.csv')
    with open(stats_csv, mode='w', newline='') as stats_file:
        writer = csv.writer(stats_file)
        writer.writerow(['Generation', 'Mean_Fitness', 'Std_Mean_Fitness', 'Max_Fitness', 'Std_Max_Fitness'])
        for gen in range(generations):
            writer.writerow([gen + 1, mean_mean_fitness[gen], std_mean_fitness[gen], mean_max_fitness[gen], std_max_fitness[gen]])

    print(f"Generation statistics saved to {stats_csv}")

    # Create line plot and save it as a jpg
    generation_axis = np.arange(1, generations + 1)
    plt.figure(figsize=(10, 6))
    plt.errorbar(generation_axis, mean_mean_fitness, yerr=std_mean_fitness, fmt='-o', label='Mean Fitness', capsize=3)
    plt.errorbar(generation_axis, mean_max_fitness, yerr=std_max_fitness, fmt='-s', label='Max Fitness', capsize=3)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title(f'NEAT: Average Mean and Max Fitness Over Generations (Enemy {enemy_number})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(experiment_name, f'fitness_plot_enemy_{enemy_number}.jpg'))
    plt.show()

    # Create a boxplot for the individual gains of best individuals across experiments and save it as a jpg
    plt.figure(figsize=(8, 6))
    plt.boxplot(individual_gains, vert=False)
    plt.title(f'Boxplot of Individual Gains of Best Individuals (Enemy {enemy_number})')
    plt.xlabel('Individual Gains')
    plt.yticks([1])
    plt.savefig(os.path.join(experiment_name, f'gain_boxplot_enemy_{enemy_number}.jpg'))
    plt.show()


if __name__ == '__main__':
    main()
