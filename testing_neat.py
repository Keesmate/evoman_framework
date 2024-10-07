import os
import pickle
import csv
import neat
import numpy as np
from evoman.environment import Environment
from evoman.controller import Controller
from neat_controller import player_controller


class NEATController(Controller):
    def __init__(self, network):
        self.network = network

    def control(self, inputs, controller):
        # Activate the network with the inputs from the environment
        outputs = self.network.activate(inputs)
        return outputs


def main():
    experiment_name = 'neat_optimization_test'
    num_experiments = 10  # Assuming 10 experiments were conducted previously
    enemy_list = [1, 2, 3, 4, 5, 6, 7, 8]  # Compete against all 8 enemies

    # Create a dictionary to store the gain statistics
    gain_stats = {f"Network {i+1}": {f"Enemy {enemy}": [] for enemy in enemy_list} for i in range(num_experiments)}

    # Load the NEAT configuration
    config_file = 'config-feedforward-0'  # Assuming the config file is shared across experiments
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Loop over all experiments (networks)
    for experiment in range(num_experiments):
        # Load the saved network (best individual) from the previous experiment
        network_file = os.path.join(experiment_name, f'best_individual_experiment_{experiment + 1}.pkl')
        with open(network_file, 'rb') as f:
            winner = pickle.load(f)
        
        # Compete against each enemy
        for enemy in enemy_list:
            # Initializes the environment for the current enemy
            env = Environment(
                experiment_name=experiment_name,
                enemies=[enemy],
                playermode="ai",
                player_controller=None,
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False,
                multiplemode="no"
            )

            # Set the controller using the loaded network
            net = neat.nn.FeedForwardNetwork.create(winner, config)
            controller = player_controller(net)
            env.player_controller = controller

            # Play the game 5 times and average the gain
            gains = []
            for _ in range(5):
                fitness, player_life, enemy_life, time = env.play()
                gain = player_life - enemy_life
                gains.append(gain)

            # Calculate the average gain against the current enemy
            average_gain = np.mean(gains)
            gain_stats[f"Network {experiment + 1}"][f"Enemy {enemy}"] = average_gain
            print(f"Network {experiment + 1} vs Enemy {enemy}: Average Gain = {average_gain}")

    # Save the gain statistics to a CSV file
    output_csv = os.path.join(experiment_name, 'network_vs_enemy_gain_stats.csv')
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['Network', 'Enemy 1', 'Enemy 2', 'Enemy 3', 'Enemy 4', 'Enemy 5', 'Enemy 6', 'Enemy 7', 'Enemy 8']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for network in gain_stats:
            row = {'Network': network}
            row.update({f'Enemy {enemy}': gain_stats[network][f'Enemy {enemy}'] for enemy in enemy_list})
            writer.writerow(row)

    print(f"Gain statistics saved to {output_csv}")


if __name__ == '__main__':
    main()
