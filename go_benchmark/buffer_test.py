from training_infra.train import train
from training_infra.configs import train_config as config
import training_infra.networks as networks
from training_infra.activations import hidden_activation_dict, output_activation_dict
from training_infra.dataset_loaders import dataset_facts
from alpharedmond.game_simulator import GameSim
from building_blocks import generate_games, Node, pull_data, clear_tree

import torch
from collections import deque


dataset_name = '3x3_go'

class TensorQueue:
    def __init__(self, capacity):
        self.queue = deque()
        self.capacity = capacity
        self.hash_map = {}
        self.x_map = {}
    
    def enqueue(self, x, observations, position_hashes):
        assert len(x) == len(observations) == len(position_hashes), "All input lists must have the same length"
        excess = []
        for x_tensor, obs_tensor, hash_value in zip(x, observations, position_hashes):
            x_tensor = x_tensor.cpu().clone()  # Clone to ensure no references are passed
            obs_tensor = obs_tensor.cpu().clone()
            if len(self.queue) >= self.capacity:
                old_hash, old_x, old_obs = self.queue.popleft()
                if old_hash in self.hash_map:
                    self.hash_map[old_hash] -= old_obs
                    if (self.hash_map[old_hash] <= 0).all():  # Check and handle non-positive values
                        del self.hash_map[old_hash]
                        del self.x_map[old_hash]
                excess.append((old_hash, old_x, old_obs))
            self.queue.append((hash_value, x_tensor, obs_tensor))
            if hash_value in self.hash_map:
                self.hash_map[hash_value] += obs_tensor
            else:
                self.hash_map[hash_value] = obs_tensor.clone()  # Clone to ensure proper management
                self.x_map[hash_value] = [x_tensor]
        return excess

    def get_dataset(self, device='cpu'):
        x_data = [tensors[0] for tensors in self.x_map.values()]
        y_data = torch.cat([self.hash_map[hash_value].unsqueeze(0) for hash_value in self.hash_map], dim=0).to(device)
        return {'x': torch.stack(x_data).to(device), 'y': y_data}

class Buffer:
    def __init__(self, test_capacity, train_capacity):
        self.test_queue = TensorQueue(test_capacity)
        self.train_queue = TensorQueue(train_capacity)

    def add_data(self, x, observations, position_hashes):
        excess = self.test_queue.enqueue(x, observations, position_hashes)
        if excess:
            excess_x = [item[1] for item in excess]
            excess_observations = [item[2] for item in excess]
            excess_hashes = [item[0] for item in excess]
            self.train_queue.enqueue(excess_x, excess_observations, excess_hashes)

    def get_dataset(self, device='cpu'):
        test_dataset = self.test_queue.get_dataset(device)
        train_dataset = self.train_queue.get_dataset(device)
        return {'Train': train_dataset, 'Test': test_dataset}
    
    def get_size(self):
        return {'Train': len(self.train_queue.queue), 'Test': len(self.test_queue.queue)}

        

device = torch.device("cuda:0" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
network = networks.architecture_dict[config['architecture']](hidden_act = hidden_activation_dict[config['hidden_activation']],
                                                    output_act = output_activation_dict[config['output_activation']],
                                                    width = config['network_width'],
                                                    depth = config['network_depth'],
                                                    input_dimensions=dataset_facts[dataset_name]['x_dimensions'],
                                                    output_dimensions=dataset_facts[dataset_name]['y_dimensions']
                                                    )

board_dimension = 3
initial_transitions = 20000
buffer = Buffer(test_capacity=20000, train_capacity=1000000)
gamesim = GameSim(board_dimension, None, 'cpu', komi=1.5)
root = Node(board_dimension**2+1, 'black', gamesim.record())
visited_nodes = {gamesim.position_hash:root}
run_stats = {'unique_nodes':1, 'positions_visited':0, 'transitions_observed': 0}
generate_games(root, run_stats, initial_transitions, visited_nodes, gamesim, network=None)

while True:
    x, outcomes, position_hashes = pull_data(root)
    print('adding data')
    buffer.add_data(x, outcomes, position_hashes)
    print('done adding data')
    print('train length')
    print(buffer.get_size())
    if root.solved:
        break
    print('getting dataset')
    dataset = buffer.get_dataset('mps')
    print('Train x:', dataset['Train']['x'].shape)
    print('Train y:', dataset['Train']['y'].shape)
    print('Test x:', dataset['Test']['x'].shape)
    print('Test y:', dataset['Test']['y'].shape)
    print('done getting dataset')
    network = train(network.to(device), dataset, f'{board_dimension}x{board_dimension}_go_tests', using_wandb=False).to('cpu')
    model = lambda x: network(x)['alpha']
    clear_tree(root)
    generate_games(root, run_stats, 10000, visited_nodes, gamesim, network=model)

