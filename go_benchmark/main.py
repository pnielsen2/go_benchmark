import torch
from building_blocks import Node, generate_games, pull_data, clear_tree

def create_go_dataset(dataset_size, board_dimension, network=None, test=True):
        num_possible_moves = board_dimension**2+1
        root = Node(num_possible_moves,'black')

        run_stats = {'unique_nodes':1, 'positions_visited':0, 'transitions_observed': 0}
        generate_games(root, run_stats, dataset_size//2, network)
        x, outcomes = pull_data(root)
        x_train, y_train = torch.cat(x), torch.stack(outcomes).to(dtype=torch.float32)
        print('x shape:', x_train.shape)
        print('y shape:', y_train.shape)

        clear_tree(root)

        run_stats = {'unique_nodes':1, 'positions_visited':0, 'transitions_observed': 0}
        generate_games(root, run_stats, dataset_size//2, network)
        x, outcomes = pull_data(root)
        x_test, y_test = torch.cat(x), torch.stack(outcomes).to(dtype=torch.float32)
        print('x shape:', x_test.shape)
        print('y shape:', y_test.shape)
        

        m = {'Train':{'x': x_train, 'y': y_train}}
        if test:
            m.update({'Test':{'x': x_test, 'y': y_test}})

        # torch.save(m,f'datasets/{dataset_size}_{board_dimension}x{board_dimension}.p')
        return m
