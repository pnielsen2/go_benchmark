from .building_blocks import generate_games, Node
from alpharedmond.game_simulator import GameSim


def benchmark_network(network, board_dim):
    gamesim = GameSim(board_dim, None, 'cpu')
    num_possible_moves = board_dim**2+1
    root = Node(num_possible_moves,'black', gamesim.record())
    visited_nodes = {gamesim.position_hash:root}
    run_stats = {'unique_nodes':1, 'positions_visited':0, 'transitions_observed': 0}
    generate_games(root, run_stats, float('inf'), visited_nodes, gamesim, network=network)
    return run_stats