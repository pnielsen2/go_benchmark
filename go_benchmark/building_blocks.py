
from alpharedmond.game_simulator import GameSim
import torch
import pyro
import numpy as np
import copy
import random

class Node():
    def __init__(self,num_possible_moves, current_player,reset_data, parent_node_move_pair=None):
        self.num_possible_moves = num_possible_moves
        # last_hashes=[[0]]*8
        self.next_nodes = {}
        self.outcomes = torch.zeros(self.num_possible_moves,2)
        self.children_nodes_solved = torch.zeros(self.num_possible_moves)
        self.solved = 0
        self.current_player = current_player
        self.network_input = None
        self.predictions = None
        self.previous_posterior_alphas = None
        self.reset_data = reset_data
        self.parents = [] if parent_node_move_pair == None else [parent_node_move_pair]
        self.position_hash = copy.deepcopy(self.reset_data[9])

def pull_data(node):
    x = [node.network_input]
    outcomes = [node.outcomes]
    position_hashes = [node.position_hash]
    for child_node in node.next_nodes:
        next_node = node.next_nodes[child_node]
        if next_node.outcomes.sum()>0:
            child_x, child_outcomes, child_position_hashes = pull_data(next_node)
            x += child_x
            outcomes += child_outcomes
            position_hashes += child_position_hashes
    return x, outcomes, position_hashes

def tree_explorer(root, index_function=None):
    node = root
    while True:
        wins, losses = node.outcomes[:,0], node.outcomes[:,1]
        print('node solved:', node.solved)
        print('children nodes solved:', node.children_nodes_solved)
        print('move wins:', (wins))
        print('move losses:', (losses))
        print('position wins:', wins.sum())
        print('position losses:', losses.sum())
        print(node.network_input[0,-1])
        if index_function!=None:
            print('move indices:', index_function(wins, wins+losses, (node.children_nodes_solved!=-1)))
        node = node.next_nodes[int(input())]

def update_search_tree(node_move_pairs, winner):
    for i in range(len(node_move_pairs)):
        node, move = node_move_pairs[-1-i]
        current_player_won = node.current_player == winner
        
        if i!=0:
            resulting_node = node.next_nodes[move]
            equivalent_moves = [move for move in node.next_nodes if node.next_nodes[move] == resulting_node]
            solutions = [-node.next_nodes[m].solved for m in equivalent_moves]
            solution = 1 if max(solutions)==1 else (-1 if min(solutions)==-1 else 0)
            for m in equivalent_moves:
                node.children_nodes_solved[m] = solution
        else:
            equivalent_moves = [move]
            node.children_nodes_solved[move] = 2*current_player_won-1
            
        
        node.outcomes[equivalent_moves] = torch.max(node.outcomes[equivalent_moves],dim=0)[0]
        
        if current_player_won:
            node.outcomes[equivalent_moves,0] +=1
        else:
            node.outcomes[equivalent_moves,1] +=1
            
        if (node.children_nodes_solved==1).any():
            node.solved = 1
        elif (node.children_nodes_solved==-1).all():
            node.solved = -1
            pass

def compute_MOSSE_indices(rewards, pulls, unsolved_nodes):
    """ Compute the current indexes for all arms, in a vectorized manner."""
    t = pulls.sum()
    pulls_of_suboptimal_arms = np.sum(pulls[(pulls < np.sqrt(t))*unsolved_nodes])

    if pulls_of_suboptimal_arms > 0:
        indices = (rewards / pulls) + np.sqrt(0.5 * np.maximum(0, np.log(t / pulls_of_suboptimal_arms)) / pulls)
    else:
        indices = (rewards / pulls) + np.sqrt(0.5 * np.maximum(0, np.log(t / (unsolved_nodes.sum() * pulls))) / pulls)
    return indices*unsolved_nodes


def choose_move(node, gamesim, run_stats, network):
    if node.network_input==None:
        player_indicator = torch.zeros(1,2,gamesim.dimension,gamesim.dimension)
        player_indicator[0,{'black':0, 'white':1}[node.current_player]] +=1
        network_input = torch.cat((player_indicator, gamesim.input_history[-8:])).view(1,18, gamesim.dimension, gamesim.dimension)
        node.network_input = network_input
    if node.predictions==None:
        node.predictions = (network(node.network_input).squeeze(0) if network != None 
                            else (node.previous_posterior_alphas if node.previous_posterior_alphas != None 
                                  else torch.zeros_like(node.outcomes) + .5))
    game_winner = None
    unsolved_nodes = (node.children_nodes_solved!=-1)
    posterior_alphas = node.outcomes + node.predictions
    indices = pyro.distributions.Beta(posterior_alphas[:,0], posterior_alphas[:,1]).sample()
    indices = indices*unsolved_nodes
    gamesim_set=False
    while True:
        chosen_move = indices.argmax().item()
        move_xy = (chosen_move // gamesim.dimension, chosen_move % gamesim.dimension)
        # if move is not already solved for the opponent:
        if node.children_nodes_solved[chosen_move] == 0:
            if chosen_move not in node.next_nodes:
                if not gamesim_set:
                    gamesim.set(node.reset_data)
                    gamesim_set=True
                if gamesim.step(move_xy):
                    run_stats['transitions_observed'] +=1
                    game_winner = gamesim.winner
                    break
                else:
                    node.children_nodes_solved[chosen_move]=-1
            else:
                break
        indices[chosen_move]=0
        if not indices.any():
            node.solved = -1
            break
    return chosen_move, game_winner


def play_game(root, run_stats, visited_nodes, gamesim, network=None):
    node=root
    node_move_pairs = []
    while True:
        run_stats['positions_visited'] +=1
        if node.solved==1:
            game_winner = node.current_player
            break
        elif node.solved == -1:
            game_winner = {'black':'white', 'white':'black'}[node.current_player]
            break
        chosen_move, game_winner = choose_move(node, gamesim, run_stats, network)
        node_move_pairs.append((node, chosen_move))
        if node.solved == -1:
            game_winner = {'black':'white', 'white':'black'}[node.current_player]
            break
        elif node.solved == 1:
            game_winner = node.current_player
            break
        if game_winner!=None:
            break
        elif chosen_move not in node.next_nodes:
            if gamesim.position_hash not in visited_nodes:
                visited_nodes[gamesim.position_hash] = Node(node.num_possible_moves, gamesim.current_player, gamesim.record())
                run_stats['unique_nodes'] +=1
            node.next_nodes[chosen_move] = visited_nodes[gamesim.position_hash]
        node = node.next_nodes[chosen_move]
    if node_move_pairs[-1] not in node.parents:
        node.parents.append(node_move_pairs[-1])
    update_search_tree(node_move_pairs, game_winner)
    return game_winner

def clear_tree(node):
    node.previous_posterior_alphas = (node.predictions if node.predictions != None else .5) + node.outcomes
    node.outcomes = torch.zeros_like(node.outcomes)
    node.predictions = None
    for child_node in node.next_nodes:
        clear_tree(node.next_nodes[child_node])

def generate_games(root, run_stats, transitions, visited_nodes, gamesim, network=None):
    initial_transitions_observed = run_stats['transitions_observed']
    while True:
        if root.solved !=0 or run_stats['transitions_observed']-initial_transitions_observed > transitions:
            print(root.solved)
            break
        game_winner = play_game(root, run_stats, visited_nodes, gamesim, network)
        print('transitions observed: ', run_stats['transitions_observed'])
    for stat in run_stats:
        print(stat, run_stats[stat])
    print('done generating games')

def create_go_dataset(train_size, test_size, device='cpu', board_dimension=None, run_stats=None, root=None, network=None, visited_nodes=None, gamesim=None):
    if gamesim == None:
        gamesim = GameSim(board_dimension, None, 'cpu')
    if root==None:
        root = Node(board_dimension**2+1, 'black', gamesim.record())
        visited_nodes = {gamesim.position_hash:root}
    else:
        clear_tree(root)

    if run_stats==None:
        run_stats = {'unique_nodes':1, 'positions_visited':0, 'transitions_observed': 0}
    solved = generate_games(root, run_stats, train_size, visited_nodes, gamesim, network)
    x, outcomes = pull_data(root)
    x_train, y_train = torch.cat(x).to(device), torch.stack(outcomes).to(device, dtype=torch.float32)
    m = {'Train':{'x': x_train, 'y': y_train}}
    if solved:
        return m, root, run_stats, solved, gamesim, visited_nodes

    clear_tree(root)
    
    if test_size > 0:
        solved = generate_games(root, run_stats, test_size, visited_nodes, gamesim, network)
        x, outcomes = pull_data(root)
        x_test, y_test = torch.cat(x).to(device), torch.stack(outcomes).to(device, dtype=torch.float32)
        m.update({'Test':{'x': x_test, 'y': y_test}})
    # torch.save(m,f'datasets/{dataset_size}_{board_dimension}x{board_dimension}.p')
    return m, root, run_stats, solved, gamesim, visited_nodes