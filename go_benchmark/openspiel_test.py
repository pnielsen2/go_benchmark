import random
import pyspiel
import numpy as np
import torch
from openspiel_buffer import Buffer, DictBuffer
import pyro
from training_infra.train import train
# import line_profiler
import memory_profiler
import gc
from rl_config import rl_config, game_params
import scipy
import os


def MOSSEindices(posteriors, player):
    """ Compute the current indexes for all arms, in a vectorized manner."""
    posteriors = posteriors+0
    t = posteriors.sum()
    pulls = posteriors.sum(axis=-1)
    pulls_of_suboptimal_arms = (pulls[pulls < torch.sqrt(t)]).sum()
    rewards = posteriors[:,player]
    if pulls_of_suboptimal_arms > 0:
        indexes = (rewards / pulls) + torch.sqrt(0.5 * torch.clamp(torch.log(t / pulls_of_suboptimal_arms), min=0) / pulls)
    else:
        indexes = (rewards / pulls) + torch.sqrt(0.5 * torch.clamp(torch.log(t / (len(posteriors) * pulls)), min=0) / pulls)
    # indexes[pulls < 1] = float('+inf')
    return indexes

def symmetrized_obs(observation_tensor, game_symmetry_info, board_size):
    tensor_list = []
    reshaped_tensor = torch.tensor(observation_tensor).view(-1,board_size, board_size)
    for i in range(game_symmetry_info['num_game_symmetries']):
        tensor_list.append(tuple(game_symmetry_info['symmetry_function'](reshaped_tensor, i).reshape(-1).tolist()))
    return sorted(tensor_list)[0]
    
def go_symmetries(tensor, i):
    _, n, _ = tensor.shape
    actions = torch.arange(n*n).reshape(n,n)
    rotation = i % 4
    flip = i // 4
    tensor, actions = torch.rot90(tensor, rotation, (-1,-2)), torch.rot90(actions, -rotation, (-1,-2))
    if flip:
        tensor, actions = torch.flip(tensor, (-1,)), torch.flip(actions, (-1,))
    tensor, actions = tuple(tensor.reshape(-1).tolist()), actions.reshape(-1).tolist() + [n*n]
    return tensor, actions

def hex_symmetries(tensor, i):
    _, n, _ = tensor.shape
    actions = torch.arange(n*n).reshape(n,n)
    rotation = i % 2
    flip = i // 2
    tensor, actions = torch.rot90(tensor, rotation, (-1,-2)), torch.rot90(actions, -rotation, (-1,-2))
    if flip:
        tensor, actions = torch.flip(tensor, (-1,)), torch.flip(actions, (-1,))
    tensor, actions = tuple(tensor.reshape(-1).tolist()), actions.reshape(-1).tolist()
    return tensor, actions

game_symmetry_info = {
    'game_symmetries': go_symmetries,
    'board_size': 4,
    'num_game_symmetries': 8}

game_symmetry_info = {
    'go':{
        'symmetry_function': go_symmetries,
        'num_game_symmetries': 8
    },
    'hex':{
        'symmetry_function': hex_symmetries,
        'num_game_symmetries': 4
    }
}

# @profile
def main(rl_config, game_params):

    network = None
    game_name, game_config = game_params
    game = pyspiel.load_game(*game_params)
    observation_tensor_shape = game.observation_tensor_shape()
    num_distinct_actions = game.num_distinct_actions()
    print(observation_tensor_shape)
    print(num_distinct_actions)
    buffer = Buffer(rl_config['buffer_test_capacity'], rl_config['buffer_train_capacity'], observation_tensor_shape, game.num_distinct_actions())

    state_data_dict = {}
    network_preds = {}
    num_games = 0
    average_depth = 1
    average_entropy = 1
    num_transitions_observed = 0
    
    print('start')
    log_branching_factors = np.zeros(game.max_game_length())
    evidences = np.zeros(game.max_game_length())
    while True:

        state = game.new_initial_state()
        game_data = []
        
        state_string = state.information_state_string()
        if state_string in state_data_dict and state_data_dict[state_string]['solution'] != None:
            solution_winner = {0:'black', 1:'white'}[state_data_dict[state_string]['solution']]
            print(state_data_dict[state_string]['children_solutions'])
            print(f'game solved for {solution_winner}')
            break
        num_games +=1
        while True:
            state_string = state.information_state_string()
            if rl_config['symmetrized_observation_tensors']:
                reshaped_tensor = torch.tensor(state.observation_tensor()).view(-1,game_config['board_size'], game_config['board_size'])
                found_symmetry = False
                for i in range(game_symmetry_info[game_name]['num_game_symmetries']):
                    tensor_symmetry, action_symmetry = game_symmetry_info[game_name]['symmetry_function'](reshaped_tensor, i)
                    if tensor_symmetry in buffer.hash_map['Test']:
                        observation_tensor, action_mapping = tensor_symmetry, action_symmetry
                        found_symmetry = True
                if not found_symmetry:
                    observation_tensor, action_mapping = tuple(state.observation_tensor()), range(num_distinct_actions)
            else:
                observation_tensor, action_mapping = tuple(state.observation_tensor()), range(num_distinct_actions)
            
            if state_string not in state_data_dict:
                num_transitions_observed +=1
                legal_actions = state.legal_actions()

                state_data_dict[state_string] = {
                    'legal_actions': legal_actions,
                    'observation_tensor': observation_tensor,
                    'action_mapping': action_mapping,
                    'solution': None,
                    'parents': [],
                    'children_solutions': [None]*len(legal_actions)
                }

            state_data = state_data_dict[state_string]
            if state_data['solution'] !=None:
                break
            
            legal_actions = state_data['legal_actions']
            children_solutions = state_data['children_solutions']
            action_options = [legal_actions[i] for i in range(len(legal_actions)) if children_solutions[i]==None]
            
            observation_tensor = state_data['observation_tensor']
            action_mapping = state_data['action_mapping']
            transformed_action_options = [action_mapping[x] for x in action_options]
            # outcomes = buffer.get_observation_outcomes(observation_tensor, transformed_action_options)
            full_conditioning_outcomes = buffer.get_observation_outcomes(observation_tensor, range(num_distinct_actions))
            outcomes = full_conditioning_outcomes[transformed_action_options]
            if network != None:
                if rl_config['input_conditioning']:
                    network_preds[observation_tensor] = network((torch.tensor(observation_tensor).view(1,*observation_tensor_shape), full_conditioning_outcomes.unsqueeze(0)))['alpha'].squeeze(0)
                elif observation_tensor not in network_preds:
                    network_preds[observation_tensor] = network(torch.tensor(observation_tensor).view(1,*observation_tensor_shape))['alpha'].squeeze(0)
            else:
                network_preds[observation_tensor] = torch.full((num_distinct_actions,2), .5)
            state_preds = network_preds[observation_tensor][action_options]
            posteriors = state_preds
            if not rl_config['input_conditioning']:
                posteriors += outcomes
            p = posteriors.sum(axis=-1)/posteriors.sum()
            # if len(game_data)==0:
            #     print(p)
            #     print(posteriors)
                
            
            entropy = -p.dot(torch.log2(p)).item()
            average_entropy= .999 * average_entropy + .001 * entropy
            log_branching_factors[len(game_data)] = log_branching_factors[len(game_data)]*.99 + entropy*.01
            evidences[len(game_data)] = evidences[len(game_data)]*.99 + posteriors.sum() * .01
            current_player = state.current_player()
            if rl_config['index_function'] == 'thompson':
                indices = pyro.distributions.Dirichlet(posteriors).sample()
                action = action_options[indices[:, current_player].argmax()]
            elif rl_config['index_function'] == 'MOSSE':
                indices = MOSSEindices(posteriors, current_player)
                a = (indices==indices.max()).nonzero()
                action = action_options[a[np.random.randint(len(a))]]
                # action = action_options[a[-1]]
            elif rl_config['index_function'] == 'mean':
                indices = posteriors[:, current_player]/posteriors.sum(-1)
                action = action_options[indices.argmax()]
            elif rl_config['index_function'] == 'beta_cdf':
                indices = torch.tensor(scipy.stats.beta.sf(.5, *posteriors.transpose(-1,-2)) if current_player == 0 else scipy.stats.beta.cdf(.5, *posteriors.transpose(-1,-2)))
                a = (indices==indices.max()).nonzero()
                action = action_options[a[np.random.randint(len(a))]]
            
            game_data.append((state_string, observation_tensor, action, current_player))
            
            state.apply_action(action)
            if state.is_terminal():
                num_transitions_observed +=1
                break
        average_depth = .999*average_depth + .001 * len(game_data)
        
        print('game number: ', num_games)
        print(buffer.size())
        print(f'transitions: {num_transitions_observed}')
        print(f'average depth: {average_depth}')
        print('estimated entropy:', average_depth*average_entropy)
        print('\t'.join([str(round(2**x,3)) for x in log_branching_factors]))
        print('\t'.join([str(round(x)) for x in evidences]))
        
        winner = current_player if state.rewards()[current_player]>0 else 1 - current_player
        last_node_solution = winner
        for state_string, observation_tensor, action, current_player in reversed(game_data):
            if (observation_tensor, state_data_dict[state_string]['action_mapping'][action]) not in state_data['parents']:
                state_data['parents'].append((observation_tensor, action))
            state_data = state_data_dict[state_string]
            action_mapping = state_data['action_mapping']
            buffer.add_data(observation_tensor, action_mapping[action], winner)
            
            children_solutions = state_data['children_solutions']
            children_solutions[state_data['legal_actions'].index(action)] = last_node_solution
            if last_node_solution == current_player:
                state_data['solution'] = last_node_solution
            else:
                opponent = 1-current_player
                if any(x != opponent for x in children_solutions):
                    last_node_solution = None
                else:
                    state_data['solution'] = opponent
        if not rl_config['tabular']:
            if rl_config['create_dataset']:
                if len(buffer.test_list) > rl_config['buffer_train_capacity'] + rl_config['buffer_test_capacity']:
                    dataset = buffer.get_dataset('cpu')
                    dataset_name = f'{game_name}/{len(buffer.train_list)}-{len(buffer.test_list)}_{game_config["board_size"]}x{game_config["board_size"]}'
                    torch.save(dataset, f'datasets/{dataset_name}.p')
                    print('train size: ', len(buffer.train_list))
                    print('test size: ', len(buffer.test_list))
                    break
            elif len(buffer.test_list)>buffer.test_capacity*(1+rl_config['buffer_refresh_proportion']):
                dataset = buffer.get_dataset('mps')
                network = train(network, dataset, f'{game_config["board_size"]}x{game_config["board_size"]}_{game_name}_tests', using_wandb=False).to('cpu')
                network_preds = {}
    print('done')
    print(f'number of transitions observed: {num_transitions_observed}')
    print(f'number of games played: {num_games}')
    return num_transitions_observed

if __name__ == "__main__":
    main(rl_config, game_params)




