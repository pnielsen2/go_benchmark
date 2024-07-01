import torch
import memory_profiler
from bitarray import bitarray, frozenbitarray
from bitarray.util import ba2int
import math
from collections import defaultdict


class Buffer:
    def __init__(self, test_capacity, train_capacity, observation_tensor_shape, num_distinct_actions):
        self.test_capacity = test_capacity
        self.train_capacity = train_capacity
        self.total_capacity = self.test_capacity + self.train_capacity
        self.observation_tensor_shape = observation_tensor_shape
        self.num_distinct_actions = num_distinct_actions
        self.num_distinct_outcomes = 2*num_distinct_actions
        self.test_list = []
        self.train_list = []
        
        self.dataset = {'Train': {'x':[], 'y': []}, 'Test': {'x':[], 'y': []}}
        self.hash_map = {'Train': {}, 'Test': {}}
        self.state_data = {}
    
    
    def size(self):
        return {'Train': len(self.train_list), 'Test': len(self.test_list)}

    # @profile
    def _increment_dataset(self, observation_tensor, action, winner, increment, split):
        if observation_tensor not in self.hash_map[split]:
            index = len(self.dataset[split]['y']) // (self.num_distinct_outcomes)
            self.hash_map[split][observation_tensor] = {'index': index, 'count': 0}
            
            self.dataset[split]['y'].extend([0]*self.num_distinct_outcomes)
            self.dataset[split]['x'].append(observation_tensor)
        else:
            index = self.hash_map[split][observation_tensor]['index']
            
        self.dataset[split]['y'][index*self.num_distinct_outcomes + 2*action + winner] += increment
        self.hash_map[split][observation_tensor]['count'] += increment
    
    # @profile
    def add_data(self, observation_tensor, action, winner):
        self.test_list.append((observation_tensor, action, winner))
        self._increment_dataset(observation_tensor, action, winner, 1, 'Test')


    def get_observation_outcomes(self, observation_tensor, actions):
        if observation_tensor in self.hash_map['Test']:
            index = self.hash_map['Test'][observation_tensor]['index']
            y = self.dataset['Test']['y']
            return torch.tensor([[y[index*self.num_distinct_outcomes + 2*action], y[index*self.num_distinct_outcomes + 2*action+1]] for action in actions])
        else:
            return torch.zeros(len(actions), 2)

    # @profile
    def _remove_empty_data(self):
        for split in ['Test', 'Train']:
            split_map = self.hash_map[split]
            x, y = self.dataset[split]['x'], self.dataset[split]['y']
            new_x, new_y = [], []
            index = 0
            for i in range(len(x)):
                observation_tensor = x[i]
                if split_map[observation_tensor]['count'] > 0:
                    new_x.append(observation_tensor)
                    new_y.extend(y[i*self.num_distinct_outcomes:(i+1)*self.num_distinct_outcomes])
                    self.hash_map[split][observation_tensor]['index'] = index
                    index +=1
                else:
                    del self.hash_map[split][observation_tensor]
            self.dataset[split]['x'], self.dataset[split]['y'] = new_x, new_y
    
    # @profile
    def get_dataset(self, device):
        print('creating dataset')
        test_excess = self.test_list[:-self.test_capacity]
        self.test_list = self.test_list[-self.test_capacity:]
        
        self.train_list = self.train_list + test_excess
        
        train_excess = self.train_list[:-self.train_capacity]
        self.train_list = self.train_list[-self.train_capacity:]
        
        for observation_tensor, action, winner in test_excess:
            self._increment_dataset(observation_tensor, action, winner, -1, 'Test')
            self._increment_dataset(observation_tensor, action, winner, 1, 'Train')
        
        for observation_tensor, action, winner in train_excess:
            self._increment_dataset(observation_tensor, action, winner, -1, 'Train')

        self._remove_empty_data()
        print('returning dataset')
        return {
            split: {
                'x': torch.tensor(dataset['x'], device=device).view(-1,*self.observation_tensor_shape),
                'y': torch.tensor(dataset['y'], device=device).view(-1,self.num_distinct_actions,2)} for split, dataset in self.dataset.items()}
        

def zero():
    return 0

def empty_list():
    return []

class DictBuffer:
    def __init__(self, test_capacity, train_capacity, board_dim=3):
        self.test_capacity = test_capacity
        self.train_capacity = train_capacity
        self.board_dim = board_dim
        self.num_possible_moves = board_dim **2 + 1
        self.test_list = []
        self.train_list = []
        
        self.dataset = {'Train': {'x':[], 'y': []}, 'Test': {'x':[], 'y': []}}
        self.hash_map = {'Train': {}, 'Test': {}}
        self.state_data = {}
        self.y_dicts = {'Train':defaultdict(zero), 'Test':defaultdict(zero)}
        self.y_nonzeros = {'Train':defaultdict(empty_list), 'Test':defaultdict(empty_list)}
        self.num_y_examples = 0
    
    
    def size(self):
        return {'Train': len(self.train_list), 'Test': len(self.test_list)}

    # @profile
    def _increment_dataset(self, observation_tensor, observation_hash, action, winner, increment, split):
        if observation_hash not in self.hash_map[split]:
            self.num_y_examples +=1
            dict_index = self.num_y_examples

            self.hash_map[split][observation_hash] = {'index': index, 'count': 0}
            
            self.dataset[split]['x'].append(observation_tensor)
        self.y_dicts[split][(observation_hash, action, winner)] += increment
        self.hash_map[split][observation_hash]['count'] += increment
    
    # @profile
    def add_data(self, observation_tensor, observation_hash, action, winner):
        self.test_list.append((observation_hash, action, winner))
        self.y_dicts['Test'][(observation_hash, action, winner)] += 1
        self.y_nonzeros['Test'][observation_hash].append((action, winner))

    # @profile
    def get_observation_outcomes(self, observation_hash, actions):
        test_y_dict = self.y_dicts['Test']
        outcomes = [[0,0] for _ in range(self.num_possible_moves)]
        for action, winner in self.y_nonzeros['Test'][observation_hash]:
            outcomes[action][winner] = test_y_dict[(observation_hash, action, winner)]
        return torch.tensor([outcomes[action] for action in actions])

    # @profile
    def _remove_empty_data(self):
        for split in ['Test', 'Train']:
            split_map = self.hash_map[split]
            x, y = self.dataset[split]['x'], self.dataset[split]['y']
            new_x, new_y = [], []
            index = 0
            for i in range(len(x)):
                observation_tensor = x[i]
                if split_map[observation_tensor]['count'] > 0:
                    new_x.append(observation_tensor)
                    new_y.append(y[i])
                    self.hash_map[split][observation_tensor]['index'] = index
                    index +=1
                else:
                    del self.hash_map[split][x[i]]
            self.dataset[split]['x'], self.dataset[split]['y'] = new_x, new_y
    
    # @profile
    def get_dataset(self, device):
        test_excess = self.test_list[:-self.test_capacity]
        self.test_list = self.test_list[-self.test_capacity:]
        
        self.train_list.extend(test_excess)
        
        train_excess = self.train_list[:-self.train_capacity]
        self.train_list = self.train_list[-self.train_capacity:]
        
        for observation_tensor, action, winner in test_excess:
            self._increment_dataset(observation_tensor, action, winner, -1, 'Test')
            self._increment_dataset(observation_tensor, action, winner, 1, 'Train')
        
        for observation_tensor, action, winner in train_excess:
            self._increment_dataset(observation_tensor, action, winner, -1, 'Train')

        self._remove_empty_data()
        
        return {
            split: {
                'x': torch.tensor(dataset['x'], device=device).view(-1,4,self.board_dim,self.board_dim),
                'y': torch.tensor(dataset['y'], device=device).view(-1,self.num_possible_moves,2)} for split, dataset in self.dataset.items()}
    
    



class MemoryEfficientBuffer:
    def __init__(self, test_capacity, train_capacity, board_dim=3):
        self.test_capacity, self.train_capacity = test_capacity, train_capacity
        self.total_capacity = self.test_capacity + self.train_capacity 
        self.board_dim = board_dim
        self.num_possible_moves = self.board_dim**2+1
        self.observation_tensors, self.actions, self.winners = bitarray(), bitarray(), bitarray()
        
        self.bits_per_action = int(math.ceil(math.log2(self.num_possible_moves)))
        self.action_bit_dict = {i: bitarray(f'{i:0{self.bits_per_action}b}') for i in range(self.num_possible_moves)}
        self.bit_action_dict = {frozenbitarray(f'{i:0{self.bits_per_action}b}'):i for i in range(self.num_possible_moves)}
        
        self.observation_size = 4*self.board_dim**2
        
        self.test_start = 0
        
        self.hash_map = {}
    
    def add_data(self, observation_tensor, action, winner):
        # self.observation_tensors.extend([int(i) for i in sum(observation_tensors,[])])
        # self.actions.encode(self.action_bit_dict, actions)
        # self.winners.extend([winner]*len(actions))
        int_tensor = [int(i) for i in observation_tensor]
        key = frozenbitarray(int_tensor)
        if key not in self.hash_map:
            self.hash_map[key] = []
        self.hash_map[key].append(len(self.winners))
        self.observation_tensors.extend([int(i) for i in observation_tensor])
        self.actions.extend(self.action_bit_dict[action])
        self.winners.append(winner)

    # @profile
    def get_observation_outcomes(self, observation_tensor, actions):
        observation_tensor = frozenbitarray([int(i) for i in observation_tensor])
        observation_outcomes = [[0]*2 for _ in range(len(actions))]
        if observation_tensor in self.hash_map:
            for index in self.hash_map[observation_tensor]:
                if index >= self.test_start:
                    action_index_start = index*self.bits_per_action
                    action_array = frozenbitarray(self.actions[action_index_start:action_index_start + self.bits_per_action])
                    action = self.bit_action_dict[action_array]
                    if action in actions:
                        outcome_action_index = actions.index(action)
                        winner = self.winners[index]
                        observation_outcomes[outcome_action_index][winner] +=1
        return torch.tensor(observation_outcomes)

    
    def get_dataset(self, device):
        self.observation_tensors, self.actions, self.winners = self.observation_tensors[-self.total_capacity*self.observation_size:], self.actions[-self.total_capacity*self.bits_per_action:], self.winners[-self.total_capacity:]
        buffer_size = len(self.winners)
        self.test_start = max( - self.test_capacity, 0)
        actions = self.actions.decode(self.action_bit_dict)

        train_x = []
        train_y = []
        for i in range(self.test_start):
            observation_tensor = self.observation_tensors[i*self.observation_size:(i+1)*self.observation_size]
            try:
                index = train_x.index(observation_tensor)
            except:
                index = len(train_x)
                train_x.append(observation_tensor)
                train_y.append([[0]*2 for _ in range(self.num_possible_moves)])
            train_y[index][actions[i]][self.winners[i]] +=1
        
        test_x = []
        test_y = []
        for i in range(self.test_start, buffer_size):
            observation_tensor = self.observation_tensors[i*self.observation_size:(i+1)*self.observation_size]
            try:
                index = test_x.index(observation_tensor)
            except:
                index = len(train_x)
                test_x.append(observation_tensor)
                test_y.append([[0]*2 for _ in range(self.num_possible_moves)])
            train_y[index][actions[i]][self.winners[i]] +=1
        
        return {'Train':{'x':torch.tensor(train_x,device=device),'y':torch.tensor(train_x,device=device)}, 'Test':{'x':torch.tensor(test_x,device=device),'y':torch.tensor(test_x,device=device)}}
        
                

    

    # def size(self):
    #     return {'Train': len(self.train_list), 'Test': len(self.test_list)}
    
    # def tabular_outcomes(self, position_hash):
    #     return self.dataset['Test']['y'][self.hash_map['Test'][position_hash]]

    # @profile
    # def _increment_dataset(self, observation_tensor, observation_hash, action, winner, increment, split):
    #     if observation_hash not in self.hash_map[split]:
    #         index = len(self.dataset[split]['y'])
    #         self.hash_map[split][observation_hash] = {'index': index, 'count': 0}
            
    #         self.dataset[split]['y'].append([[0]*2 for _ in range(self.num_possible_moves)])
    #         self.dataset[split]['x'].append(observation_tensor)
    #     else:
    #         index = self.hash_map[split][observation_hash]['index']
            
    #     self.dataset[split]['y'][index][action][winner] += increment
    #     self.hash_map[split][observation_hash]['count'] +=1
    
    # @profile
    # def add_data(self, observation_tensor, observation_hash, action, winner):
    #     self.test_list.append((observation_hash, action, winner))
    #     self._increment_dataset(observation_tensor, observation_hash, action, winner, 1, 'Test')


    # def get_observation_outcomes(self, observation_hash, actions):
    #     if observation_hash in self.hash_map['Test']:
    #         outcomes = self.dataset['Test']['y'][self.hash_map['Test'][observation_hash]['index']]
    #         return torch.tensor([outcomes[action] for action in actions])
    #     else:
    #         return torch.zeros(len(actions), 2)

    # @profile
    # def _remove_empty_data(self):
    #     for split in ['Test', 'Train']:
    #         split_map = self.hash_map[split]
    #         x, y = self.dataset[split]['x'], self.dataset[split]['y']
    #         new_x, new_y = [], []
    #         index = 0
    #         for i in range(len(x)):
    #             observation_tensor = x[i]
    #             if split_map[observation_tensor]['count'] > 0:
    #                 new_x.append(observation_tensor)
    #                 new_y.append(y[i])
    #                 self.hash_map[split][observation_tensor]['index'] = index
    #                 index +=1
    #             else:
    #                 del self.hash_map[split][x[i]]
    #         self.dataset[split]['x'], self.dataset[split]['y'] = new_x, new_y
    
    # @profile
    # def get_dataset(self, device):
    #     test_excess = self.test_list[:-self.test_capacity]
    #     self.test_list = self.test_list[-self.test_capacity:]
        
    #     self.train_list = self.train_list + test_excess
        
    #     train_excess = self.train_list[:-self.train_capacity]
    #     self.train_list = self.train_list[-self.train_capacity:]
        
    #     for observation_tensor, action, winner in test_excess:
    #         self._increment_dataset(observation_tensor, action, winner, -1, 'Test')
    #         self._increment_dataset(observation_tensor, action, winner, 1, 'Train')
        
    #     for observation_tensor, action, winner in train_excess:
    #         self._increment_dataset(observation_tensor, action, winner, -1, 'Train')

    #     self._remove_empty_data()
        
    #     return {
    #         split: {
    #             'x': torch.tensor(dataset['x'], device=device).view(-1,4,self.board_dim,self.board_dim),
    #             'y': torch.tensor(dataset['y'], device=device).view(-1,self.num_possible_moves,2)} for split, dataset in self.dataset.items()}
    

    

    





