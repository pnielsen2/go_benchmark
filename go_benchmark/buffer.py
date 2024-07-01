import torch

class Buffer:
    def __init__(self, test_capacity, train_capacity, board_dim=3):
        self.test_capacity = test_capacity
        self.train_capacity = train_capacity
        self.test_list = []
        self.train_list = []
        
        self.dataset = {
            'Train': {
                'x':torch.zeros((0, 18, board_dim, board_dim)), 
                'y': torch.zeros((0, board_dim **2 + 1, 2), dtype=torch.int64)
                },
            'Test': {
                'x':torch.zeros((0, 18, board_dim, board_dim)), 
                'y': torch.zeros((0, board_dim **2 + 1, 2), dtype=torch.int64)
                }
        }
        
        self.hash_map = {
            'Train': {},
            'Test': {}
        }
    
    def get_dataset(self, device):
        test_excess = self.test_list[:-self.test_capacity]
        self.test_list = self.test_list[-self.test_capacity:]
        
        self.train_list = self.train_list + test_excess
        
        train_excess = self.train_list[:-self.train_capacity]
        self.train_list = self.train_list[-self.train_capacity:]
        
        for position_hash, move_outcome, x in test_excess:
            self._increment_dataset(position_hash, move_outcome, x, -1, 'Test')
            self._increment_dataset(position_hash, move_outcome, x, 1, 'Train')
        
        for position_hash, move_outcome, x in train_excess:
            self._increment_dataset(position_hash, move_outcome, x, -1, 'Train')

        self._remove_empty_data()
        
        return {split: {k: v.to(device) for k, v in dataset.items()} for split, dataset in self.dataset.items()}

    def add_data(self, data):
        self.test_list = self.test_list + data
        for position_hash, move_outcome, x in data:
            self._increment_dataset(position_hash, move_outcome, x, 1, 'Test')
    
    def _remove_empty_data(self):
        for split in ['Test', 'Train']:
            nonempty_indices = ~(self.dataset[split]['y']==0).all(dim=(-1,-2))
            new_indices = torch.cumsum(nonempty_indices, dim=0)-1
            for k, v in list(self.hash_map[split].items()):
                if nonempty_indices[v]:
                    self.hash_map[split][k] = new_indices[v]
                else:
                    del self.hash_map[split][k]
            self.dataset[split]['y'] = self.dataset[split]['y'][nonempty_indices]
            self.dataset[split]['x'] = self.dataset[split]['x'][nonempty_indices]

    def _increment_dataset(self, position_hash, move_outcome, x, increment, split):
        if position_hash not in self.hash_map[split]:
            self.hash_map[split][position_hash] = len(self.dataset[split]['y'])
            self.dataset[split]['y'] = torch.cat((self.dataset[split]['y'], torch.zeros((1, 10, 2), dtype=torch.int64)), dim=0)
            self.dataset[split]['x'] = torch.cat((self.dataset[split]['x'], x), dim=0)
        
        index = self.hash_map[split][position_hash]
        self.dataset[split]['y'][index][move_outcome] += increment

    def size(self):
        return {'Train': len(self.train_list), 'Test': len(self.test_list)}
    
    def tabular_outcomes(self, position_hash):
        return self.dataset['Test']['y'][self.hash_map['Test'][position_hash]]
