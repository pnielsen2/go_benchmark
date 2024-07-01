# game_params = ('go', {'board_size': 3, 'komi':1.5})
game_params = ('hex', {'board_size': 4})

rl_config = {
    'create_dataset': False,
    'tabular': True,
    'buffer_test_capacity' : 150000,
    'buffer_train_capacity' : 150000,
    'index_function': 'MOSSE',
    'buffer_refresh_proportion': .5,
    'symmetrized_observation_tensors': False,
    'input_conditioning': False
    }