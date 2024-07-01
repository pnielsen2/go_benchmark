hex_sweep_config = {
    'method': 'bayes',  # or 'grid', 'bayes'
    'metric': {
        'name': 'Best_Loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'batch_size': {
            'values': [10000, 20000, 50000, 100000]
        },
        'network_width': {
            'values': [125, 150, 175, 200, 300]
        },
        'network_depth': {
            'values': [4, 5, 6, 7, 8]
        },
        'hidden_activation': {
            'values': ['selu', 'tanh', 'elu']
        },
        'architecture': {
            'value': 'FC'
        },
        'input_conditioning': {
            'value': False
        },
        'output_activation': {
            'values': ['exp_activation']
        },
        'training_loss': {
            'values': ['dirichlet_multinomial_loss']
        },
        'test_loss': {
            'value': 'dirichlet_multinomial_loss'
        },
        'learning_rate': {
            'values': [5e-4, 1e-3, 2e-3, 3e-3]
        },
        'sampler': {
            'values': ['sequential', 'random']
        },
        'optimizer': {
            'values': ['AdamW']
        },
        'data_augmentation': {
            'values': ['None']
        },
        'training_time': {
            'value': 40
        },
    }
}