import wandb
import random
from sweep_configs import hex_sweep_config
from training_infra.train import train
from training_infra.dataset_loaders import dataset_loaders_dict

# def train():
#     # Initialize wandb
#     run = wandb.init()

#     # Access hyperparameters
#     config = wandb.config

#     # Simulated training loop
#     for epoch in range(10):
#         acc = random.random() * config.lr * config.batch_size
#         loss = 1 - acc
        
#         # Log metrics to wandb
#         wandb.log({"accuracy": acc, "loss": loss})

# # Define the sweep configuration
print('loading dataset')
dataset = dataset_loaders_dict['4x4_hex']('mps')

# Initialize the sweep
sweep_id = wandb.sweep(hex_sweep_config, project="4x4_hex")

f = lambda: train(None, dataset, '4x4_hex', using_wandb=True)
# Start the sweep
# sweep_id = '4x4_hex/zierlpct'
wandb.agent(sweep_id, function=f, count=100000)