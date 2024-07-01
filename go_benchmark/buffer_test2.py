from buffer import Buffer
import torch
import random

test_capacity, train_capacity = 10000, 10000

buffer = Buffer(test_capacity, train_capacity, 3)

data = [(random.randrange(0,10), (random.randrange(0,10), random.randrange(0,2)), torch.randn(1,18,3,3)) for _ in range(30000)]

buffer.add_data(data)
print(buffer.tabular_outcomes(3))