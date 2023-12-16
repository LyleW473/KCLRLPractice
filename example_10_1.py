"""
Work relating to the K-Armed bandits example
"""

import torch

my_generator = torch.Generator()
my_generator.manual_seed(2004)

num_samples = 100
mean_A, std_A = torch.tensor(10.0), torch.tensor(1.0)
mean_B, std_B = torch.tensor(11.0), torch.tensor(4.0)

samples_A = torch.normal(mean = mean_A, std = std_A, generator = my_generator, size = (num_samples,))
samples_B = torch.normal(mean = mean_B, std = std_B, generator = my_generator, size = (num_samples,))

print(f"Samples A:\n{samples_A}\nAverage: {torch.mean(samples_A)}\n")
print(f"Samples B:\n{samples_B}\nAverage: {torch.mean(samples_B)}")