# For a list of logits, there are 2 sampling methods
# 1. argmax: always picks the most probable logit
# 2. torch.multinomial: samples from a probability distribution, meaning even low-probability logits have a chance of being selected

import torch

# Define a probability distribution over 5 tokens
probs = torch.tensor([0.1, 0.2, 0.3, 0.25, 0.15]) # probabilities sum to 1

# Using argmax (always picks the highest probability logit)
argmax_choice = torch.argmax(probs).item()

# Using multinomial (samples randomly based on probabilities)
multinomial_choice = torch.multinomial(probs, num_samples=1, replacement=True)

print(f"argmax choice: {argmax_choice}")
print(f"Multinomial choice: {multinomial_choice}")

# Running multiple multinomial samples
samples = torch.multinomial(probs, num_samples=10, replacement=True)
print("Multinomial samples:", samples.tolist())

# Effedct of tempearture
# High temperature -> more uniform sampling
# Lower temperature -> more deterministic

temperature = 0.9
scaled_probs = torch.softmax(probs / temperature, dim=0)

samples = torch.multinomial(scaled_probs, num_samples=10, replacement=True)
print("Multinomial samples with temperature 0.5:", samples.tolist())