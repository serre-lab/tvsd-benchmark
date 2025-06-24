"""
pseudo-code

N_neurons = 1024

neural_activations # [N, N_neurons]
model_activations # [N, C, H, W]
flat_model_activations = model_activations.reshape(N, C * H * W) # [N, C*H*W]

for i in range(N_neurons):
    neuron_activations = neural_activations[:, i]  # [N]
    # get correlation of flat_model_activations with neuron_activations

"""

import h5py
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Calculate correlations between model activations and neural data.")
parser.add_argument('--array', type=int)
parser.add_argument('--layer', type=str, default='layer3',
                    help='Layer of the model to use for activations. Default is layer3.')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_activations = torch.load('/users/jamullik/scratch/TVSD-real/monkeyF/activations/layer3_train_activations.pt')
print("Loaded model activations with shape:", model_activations.shape)
model_activations = model_activations.reshape(model_activations.shape[0], -1).to(device)  # [N, C*H*W]
print("Reshaped model activations to:", model_activations.shape)

test_activations = torch.load('/users/jamullik/scratch/TVSD-real/monkeyF/activations/layer3_test_activations.pt')
print("Loaded test activations with shape:", test_activations.shape)
test_activations = test_activations.reshape(test_activations.shape[0], -1).to(device)  # [N', C*H*W]
print("Reshaped test activations to:", test_activations.shape)

with h5py.File('/users/jamullik/scratch/TVSD-real/monkeyF/THINGS_normMUA.mat', 'r') as f:
    train_set = f["train_MUA"][()]
    train_reliability = torch.mean(
        torch.tensor(f["reliab"][()], dtype=torch.float32), dim=0
    )

with h5py.File('/users/jamullik/scratch/TVSD-real/monkeyF/THINGS_normMUA.mat', 'r') as f:
    test_set = f["test_MUA"][()]
    test_reliability = torch.mean(
        torch.tensor(f["reliab"][()], dtype=torch.float32), dim=0
    )

arr = args.array

print(f"Processing array {arr}...")
arr_set = train_set[:, arr * 64 : (arr + 1) * 64] # [N, 64]
arr_set = torch.tensor(arr_set, dtype=torch.float32).to(device)

test_arr_set = test_set[:, arr * 64 : (arr + 1) * 64] # [N', 64]
test_arr_set = torch.tensor(test_arr_set, dtype=torch.float32).to(device)
neuron_correlations = []
for i in range(arr_set.shape[1]):
    neuron_activations = arr_set[:, i]
    # Calculate correlation with model activations
    XtX = torch.matmul(model_activations.T, model_activations)
    Xty = torch.matmul(model_activations.T, neuron_activations)
    W = torch.linalg.solve(XtX, Xty)
    
    # Calculate correlation on test set
    Y_pred = torch.matmul(test_activations, W)
    test_neuron = test_arr_set[:, i]
    a = Y_pred - Y_pred.mean()
    b = test_neuron - test_neuron.mean()
    corr = (a * b).sum() / (torch.norm(a) * torch.norm(b))
    neuron_correlations.append(corr.item())
    print(f"Array {arr}, Neuron {i} correlation: {corr.item()}")

# After the loop, convert to numpy and print summary stats
neuron_correlations = np.array(neuron_correlations)
mean_corr = np.mean(neuron_correlations)
median_corr = np.median(neuron_correlations)
print(f"Array {arr} mean correlation: {mean_corr}, median correlation: {median_corr}")
