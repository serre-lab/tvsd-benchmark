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
import json
import h5py
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Calculate correlations between model activations and neural data.")
parser.add_argument('--array', type=int)
parser.add_argument('--monkey', type=str, default='monkeyF',
                    help='Monkey name to use for the analysis. Default is monkeyF.')
parser.add_argument('--layer', type=str, default='layer3',
                    help='Layer of the model to use for activations. Default is layer3.')
parser.add_argument('--output_file', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_activations = torch.load(f'data/TVSD/{args.monkey}/activations/{args.layer}_train_activations.pt')
print("Loaded model activations with shape:", model_activations.shape)
model_activations = model_activations.reshape(model_activations.shape[0], -1).to(device)  # [N, C*H*W]
print("Reshaped model activations to:", model_activations.shape)

test_activations = torch.load(f'data/TVSD/{args.monkey}/activations/{args.layer}_test_activations.pt')
print("Loaded test activations with shape:", test_activations.shape)
test_activations = test_activations.reshape(test_activations.shape[0], -1).to(device)  # [N', C*H*W]
print("Reshaped test activations to:", test_activations.shape)

with h5py.File(f'data/TVSD/{args.monkey}/THINGS_normMUA.mat', 'r') as f:
    train_set = f["train_MUA"][()]
    train_reliability = torch.mean(
        torch.tensor(f["reliab"][()], dtype=torch.float32), dim=0
    )

with h5py.File(f'data/TVSD/{args.monkey}/THINGS_normMUA.mat', 'r') as f:
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

# --- Ridge regression for all neurons at once ---
lambda_reg = 1e-3  # Regularization strength, can be tuned
D = model_activations.shape[1]
XtX = torch.matmul(model_activations.T, model_activations)
reg = lambda_reg * torch.eye(D, device=model_activations.device)
XtX_reg = XtX + reg
Xty = torch.matmul(model_activations.T, arr_set)  # [D, 64]
W = torch.linalg.solve(XtX_reg, Xty)              # [D, 64]

# Predict on test set
Y_pred = torch.matmul(test_activations, W)        # [N', 64]

# Compute correlation for each neuron (vectorized)
a = Y_pred - Y_pred.mean(dim=0, keepdim=True)
b = test_arr_set - test_arr_set.mean(dim=0, keepdim=True)
corr = (a * b).sum(dim=0) / (torch.norm(a, dim=0) * torch.norm(b, dim=0))  # [64]
neuron_correlations = corr.cpu().numpy().tolist()
for i, c in enumerate(neuron_correlations):
    print(f"Array {arr}, Neuron {i} correlation: {c}")

# After the loop, convert to numpy and print summary stats
neuron_correlations = np.array(neuron_correlations)
mean_corr = np.mean(neuron_correlations)
median_corr = np.median(neuron_correlations)
print(f"Array {arr} mean correlation: {mean_corr}, median correlation: {median_corr}")

# Save the results to json
output_dict = {
    'array': arr,
    'layer': args.layer,
    'monkey': args.monkey,
    'mean_correlation': mean_corr,
    'median_correlation': median_corr,
    'neuron_correlations': neuron_correlations.tolist()
}
with open(args.output_file, 'w') as f:
    json.dump(output_dict, f, indent=4)
