model_dir = '/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist'
model_name = "ip_18_chresmax_v3_2_abs_gpu_2_cl_0.1_ip_3_224_224_15616_c1\[_6\,3\,1_\]_bypass_2"
model_path = f"{model_dir}/{model_name}/"

import torch

model = torch.load(model_path)
