# TVSD Benchmark

This repo contains tools for loading and benchmarking models on the TVSD (THINGS Ventral Stream Spiking Dataset) from Papale et. al. 2025. 

## Setup

Begin by cloning the repository.
```bash
git clone git@github.com:serre-lab/tvsd-benchmark.git
cd tvsd-benchmark
```

### Option 1: Docker + Make (recommended)
Build the container image and run the unit tests:
```bash
make build
make test
```
Open a shell in the container:
```bash
make shell
```

### Option 2: Local Python environment
Create a `conda` environment with our requirements.
```bash
conda create -n tvsd-benchmark
conda activate tvsd-benchmark
pip install -r requirements.txt
```

For timm model support, ensure timm is installed:
```bash
pip install timm
```

Alternatively, you can use a `venv` environment.
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install timm
```
To obtain the TVSD dataset, run
```bash
chmod +x scripts/download_tvsd.sh
./scripts/download_tvsd.sh
```
Which will download the normalized MUA and metadata `.mat` files into a new `data` directory. To obtain the THINGS dataset, you should analogously run the following snippet. You will be prompted by `osfclient` to provide a password in order to unzip the dataset. You can easily obtain this password [here](https://osf.io/j6a3m).
```bash
chmod +x scripts/download_things.sh
./scripts/download_things.sh
```

## Benchmarking a Model

Ensure that you have your environment activated, and run
```bash
sbatch scripts/generate_activations.sh [MODEL_CONFIG_PATH]
```
When this completes, run
```bash
sbatch scripts/benchmark.sh [MODEL_CONFIG_PATH]
```
(We separate the two jobs, as only the former requires a GPU.) The results will populate `outputs/results/[model]`.

To run unit tests locally without Docker:
```bash
make test-local
```

## Benchmarking a Suite of Models

Fill `configs/models.csv` with the names of the models you want to benchmark. Then run
```bash
sbatch scripts/all_models.sh
```
Which will generative and evaluate activations for each model.

## Adding Your Own Model

In the current configuration, each model is specified by a corresponding config file in `configs`. Making a new config for your model is self-explanatory--just follow the outline of the existing ones. You will also have to build out `utils/load_model.py` to accept your added model.

## Using Timm Models

This repository now has full support for models from the [timm](https://github.com/huggingface/pytorch-image-models) library. To benchmark a timm model:

1. **Create a config file** (or use one of the provided examples in `configs/examples/`):

```yaml
model-name: resnet50  # Any timm model name
model-source: timm
pretrained: true  # Set to false to use random weights
hook-interval: 8  # Interval for activation extraction
transform:
  - name: Resize
    size: [224, 224]
  - name: ToTensor
  - name: Normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

2. **Run benchmarking** as usual:

```bash
sbatch scripts/generate_activations.sh configs/examples/resnet50.yaml
sbatch scripts/benchmark.sh configs/examples/resnet50.yaml
```

### Supported Model Families

The following timm model families are fully supported with specialized activation extraction:

- **ResNet** (`resnet50`, `resnet101`, etc.) - Standard CNN architecture
- **Vision Transformers** (`vit_base_patch16_224`, `vit_tiny_patch16_224`, etc.) - Handles token/cls shapes
- **ConvNeXt** (`convnext_base`, `convnext_tiny`, etc.) - Modern CNN with stages
- **Swin Transformer** (`swin_base_patch4_window7_224`, `swin_tiny_patch4_window7_224`, etc.) - Hierarchical vision transformers

Example configs for these models are provided in `configs/examples/`.

### Note on Model Names

You can list all available timm models using:
```python
import timm
print(timm.list_models(pretrained=True))
``` 
