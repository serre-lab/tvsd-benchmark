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
Alternatively, you can use a `venv` environment.
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
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

## Benchmarking timm Models

Any pretrained [`timm`](https://github.com/huggingface/pytorch-image-models) model is supported out of the box. To generate a config for every pretrained timm model, run
```bash
python -m utils.populate_timm
```
which writes one config per model into `configs/timm/`. Each config uses `transform: timm`, which builds the model's *native* evaluation transform (correct input size, crop, interpolation, and normalization) from its pretrained config at load time--so no transform needs to be hand-written per model.

To benchmark a single timm model, either point the pipeline at a generated config or write a minimal one yourself:
```yaml
model-name: vit_base_patch16_384
model-type: timm
model-source: timm
hook-interval: 5
transform: timm
```

## Adding Your Own Model

In the current configuration, each model is specified by a corresponding config file in `configs`. Making a new config for your model is self-explanatory--just follow the outline of the existing ones. For non-timm/torchvision models you will also have to build out `utils/load_model.py` to accept your added model.

### HMAX / chresmax models

The `chresmax_v3` (HMAX) model is defined in the Serre Lab [fork of `pytorch-image-models`](https://github.com/npant14/pytorch-image-models) (as `timm.models.RESMAX`) and is **not** part of the upstream `timm` package. To benchmark it, install that fork in place of the pip `timm` package:
```bash
pip install -e /path/to/serre-lab/pytorch-image-models
```
The import is lazy, so the rest of the pipeline (including all standard timm models) works fine without it.
