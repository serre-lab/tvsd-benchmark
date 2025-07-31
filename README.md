# TVSD Benchmark

This repo contains tools for loading and benchmarking models on the TVSD (THINGS Ventral Stream Spiking Dataset) from Papale et. al. 2025. 

## Setup

Begin by cloning the repository.
```bash
git clone git@github.com:serre-lab/tvsd-benchmark.git
cd tvsd-benchmark
```
Next, create a `conda` environment with our requirements.
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

Ensure that you have your virtual envirovnment activated, and run
```bash
sbatch scripts/generate_activations.sh [MODEL_CONFIG_PATH]
```
When this completes, run
```bash
sbatch scripts/benchmark.sh [MODEL_CONFIG_PATH]
```
(We separate the two jobs, as only the former requires a GPU.) The results will populate `outputs/results/[model]`.

## Benchmarking a Suite of Models

Fill `configs/models.csv` with the names of the models you want to benchmark. Then run
```bash
sbatch all_models.sh
```
Which will generative and evaluate activations for each model.

## Adding Your Own Model

In the current configuration, each model is specified by a corresponding config file in `configs`. Making a new config for your model is self-explanatory--just follow the outline of the existing ones. You will also have to build out `utils/load_model.py` to accept your added model. In the future, direct integration with `timm` will be provided. 
