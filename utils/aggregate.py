import os
import pandas as pd

from utils.utils import get_region


def aggregate_results(results_dir: str, monkey: str = "monkeyF"):
    """
    Our methodology is as follows:
    1. For each layer, calculate the mean performance on each region
    2. For each region, calculate the best layer
    3. This best layer is the one that we will use to calculate the performance of the model on the region
    """
    regions = ["V1", "V4", "IT"]
    for region in regions:
        model_best_layers = {}
        for model in os.listdir(results_dir):
            model_dir = os.path.join(results_dir, model)
            if not os.path.isdir(model_dir):
                continue
            results = collect_results(model_dir, monkey)
            if region not in results or not results[region]:
                continue
            region_df = pd.concat(results[region], ignore_index=True)
            layers = region_df["Layer"].unique()
            layer_scores = {}
            for layer in layers:
                layer_results = region_df[region_df["Layer"] == layer]
                layer_scores[layer] = {
                    "model": model,
                    "layer": layer,
                    "mean_score": layer_results["Score"].mean(),
                    "std_score": layer_results["Score"].std(),
                }
            if layer_scores:
                best_layer_info = max(
                    layer_scores.values(), key=lambda x: x["mean_score"]
                )
                model_best_layers[model] = {
                    "best_layer": best_layer_info["layer"],
                    "best_layer_score": best_layer_info["mean_score"],
                    "best_layer_std": best_layer_info["std_score"],
                }
        if model_best_layers:
            df_data = []
            for model, info in model_best_layers.items():
                df_data.append(
                    {
                        "model": model,
                        "best_layer": info["best_layer"],
                        "best_layer_score": info["best_layer_score"],
                        "best_layer_std": info["best_layer_std"],
                    }
                )
            model_best_layers_df = pd.DataFrame(df_data)
            model_best_layers_df.to_csv(
                os.path.join(results_dir, f"agg_results_{region}.csv"), index=False
            )


def collect_results(model_dir: str, monkey: str = "monkeyF"):
    """
    Collect results from a directory of results, containing files as:
        - monkey[X]_arr_[Y].csv
        - ...
    """
    results = {}
    for file in os.listdir(model_dir):
        if file.endswith(".csv") and "agg" not in file:
            monkey_name = file.split("_")[0]
            arr_number = int(file.split("_")[2].split(".")[0])
            if monkey_name != monkey:
                continue
            arr_number = int(file.split("_")[2].split(".")[0])
            df = pd.read_csv(os.path.join(model_dir, file))
            region = get_region(monkey, arr_number)
            if region not in results:
                results[region] = []
            results[region].append(df)
    return results


if __name__ == "__main__":
    results_dir = "/users/jamullik/scratch/tvsd-copy/outputs/results/"
    aggregate_results(results_dir)
