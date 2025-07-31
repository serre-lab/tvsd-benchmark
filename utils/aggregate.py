import os
import pandas as pd

from utils.utils import get_region

def aggregate_results(results_dir: str):
    """
    Aggregate results from a directory of results, containing files as:
        - monkey[X]_arr_[Y].csv
        - ...
    """
    results = []
    for file in os.listdir(results_dir):
        if file.endswith(".csv") and "agg" not in file:
            monkey_name = file.split("_")[0]
            arr_number = int(file.split("_")[2].split(".")[0])
            df = pd.read_csv(os.path.join(results_dir, file))
            best_layer = df.loc[df['Score'].idxmax()]
            best_layer_name = best_layer['Layer']
            best_layer_score = best_layer['Score']
            best_layer_std = best_layer['Std']
            region = get_region(monkey_name, arr_number)
            results.append({
                "monkey_name": monkey_name,
                "arr_number": arr_number,
                "region": region,
                "best_layer": best_layer_name,
                "best_layer_score": best_layer_score,
                "best_layer_std": best_layer_std,
            })
    return results


if __name__ == "__main__":
    results_dir = "/users/jamullik/scratch/tvsd-copy/outputs/results/chresmax_v3"
    results = aggregate_results(results_dir)
    results = pd.DataFrame(results)
    results = results.sort_values(by="arr_number", ascending=False)
    results.to_csv(os.path.join(results_dir, "agg_results.csv"), index=False)
