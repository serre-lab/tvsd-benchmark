import os
import pickle
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from typing import List, Dict, Optional


class Activations:
    def __init__(
        self,
        output_dir: str,
        model_name: str,
        dataset_name: str,
        pca_components: Optional[int] = None,
    ):
        self.hooks = []
        self._active = True
        self._current_batch = None
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.activations = {}  # {layer_name: {1: activations, 2: ...}}
        self.output_dir = output_dir
        self.ipca_models: Dict[str, IncrementalPCA] = {}
        self._training_mode = False
        self.pca_components = pca_components
        self._training_activations = {}  # {layer_name: [activations]}

    def set_batch(self, batch):
        self._current_batch = batch

    def set_training_mode(self, training: bool):
        self._training_mode = training

    def _get_hook(self, layer_name, debug=False):
        def hook_fn(module, input, output):
            if self._active:
                if isinstance(output, list) or isinstance(output, tuple):
                    output = torch.stack(output, dim=0)

                if self._training_mode:
                    self._handle_training_mode(layer_name, output)
                else:
                    self._handle_inference_mode(layer_name, output)

        return hook_fn

    def _handle_training_mode(self, layer_name: str, output: torch.Tensor):
        output_flat = output.reshape(output.shape[0], -1).detach().cpu()

        if layer_name not in self._training_activations:
            self._training_activations[layer_name] = []

        self._training_activations[layer_name].append(output_flat.numpy())

    def _handle_inference_mode(self, layer_name: str, output: torch.Tensor):
        output_flat = output.reshape(output.shape[0], -1).detach().cpu()
        if not hasattr(self, "_inference_accum"):
            self._inference_accum = {}
        if layer_name not in self._inference_accum:
            self._inference_accum[layer_name] = []
        self._inference_accum[layer_name].append(output_flat)

    def finalize_batch_inference(self):
        if not hasattr(self, "_inference_accum"):
            return
        for layer_name, activations_list in self._inference_accum.items():
            concat = torch.cat(activations_list, dim=1)
            if layer_name in self.ipca_models and self.pca_components is not None:
                transformed = self.ipca_models[layer_name].transform(concat.numpy())
                output_tensor = torch.tensor(transformed, dtype=torch.float16)
            else:
                output_tensor = concat.to(torch.float16)
            # Store as the only activation for this batch
            if layer_name not in self.activations:
                self.activations[layer_name] = {}
            self.activations[layer_name][self._current_batch] = [output_tensor]
        self._inference_accum = {}

    def finalize_batch_training(self):
        print("[DEBUG] Finalizing batch training...")
        # For each layer, concatenate across feature dim and partial_fit IPCA
        for layer_name, activations_list in self._training_activations.items():
            if not activations_list:
                continue
            print(f"[DEBUG] Found {len(activations_list)} activations for {layer_name}")
            try:
                concat = np.concatenate([a for a in activations_list], axis=1)
                print(f"[DEBUG] Concatenated shape: {concat.shape}")
            except Exception as e:
                print(
                    f"[ERROR] Failed to concatenate activations for {layer_name}: {e}"
                )
                continue
            # Adjust n_components if we have fewer features
            n_features = concat.shape[1]
            n_components = min(self.pca_components, n_features)

            if layer_name not in self.ipca_models:
                self.ipca_models[layer_name] = IncrementalPCA(n_components=n_components)
            self.ipca_models[layer_name].partial_fit(concat)
        self._training_activations = {}

    def _resolve_layer(self, model, layer_path: str):
        parts = layer_path.split(".")
        layer = model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    def register(self, model: torch.nn.Module, layer_names: List[str]):
        self.clear()
        for name in layer_names:
            try:
                layer = self._resolve_layer(model, name)
            except AttributeError as e:
                print(f"[ERROR] Could not resolve layer '{name}': {e}")
                continue
            hook = layer.register_forward_hook(self._get_hook(name))
            self.hooks.append(hook)
        self._active = True

    def save(self):
        for layer_name, layer_activations in self.activations.items():
            activations = []
            for batch_id, batch_activations in layer_activations.items():
                # Concatenate activations for this batch across the feature dimension
                temp = torch.cat(batch_activations, dim=1)
                activations.append(temp)

            if activations:
                try:
                    tensor = torch.cat(activations, dim=0)
                except Exception as e:
                    print(f"[ERROR] Layer {layer_name} could not be stacked.")
                    print(
                        f"[ERROR] Shapes of activations: {[a.shape for a in activations]}"
                    )
                    print(f"[ERROR] Skipping saving for this layer.")
                    continue

                file_path = f"{self.output_dir}/activations/{self.dataset_name}/{self.model_name}/{layer_name}/activations.pt"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                print(f"Saving activations to: {file_path}")
                torch.save(tensor, file_path)

    def clear(self):
        self.hooks = []
        self.activations = {}

    def get(self):
        return self.activations

    def flush(self):
        self.save()
        self.clear()

    def get_ipca_models(self):
        return self.ipca_models

    def save_ipca_models(self):
        if not self.ipca_models:
            return

        for layer_name, ipca_model in self.ipca_models.items():
            file_path = f"{self.output_dir}/activations/{self.dataset_name}/{self.model_name}/{layer_name}/ipca_model.pkl"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as f:
                pickle.dump(ipca_model, f)
