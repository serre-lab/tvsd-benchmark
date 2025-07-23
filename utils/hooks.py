import os
import torch
from typing import List

class Activations:
    def __init__(self, output_dir: str, model_name: str, dataset_name: str):
        self.hooks = []
        self._active = True
        self._current_chunk = None
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.activations = {}  # {layer_name: [activations]}
        self.output_dir = output_dir

    def set_chunk(self, chunk):
        print("Setting current chunk to:", chunk)
        self._current_chunk = chunk

    def _get_hook(self, layer_name, debug=True):
        if debug:
            print(f"[DEBUG] Registering hook for layer: {layer_name}")
        def hook_fn(module, input, output):
            if debug:
                print(f"[DEBUG] Hook called for layer: {layer_name}")
            if self._active:
                if layer_name in self.activations:
                    self.activations[layer_name].append(output.detach().cpu())
                else:
                    self.activations[layer_name] = [output.detach().cpu()]
                if debug:
                    print(f"[DEBUG] Layer: {layer_name}, Output shape: {output.shape}")
        return hook_fn

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
        print("[DEBUG] Registering hooks for layers:", layer_names)
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
        print("[DEBUG] Saving activations to files in directory:", self.output_dir)
        for layer_name, activations in self.activations.items():
            if activations:
                print(f"IND ACTIVATION SHAPE: {activations[0].shape}")
                try:
                    tensor = torch.stack(activations)
                except Exception as e:
                    print(f"[ERROR] Layer {layer_name} could not be stacked.")
                    print(f"[ERROR]Shapes of activations: {[a.shape for a in activations]}")
                    raise e
                    
                file_path = f"{self.output_dir}/{self.dataset_name}/{self.model_name}/{layer_name}/chunk_{self._current_chunk}.pt"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                torch.save(tensor, file_path)

    def clear(self):
        self.hooks = []
        self.activations = {}

    def get(self):
        return self.activations

    def flush(self):
        self.save()
        self.clear()
