"""Helper utilities for handling timm model activations."""

import torch


def detect_model_family(model_name: str) -> str:
    """
    Detect the model family from the model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model family name: 'vit', 'swin', 'convnext', 'resnet', or 'default'
    """
    model_name_lower = model_name.lower()
    
    if 'vit' in model_name_lower or 'vision_transformer' in model_name_lower:
        return 'vit'
    elif 'swin' in model_name_lower:
        return 'swin'
    elif 'convnext' in model_name_lower:
        return 'convnext'
    elif 'resnet' in model_name_lower:
        return 'resnet'
    else:
        return 'default'


def process_vit_activations(output: torch.Tensor) -> torch.Tensor:
    """
    Process Vision Transformer activations.
    
    ViT models have outputs with shape (batch, num_tokens, embed_dim)
    where num_tokens = num_patches + 1 (cls token).
    We flatten the token and embedding dimensions.
    
    Args:
        output: Activation tensor from a ViT layer
        
    Returns:
        Processed tensor with shape (batch, num_tokens * embed_dim)
    """
    if len(output.shape) == 3:  # (batch, num_tokens, embed_dim)
        # Flatten token and embedding dimensions
        batch_size = output.shape[0]
        return output.reshape(batch_size, -1)
    return output


def process_swin_activations(output: torch.Tensor) -> torch.Tensor:
    """
    Process Swin Transformer activations.
    
    Swin models have hierarchical outputs with varying shapes depending on stage.
    Early stages: (batch, H, W, embed_dim)
    Later stages: (batch, num_patches, embed_dim)
    
    Args:
        output: Activation tensor from a Swin layer
        
    Returns:
        Processed tensor with shape (batch, features)
    """
    batch_size = output.shape[0]
    return output.reshape(batch_size, -1)


def process_convnext_activations(output: torch.Tensor) -> torch.Tensor:
    """
    Process ConvNeXt activations.
    
    ConvNeXt models have standard conv outputs with shape (batch, channels, H, W).
    
    Args:
        output: Activation tensor from a ConvNeXt layer
        
    Returns:
        Processed tensor with shape (batch, channels * H * W)
    """
    batch_size = output.shape[0]
    return output.reshape(batch_size, -1)


def process_activations(output: torch.Tensor, model_family: str) -> torch.Tensor:
    """
    Process activations based on model family.
    
    Args:
        output: Raw activation tensor
        model_family: Type of model ('vit', 'swin', 'convnext', 'resnet', 'default')
        
    Returns:
        Processed activation tensor with shape (batch, features)
    """
    # Handle tuple/list outputs (e.g., from some model layers)
    if isinstance(output, (list, tuple)):
        # Take the first element or stack if multiple
        if len(output) == 1:
            output = output[0]
        else:
            output = torch.stack(list(output), dim=0)
    
    # Apply family-specific processing
    if model_family == 'vit':
        return process_vit_activations(output)
    elif model_family == 'swin':
        return process_swin_activations(output)
    elif model_family == 'convnext':
        return process_convnext_activations(output)
    else:
        # Default processing: flatten all dimensions except batch
        batch_size = output.shape[0]
        return output.reshape(batch_size, -1)
