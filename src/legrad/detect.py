# src/my_transformers/detect.py
from typing import Optional, List, Any
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize, ToTensor
from .types import TimmViT, TorchViT, OpenCLIPViT
from .wrap import CopyAttrWrapper
# Import the specific backend wrapper (if it exists)
from .backends.openclip.wrapper import OpenCLIPWrapper, OpenCLIPCVWrapper
# TODO: create timm and torchvision wrapper files similarly
from .backends.timm.wrapper import TimmWrapper, TimmCVWrapper  # create file similarly
from .backends.torchvision.wrapper import TorchvisionWrapper, TorchvisionCVWrapper  # create file similarly

def detect_and_wrap(model: Any,
                    layer_indices: Optional[List[int]] = None,
                    prefer: Optional[str] = None,
                    include_private: bool = False,
                    use_cv_wrapper: bool = False) -> CopyAttrWrapper:
    """
    Simply determines and returns a specific backend CopyAttrWrapper instance based on isinstance.
    prefer: optional, string: 'openclip'|'timm'|'torchvision' to force branching (if matched)
    use_cv_wrapper: if True, returns the Chained-Vector (CV) variant of the wrapper
    """
    if model is None:
        raise ValueError("model cannot be None")

    # prefer preferred (override automatic judgment if needed)
    if prefer == "openclip" and isinstance(model.visual, OpenCLIPViT):
        if use_cv_wrapper:
            return OpenCLIPCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        return OpenCLIPWrapper(model, layer_indices=layer_indices, include_private=include_private)
    if prefer == "timm" and isinstance(model, TimmViT):
        if use_cv_wrapper:
            return TimmCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        return TimmWrapper(model, layer_indices=layer_indices, include_private=include_private)
    if prefer == "torchvision" and isinstance(model, TorchViT):
        if use_cv_wrapper:
            return TorchvisionCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        return TorchvisionWrapper(model, layer_indices=layer_indices, include_private=include_private)

    # Default type-based judgment (order can be adjusted)
    if isinstance(model.visual, OpenCLIPViT):
        if use_cv_wrapper:
            return OpenCLIPCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        return OpenCLIPWrapper(model, layer_indices=layer_indices, include_private=include_private)
    if isinstance(model, TimmViT):
        if use_cv_wrapper:
            return TimmCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        return TimmWrapper(model, layer_indices=layer_indices, include_private=include_private)
    if isinstance(model, TorchViT):
        if use_cv_wrapper:
            return TorchvisionCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        return TorchvisionWrapper(model, layer_indices=layer_indices, include_private=include_private)

    # fallback: Raise error
    raise TypeError("Unable to detect the backend type of the model, or the backend is not supported. Please ensure the model is an open_clip, timm, or torchvision ViT model, or use the prefer parameter to force specify the backend.")

def wrap_clip_preprocess(preprocess, image_size=224):
    """
    Modify OpenCLIP preprocessing to accept arbitrary image size.
    Args:
        preprocess: original OpenCLIP preprocess transform
        image_size: target image size (square)
    """
    return Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        preprocess.transforms[-3],
        preprocess.transforms[-2],
        preprocess.transforms[-1],
    ])

def wrap_timm_preprocess(preprocess, image_size=224):
    """
    Modify timm preprocessing to accept arbitrary image size.
    Args:
        preprocess: original timm preprocess transform
        image_size: target image size (square)
    """
    return Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        *preprocess.transforms[-2:],  # skip the first resize
    ])

def wrap_torchvision_preprocess(preprocess, image_size=224):
    """
    Modify torchvision preprocessing to accept arbitrary image size.
    Args:
        preprocess: original torchvision preprocess transform
        image_size: target image size (square)
    """
    return Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        # *preprocess.transforms[-2:],  # ToTensor and Normalize
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])