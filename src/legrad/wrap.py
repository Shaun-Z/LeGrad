# src/my_transformers/wrap.py
from typing import List, Optional, Any, Iterable
import torch.nn as nn
import warnings
import torch

class CopyAttrWrapper(nn.Module):
    """
    Copies public attributes from the given model onto this wrapper (shallow copy).
    Keeps a reference to the original model as _orig_model_ref.
    DOES NOT implement specific hook behavior — externals should register hooks
    via `register_hooks` or call `hooks` helpers.
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None,
                 include_private: bool = False):
        super().__init__()
        object.__setattr__(self, "_orig_model_ref", model)
        object.__setattr__(self, "_copied_attrs", [])
        object.__setattr__(self, "_hook_handles", [])

        # copy public attributes (skip dunders)
        for attr in dir(model):
            if attr.startswith("__"):
                continue
            if not include_private and attr.startswith("_"):
                continue
            # avoid overwriting wrapper's own attrs
            if hasattr(self, attr):
                continue
            try:
                val = getattr(model, attr)
            except Exception:
                continue
            try:
                setattr(self, attr, val)
                self._copied_attrs.append(attr)
            except Exception:
                continue

        object.__setattr__(self, "_requested_hook_indices", list(layer_indices))

    def forward(self, *args, **kwargs):
        """Delegate forward to original model to avoid recursion issues."""
        orig = object.__getattribute__(self, "_orig_model_ref")
        if hasattr(orig, "forward"):
            return orig.forward(*args, **kwargs)
        if callable(orig):
            return orig(*args, **kwargs)
        raise RuntimeError("Original model has no forward or is not callable")

    def attach_hook_handles(self, handles: List[Any]):
        """Store hook handles returned by register_forward_hook so we can remove them later."""
        object.__setattr__(self, "_hook_handles", list(handles))

    def remove_hook_handles(self):
        handles = object.__getattribute__(self, "_hook_handles")
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        object.__setattr__(self, "_hook_handles", [])

    def get_hook_outputs(self):
        return getattr(self, "_hook_outputs", {})

    def original_model(self):
        return object.__getattribute__(self, "_orig_model_ref")
    
    def _get_device_for_call(self, device: Optional[str] = None):
        # Try to get the device from the original model's parameters, otherwise use the passed device or cpu
        orig = self.original_model()
        if device is not None:
            return torch.device(device)
        try:
            # Find the device of the first parameter
            for p in orig.parameters():
                return p.device
        except Exception:
            pass
        return torch.device("cpu")

    def to(self, *args, **kwargs):
        # Move the original model to the target device as well
        orig = self.original_model()
        try:
            if hasattr(orig, "to"):
                orig.to(*args, **kwargs)
        except Exception:
            # Ignore errors when moving the original model, but still try to call the parent class's to
            pass
        # CopyAttrWrapper has no tensor buffers of its own, still call the parent class (it will move parameters registered to the wrapper)
        return super().to(*args, **kwargs)
