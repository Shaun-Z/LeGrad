
# Unified safe import may require types for isinstance
try:
    # timm VisionTransformer (location may vary with timm version)
    from timm.models.vision_transformer import VisionTransformer as TimmViT
except Exception:
    TimmViT = tuple()  # empty tuple -> isinstance(..., ()) is always False

try:
    # torchvision ViT (newer torchvision may expose VisionTransformer)
    from torchvision.models.vision_transformer import VisionTransformer as TorchViT
except Exception:
    TorchViT = tuple()

try:
    # open_clip VisionTransformer (location may vary)
    from open_clip.model import VisionTransformer as OpenCLIPViT
except Exception:
    OpenCLIPViT = tuple()

try:
    from open_clip.transformer import ResidualAttentionBlock as ResAttnBlk
except Exception:
    ResAttnBlk = tuple()