"""Stub module for mlx_vlm to satisfy vision detection/runtime checks in tests."""


def load(model_path, *args, **kwargs):
    # Return simple tuple to mirror real API shape
    return ("vision-model", "vision-processor")


def generate(model, processor, prompt=None, images=None, **kwargs):
    suffix = ""
    if images:
        suffix = f" [images={len(images)}]"
    return f"VISION:{prompt}{suffix}"


__all__ = ["load", "generate"]
