# Model registry: config selects by name, args go verbatim into the
# constructor. Registered models must expose param_groups() for the
# optimizer factory and, for fused-CE training, support
# forward(x, return_hidden=True) plus a bias-free `head.weight`.
# Optionally, metric_hooks() -> {'fast': [...], 'slow': [...]} contributes
# architecture-specific observability (see train/metrics.py).

from torch import nn

from .transformer_pp import TransformerPP

MODELS: dict[str, type[nn.Module]] = {
    'TransformerPP': TransformerPP,
}


def build_model(name: str, args: dict) -> nn.Module:
    return MODELS[name](**args)
