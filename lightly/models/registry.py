"""Simple registry for custom models.

Users can register a model by name via the @register_model("name") decorator
in any module under lightly.models. The CLI can then instantiate it by setting
model.name: "name" in the config (optionally with model.init_args for kwargs).
"""

from typing import Dict, Optional, Type

from torch import nn


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class under a lowercase name."""

    def _decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name.lower()] = cls
        return cls

    return _decorator


def get_registered_model(name: str) -> Optional[Type[nn.Module]]:
    return MODEL_REGISTRY.get(name.lower())
