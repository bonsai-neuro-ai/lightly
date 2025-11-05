"""Loss registry for simple name-based selection.

Usage in your loss module under lightly.loss:

from lightly.loss.registry import register_loss

@register_loss("my_loss")
class MyLoss(nn.Module):
    ...

Then set in config: criterion.name: "my_loss"
Optionally pass kwargs via other criterion fields (e.g., margin: 0.1).
"""

from typing import Dict, Optional, Type

from torch import nn


LOSS_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_loss(name: str):
    def _decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        LOSS_REGISTRY[name.lower()] = cls
        return cls

    return _decorator


def get_registered_loss(name: str) -> Optional[Type[nn.Module]]:
    return LOSS_REGISTRY.get(name.lower())
