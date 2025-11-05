"""Collate/transform registry for simple name-based selection.

Usage in your collate module under lightly.data:

from lightly.data.registry import register_collate

@register_collate("my_collate")
class MyCollate:
    def __init__(self, ...): ...
    def __call__(self, batch): ...

Then set in config: collate.name: "my_collate"
Other collate keys are passed as kwargs to the constructor.
"""

from typing import Any, Callable, Dict, Optional, Type


CollateType = Any  # Accept classes or callables; constructor will be called with kwargs

COLLATE_REGISTRY: Dict[str, Type] = {}


def register_collate(name: str):
    def _decorator(cls_or_callable: Type) -> Type:
        COLLATE_REGISTRY[name.lower()] = cls_or_callable
        return cls_or_callable

    return _decorator


def get_registered_collate(name: str) -> Optional[Type]:
    return COLLATE_REGISTRY.get(name.lower())
