"""Registry for SSL method builders.

An SSL method builder is a callable that accepts the `cfg["model"]` dict and
returns an `nn.Module` to be used for training/embedding.
"""

from typing import Any, Callable, Dict, Optional

from torch import nn


SSL_METHOD_REGISTRY: Dict[str, Callable[[Dict[str, Any]], nn.Module]] = {}


def register_ssl_method(name: str):
    def _decorator(builder: Callable[[Dict[str, Any]], nn.Module]):
        SSL_METHOD_REGISTRY[name.lower()] = builder
        return builder

    return _decorator


def get_registered_ssl_method(name: str) -> Optional[Callable[[Dict[str, Any]], nn.Module]]:
    return SSL_METHOD_REGISTRY.get(name.lower())
