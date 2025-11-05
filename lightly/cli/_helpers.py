""" Command-Line Interface Helpers

This module also provides small factories to build models and utilities via
configuration, enabling users to plug in custom models/losses/collates by
supplying a dotted import path (e.g., "mypkg.mymodule:MyClass").
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import os
import importlib
import pkgutil
from typing import Any, Callable, Dict, Optional

import hydra
import torch
from hydra import utils
from torch import nn as nn

from lightly.cli._cli_simclr import _SimCLR
from lightly.embedding import SelfSupervisedEmbedding
from lightly.models import ZOO as model_zoo
from lightly.models import ResNetGenerator
from lightly.models.batchnorm import get_norm_layer
from lightly.utils.version_compare import version_compare
from lightly.models.registry import get_registered_model
from lightly.models.ssl_registry import get_registered_ssl_method
from lightly.loss.registry import get_registered_loss
from lightly.data.registry import get_registered_collate


def cpu_count():
    """Returns the number of CPUs which are present in the system.

    This number is not equivalent to the number of available CPUs to the process.

    """
    return os.cpu_count()


def fix_input_path(path):
    """Fix broken relative paths."""
    if not os.path.isabs(path):
        path = utils.to_absolute_path(path)
    return path


def fix_hydra_arguments(config_path: str = "config", config_name: str = "config"):
    """Helper to make hydra arugments adaptive to installed hydra version

    Hydra introduced the `version_base` argument in version 1.2.0
    We use this helper to provide backwards compatibility to older hydra verisons.
    """

    hydra_args = {"config_path": config_path, "config_name": config_name}

    try:
        if version_compare(hydra.__version__, "1.2.0") >= 0:
            hydra_args["version_base"] = None
        elif version_compare(hydra.__version__, "1.1.2") > 0:
            hydra_args["version_base"] = "1.1"
    except ValueError:
        pass

    return hydra_args


def is_url(checkpoint):
    """Check whether the checkpoint is a url or not."""
    is_url = "https://storage.googleapis.com" in checkpoint
    return is_url


def get_ptmodel_from_config(model):
    """Get a pre-trained model from the lightly model zoo."""
    key = model["name"]
    key += "/simclr"
    key += "/d" + str(model["num_ftrs"])
    key += "/w" + str(float(model["width"]))

    if key in model_zoo.keys():
        return model_zoo[key], key
    else:
        return "", key


def load_state_dict_from_url(url, map_location=None):
    """Try to load the checkopint from the given url."""
    try:
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=map_location)
        return state_dict
    except Exception:
        print("Not able to load state dict from %s" % (url))
        print("Retrying with http:// prefix")
    try:
        url = url.replace("https", "http")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=map_location)
        return state_dict
    except Exception:
        print("Not able to load state dict from %s" % (url))

    # in this case downloading the pre-trained model was not possible
    # notify the user and return
    return {"state_dict": None}


# ---------------------------------------------------------------------------
# Import utilities and small factories
# ---------------------------------------------------------------------------

def _import_from_path(class_path: str) -> Any:
    """Import a symbol from a dotted path.

    Accepts either "pkg.mod:ClassName" or "pkg.mod.ClassName".
    """
    if not class_path:
        raise ValueError("Empty class_path provided for import.")
    if ":" in class_path:
        module_path, attr = class_path.split(":", 1)
    else:
        parts = class_path.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid class_path '{class_path}'. Expected 'pkg.mod:Class' or 'pkg.mod.Class'."
            )
        module_path, attr = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _auto_import_submodules(package_name: str) -> None:
    """Import all submodules of a package to trigger decorator registration.

    This scans the package for modules and imports them. Safe to call multiple times.
    """
    try:
        pkg = importlib.import_module(package_name)
    except Exception:
        return
    if not hasattr(pkg, "__path__"):
        return
    for modinfo in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        # Skip private modules
        name = modinfo.name
        if any(part.startswith("_") for part in name.split(".")):
            continue
        try:
            importlib.import_module(name)
        except Exception:
            # Best-effort import; ignore failures to avoid breaking CLI
            pass


def _ensure_registries_loaded() -> None:
    """Ensure lightly.models/loss/data packages are imported so registries are populated."""
    _auto_import_submodules("lightly.models")
    _auto_import_submodules("lightly.loss")
    _auto_import_submodules("lightly.data")


def build_ssl_model(cfg_model: Dict[str, Any]) -> nn.Module:
    """Factory to build the SSL model used by CLI tools.

    Supports two modes:
    - Default SimCLR stack using ResNetGenerator and _SimCLR head (backward compatible).
    - Custom class via `cfg_model["class_path"]` with optional `cfg_model["init_args"]`.
    """
    # 1) Prefer explicit class_path over predefined methods
    class_path: Optional[str] = cfg_model.get("class_path") or ""
    if class_path:
        cls = _import_from_path(class_path)
        init_args = dict(cfg_model.get("init_args", {}))
        return cls(**init_args)

    # Load potential registered components
    _ensure_registries_loaded()

    # 2) SSL method builder via registry (model.method)
    method_name = (cfg_model.get("method") or "").strip()
    if method_name:
        builder = get_registered_ssl_method(method_name)
        if builder is not None:
            return builder(cfg_model)

    # 3) A concrete model class registered via lightly.models.registry with model.name
    name = str(cfg_model.get("name", "")).strip()
    if name:
        registered = get_registered_model(name)
        if registered is not None:
            init_args = dict(cfg_model.get("init_args", {}))
            return registered(**init_args)

    # 4) Fallback/default: SimCLR as before
    resnet = ResNetGenerator(cfg_model["name"], cfg_model["width"])
    last_conv_channels = list(resnet.children())[-1].in_features
    features = nn.Sequential(
        get_norm_layer(3, 0),
        *list(resnet.children())[:-1],
        nn.Conv2d(last_conv_channels, cfg_model["num_ftrs"], 1),
        nn.AdaptiveAvgPool2d(1),
    )

    return _SimCLR(
        features, num_ftrs=cfg_model["num_ftrs"], out_dim=cfg_model["out_dim"]
    )


def build_criterion(criterion_cfg: Dict[str, Any]) -> nn.Module:
    """Factory for criterion (loss): registry by name -> class_path -> default NTXentLoss."""
    _ensure_registries_loaded()
    cfg = dict(criterion_cfg) if isinstance(criterion_cfg, dict) else {}
    name = (cfg.get("name") or "").strip()
    if name:
        registered = get_registered_loss(name)
        if registered is None:
            raise ValueError(f"Unknown registered loss '{name}'.")
        kwargs = {k: v for k, v in cfg.items() if k not in ("name", "class_path")}
        return registered(**kwargs)
    if cfg.get("class_path"):
        class_path = cfg.pop("class_path")
        cls = _import_from_path(class_path)
        return cls(**cfg)
    return __import__("lightly.loss", fromlist=["NTXentLoss"]).NTXentLoss(**cfg)


def build_collate(collate_cfg: Dict[str, Any]):
    """Factory for collate: registry by name -> class_path -> default ImageCollateFunction."""
    from lightly.data import ImageCollateFunction

    _ensure_registries_loaded()
    cfg = dict(collate_cfg) if isinstance(collate_cfg, dict) else {}
    name = (cfg.get("name") or "").strip()
    if name:
        registered = get_registered_collate(name)
        if registered is None:
            raise ValueError(f"Unknown registered collate '{name}'.")
        kwargs = {k: v for k, v in cfg.items() if k not in ("name", "class_path")}
        return registered(**kwargs)
    if cfg.get("class_path"):
        class_path = cfg.pop("class_path")
        builder = _import_from_path(class_path)
        return builder(**cfg)
    return ImageCollateFunction(**cfg)


def _maybe_expand_batchnorm_weights(model_dict, state_dict, num_splits):
    """Expands the weights of the BatchNorm2d to the size of SplitBatchNorm."""
    running_mean = "running_mean"
    running_var = "running_var"

    for key, item in model_dict.items():
        # not batchnorm -> continue
        if not running_mean in key and not running_var in key:
            continue

        state = state_dict.get(key, None)
        # not in dict -> continue
        if state is None:
            continue
        # same shape -> continue
        if item.shape == state.shape:
            continue

        # found running mean or running var with different shapes
        state_dict[key] = state.repeat(num_splits)

    return state_dict


def _filter_state_dict(state_dict, remove_model_prefix_offset: int = 1):
    """Makes the state_dict compatible with the model.

    Prevents unexpected key error when loading PyTorch-Lightning checkpoints.
    Allows backwards compatability to checkpoints before v1.0.6.

    """

    prev_backbone = "features"
    curr_backbone = "backbone"

    new_state_dict = {}
    for key, item in state_dict.items():
        # remove the "model." prefix from the state dict key
        key_parts = key.split(".")[remove_model_prefix_offset:]
        # with v1.0.6 the backbone of the models will be renamed from
        # "features" to "backbone", ensure compatability with old ckpts
        key_parts = [k if k != prev_backbone else curr_backbone for k in key_parts]

        new_key = ".".join(key_parts)
        new_state_dict[new_key] = item

    return new_state_dict


def _fix_projection_head_keys(state_dict):
    """Makes the state_dict compatible with the refactored projection heads.

    TODO: Remove once the models are refactored and the old checkpoints were
    replaced! Relevant issue: https://github.com/lightly-ai/lightly/issues/379

    Prevents unexpected key error when loading old checkpoints.

    """

    projection_head_identifier = "projection_head"
    prediction_head_identifier = "prediction_head"
    projection_head_insert = "layers"

    new_state_dict = {}
    for key, item in state_dict.items():
        if (
            projection_head_identifier in key or prediction_head_identifier in key
        ) and projection_head_insert not in key:
            # insert layers if it's not part of the key yet
            key_parts = key.split(".")
            key_parts.insert(1, projection_head_insert)
            new_key = ".".join(key_parts)
        else:
            new_key = key

        new_state_dict[new_key] = item

    return new_state_dict


def load_from_state_dict(
    model,
    state_dict,
    strict: bool = True,
    apply_filter: bool = True,
    num_splits: int = 0,
):
    """Loads the model weights from the state dictionary."""

    # step 1: filter state dict
    if apply_filter:
        state_dict = _filter_state_dict(state_dict)

    state_dict = _fix_projection_head_keys(state_dict)

    # step 2: expand batchnorm weights
    state_dict = _maybe_expand_batchnorm_weights(
        model.state_dict(), state_dict, num_splits
    )

    # step 3: load from checkpoint
    model.load_state_dict(state_dict, strict=strict)


def get_model_from_config(cfg, is_cli_call: bool = False) -> SelfSupervisedEmbedding:
    checkpoint = cfg["checkpoint"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detect if the model should be considered "custom" (no zoo download required)
    model_cfg = cfg["model"]
    is_custom = False
    # class_path explicitly set
    if model_cfg.get("class_path"):
        is_custom = True
    # registered SSL method
    elif (model_cfg.get("method") or "") and get_registered_ssl_method(model_cfg.get("method") or "") is not None:
        is_custom = True
    # registered concrete model by name
    elif (model_cfg.get("name") or "") and get_registered_model(str(model_cfg.get("name") or "")) is not None:
        is_custom = True

    state_dict = None
    if checkpoint:
        checkpoint = fix_input_path(checkpoint) if is_cli_call else checkpoint
        state_dict = torch.load(checkpoint, map_location=device)["state_dict"]
    else:
        if not is_custom:
            # For built-in SimCLR models, try to get from the zoo; otherwise, require a checkpoint
            checkpoint, key = get_ptmodel_from_config(model_cfg)
            if not checkpoint:
                msg = "Cannot download checkpoint for key {} ".format(key)
                msg += "because it does not exist!"
                raise RuntimeError(msg)
            state = load_state_dict_from_url(checkpoint, map_location=device)
            state_dict = state.get("state_dict") if isinstance(state, dict) else None

    # Build model via factory (will construct custom/registered models or default SimCLR)
    model = build_ssl_model(model_cfg).to(device)

    if state_dict is not None:
        load_from_state_dict(model, state_dict)

    encoder = SelfSupervisedEmbedding(model, None, None, None)
    return encoder
