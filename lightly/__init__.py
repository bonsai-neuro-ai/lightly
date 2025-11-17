"""Lightly is a computer vision framework for self-supervised learning.

With Lightly you can train deep learning models using
self-supervision. This means, that you don't require
any labels to train a model. Lightly has been built
to help you understand and work with large unlabeled datasets.
It is built on top of PyTorch and therefore fully compatible 
with other frameworks such as Fast.ai.

The framework is structured into the following modules:

- **api**: 

  The lightly.api module handles communication with the Lightly web-app.

- **cli**:

  The lightly.cli module provides a command-line interface for training 
  self-supervised models and embedding images. Furthermore, the command-line
  tool can be used to upload and download images from/to the Lightly web-app.

- **core**:

  The lightly.core module offers one-liners for simple self-supervised learning.

- **data**:

  The lightly.data module provides a dataset wrapper and collate functions. The
  collate functions are in charge of the data augmentations which are crucial for
  self-supervised learning.

- **loss**:

  The lightly.loss module contains implementations of popular self-supervised training
  loss functions.

- **models**:

  The lightly.models module holds the implementation of the ResNet as well as heads
  for self-supervised methods. It currently implements the heads of:

  - Barlow Twins

  - BYOL
  
  - MoCo
  
  - NNCLR
  
  - SimCLR
  
  - SimSiam
  
  - SwaV

- **transforms**:

  The lightly.transforms module implements custom data transforms. Currently implements:

  - Gaussian Blur

  - Random Rotation

  - Random Solarization

- **utils**:

  The lightly.utils package provides global utility methods.
  The io module contains utility to save and load embeddings in a format which is
  understood by the Lightly library.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

__name__ = "lightly"
__version__ = "1.5.22"


import os


def _populate_registries():
    """Import submodules to trigger registration decorators."""
    import importlib
    import pkgutil
    
    for pkg_name in ["lightly.models", "lightly.loss", "lightly.data"]:
        try:
            pkg = importlib.import_module(pkg_name)
            if hasattr(pkg, "__path__"):
                for modinfo in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
                    # Skip private modules
                    if any(part.startswith("_") for part in modinfo.name.split(".")):
                        continue
                    try:
                        importlib.import_module(modinfo.name)
                    except Exception:
                        # Best-effort; ignore import failures
                        pass
        except Exception:
            pass


_populate_registries()


if os.getenv("LIGHTLY_DID_VERSION_CHECK", "False") == "False":
    os.environ["LIGHTLY_DID_VERSION_CHECK"] = "True"
    import multiprocessing

    if multiprocessing.current_process().name == "MainProcess":
        try:
            from lightly.api import _version_checking
            _version_checking.check_is_latest_version_in_background(
                current_version=__version__
            )
        except Exception:
            # Ignore version check failures (e.g., missing pydantic)
            pass
