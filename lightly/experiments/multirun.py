"""
Universal trainer supporting multiple datasets, architectures, and training methods.

Adapted to run within the Lightly repo without external local modules.
Supports:
  - Supervised: torchvision ResNet backbones with CrossEntropy
  - SSL (minimal): simclr and barlow_twins using Lightly components

Notes:
  - COCO detection is not wired (placeholder left commented).
  - MoCo path is left unimplemented to avoid partial/incorrect behavior.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CIFAR10
import torchvision.models as tvm

from lightly.loss import BarlowTwinsLoss, NTXentLoss
from lightly.models import ResNetGenerator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StandardTransforms:
    @staticmethod
    def train_transforms(image_size: int) -> T.Compose:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def val_transforms(image_size: int) -> T.Compose:
        return T.Compose(
            [
                T.Resize(int(image_size * 1.14)),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class SSLPairTransform:
    """Simple two-view transform for SSL use-cases (SimCLR-style)."""

    def __init__(self, image_size: int):
        base = [
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.t1 = T.Compose(base)
        self.t2 = T.Compose(base)

    def __call__(self, img):
        return [self.t1(img), self.t2(img)]


class UniversalDataModule:
    SUPPORTED_DATASETS = {
        "imagenet": {
            "loader_class": ImageFolder,
            "num_classes": 1000,
            "image_size": 224,
            "structure": "train/val",
        },
        # "coco": { ... }  # detection requires different handling
        "cifar10": {
            "loader_class": CIFAR10,
            "num_classes": 10,
            "image_size": 32,
            "structure": "built-in",
        },
    }

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        batch_size: int = 256,
        num_workers: int = 8,
        training_method: str = "supervised",
    ):
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not supported. Available: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.training_method = training_method
        self.dataset_info = self.SUPPORTED_DATASETS[dataset_name]

    def get_transforms(self) -> Dict[str, T.Compose]:
        image_size = self.dataset_info["image_size"]

        if self.training_method == "supervised":
            return {
                "train": StandardTransforms.train_transforms(image_size),
                "val": StandardTransforms.val_transforms(image_size),
            }
        elif self.training_method in {"simclr", "barlow_twins"}:
            return {
                "train": SSLPairTransform(image_size),
                "val": StandardTransforms.val_transforms(image_size),
            }
        else:
            # Fallback to supervised transforms
            return {
                "train": StandardTransforms.train_transforms(image_size),
                "val": StandardTransforms.val_transforms(image_size),
            }

    def create_datasets(self):
        transforms_dict = self.get_transforms()

        if self.dataset_name == "imagenet":
            train_dataset = ImageFolder(
                str(self.data_dir / "train"), transform=transforms_dict["train"]
            )
            val_dataset = (
                ImageFolder(str(self.data_dir / "val"), transform=transforms_dict["val"])
                if (self.data_dir / "val").exists()
                else None
            )
        elif self.dataset_name == "cifar10":
            train_dataset = CIFAR10(
                str(self.data_dir), train=True, download=True, transform=transforms_dict["train"]
            )
            val_dataset = CIFAR10(
                str(self.data_dir), train=False, download=True, transform=transforms_dict["val"]
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        return train_dataset, val_dataset

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader | None]:
        train_dataset, val_dataset = self.create_datasets()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if self.training_method != "supervised" else False,
        )

        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            if val_dataset
            else None
        )

        return train_loader, val_loader


def _create_supervised_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    arch = arch.lower()
    fn_map = {
        "resnet18": tvm.resnet18,
        "resnet34": tvm.resnet34,
        "resnet50": tvm.resnet50,
        "resnet101": tvm.resnet101,
        "resnet152": tvm.resnet152,
    }
    if arch not in fn_map:
        raise ValueError(f"Unsupported architecture: {arch}")
    model = fn_map[arch](weights=tvm.ResNet50_Weights.IMAGENET1K_V1 if (pretrained and arch=="resnet50") else None)
    # Replace classifier to match num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class SSLModel(nn.Module):
    def __init__(self, arch: str = "resnet50", method: str = "simclr", proj_hidden: int = 512, out_dim: int = 128):
        super().__init__()
        self.method = method

        # Backbone via Lightly's ResNetGenerator
        resnet = ResNetGenerator(arch, 1)
        backbone_dim = resnet.model.fc.in_features
        resnet.model.fc = nn.Identity()
        self.backbone = resnet

        # Simple projection head
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, out_dim),
        )

        # Loss per method
        if method == "simclr":
            self.criterion = NTXentLoss()
        elif method == "barlow_twins":
            self.criterion = BarlowTwinsLoss()
        elif method == "moco":
            raise NotImplementedError("MoCo path not implemented in this minimal integration.")
        else:
            raise ValueError(f"SSL method {method} not supported")

    def forward(self, x):
        if isinstance(x, list):  # two views for SSL
            feats = [self.backbone(v) for v in x]
            proj = [self.projection_head(f) for f in feats]
            return proj
        feats = self.backbone(x)
        return self.projection_head(feats)


class Multirun:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        if config["training"]["method"] == "supervised":
            self.model = _create_supervised_model(
                arch=config["model"]["arch"],
                num_classes=config["model"]["num_classes"],
                pretrained=config["model"].get("pretrained", False),
            )
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.model = SSLModel(
                arch=config["model"]["arch"],
                method=config["training"]["method"],
            )
            self.criterion = self.model.criterion

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config["training"]["lr"],
            momentum=0.9,
            weight_decay=1e-4,
        )

        self.data_module = UniversalDataModule(
            dataset_name=config["data"]["dataset"],
            data_dir=config["data"]["data_dir"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            training_method=config["training"]["method"],
        )

        self.train_loader, self.val_loader = self.data_module.get_data_loaders()

        self.checkpoint_dir = Path(config["training"]["checkpoint_dir"]).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if self.config["training"]["method"] == "supervised":
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            else:
                inputs, _ = batch
                if isinstance(inputs, list):
                    inputs = [x.to(self.device) for x in inputs]
                else:
                    inputs = inputs.to(self.device)

                self.optimizer.zero_grad()
                features = self.model(inputs)
                loss = self.criterion(*features) if isinstance(features, list) else self.criterion(features)
                loss.backward()
                self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

            if batch_idx % 100 == 0:
                logger.info(
                    f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}"
                )

        return total_loss / max(1, num_batches)

    def validate(self) -> float | None:
        if self.config["training"]["method"] != "supervised" or self.val_loader is None:
            return None

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / max(1, total)

    def train(self):
        epochs = self.config["training"]["epochs"]

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Method: {self.config['training']['method']}")
        logger.info(f"Dataset: {self.config['data']['dataset']}")
        logger.info(f"Model: {self.config['model']['arch']}")

        for epoch in range(epochs):
            avg_loss = self.train_epoch()
            val_acc = self.validate()

            log_msg = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}"
            if val_acc is not None:
                log_msg += f", Val Acc: {val_acc:.2f}%"
            logger.info(log_msg)

            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": self.config,
                }
                torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pth")


def main():
    parser = argparse.ArgumentParser(description="Universal Model Trainer")

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["imagenet", "cifar10"],
        help="Dataset to use",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset")

    # Model options
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="Model architecture",
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights (supervised only)")

    # Training options
    parser.add_argument(
        "--method",
        type=str,
        default="supervised",
        choices=["supervised", "simclr", "barlow_twins"],
        help="Training method",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data workers")

    # System options
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory"
    )

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = (
        1000 if args.dataset == "imagenet" else 10 if args.dataset == "cifar10" else 1000
    )

    config = {
        "data": {
            "dataset": args.dataset,
            "data_dir": args.data_dir,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "model": {
            "arch": args.arch,
            "num_classes": num_classes,
            "pretrained": args.pretrained,
        },
        "training": {
            "method": args.method,
            "epochs": args.epochs,
            "lr": args.lr,
            "checkpoint_dir": args.checkpoint_dir,
        },
        "device": args.device,
    }

    logger.info("Training configuration:")
    logger.info(json.dumps(config, indent=2))

    trainer = Multirun(config)
    trainer.train()


if __name__ == "__main__":
    main()
