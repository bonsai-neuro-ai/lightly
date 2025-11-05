from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from itertools import product
from typing import Any, Dict

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
import torchvision.models as tvm
import torchvision.transforms as T
from tqdm import tqdm


def _train_transforms(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_model(model_cfg: Dict[str, Any]) -> torch.nn.Module:
    arch = model_cfg.get("arch", "resnet50").lower()
    num_classes = int(model_cfg.get("num_classes", 1000))
    weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if (model_cfg.get("pretrained", False) and arch == "resnet50") else None
    fn_map = {
        "resnet18": tvm.resnet18,
        "resnet34": tvm.resnet34,
        "resnet50": tvm.resnet50,
        "resnet101": tvm.resnet101,
        "resnet152": tvm.resnet152,
    }
    if arch not in fn_map:
        raise ValueError(f"Unsupported arch: {arch}")
    model = fn_map[arch](weights=weights)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


@dataclass
class BenchmarkResult:
    params: Dict[str, Any]
    avg_time_per_batch: float
    throughput_samples_per_sec: float
    gpu_memory_used_gb: float
    cpu_usage_percent: float
    successful: bool
    error_msg: str = ""


class ParameterOptimizer:
    def __init__(self, base_config: Dict[str, Any], device: str = "cuda"):
        self.base_config = base_config
        self.device = device
        self.results = []

    def create_fake_dataset(
        self, num_samples: int = 1000, num_classes: int = 1000, image_size: int = 224
    ) -> torch.utils.data.Dataset:
        transform = _train_transforms(image_size)
        return FakeData(
            size=num_samples,
            image_size=(3, image_size, image_size),
            num_classes=num_classes,
            transform=transform,
        )

    def benchmark_dataloader_config(self, config: Dict[str, Any], num_batches: int = 50) -> BenchmarkResult:
        try:
            dataset = self.create_fake_dataset(
                num_samples=config.get("fake_dataset_size", 1000),
                image_size=config.get("image_size", 224),
            )

            dataloader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config.get("pin_memory", True),
                shuffle=True,
                persistent_workers=config.get("persistent_workers", True) if config["num_workers"] > 0 else False,
                prefetch_factor=config.get("prefetch_factor", 2) if config["num_workers"] > 0 else None,
            )

            # Warmup
            for i, (data, target) in enumerate(dataloader):
                if i >= 3:
                    break
                if self.device == "cuda":
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

            if self.device == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            total_samples = 0

            process = psutil.Process()
            cpu_usage = []
            gpu_memory_peak = 0.0

            for i, (data, target) in enumerate(dataloader):
                if i >= num_batches:
                    break
                if self.device == "cuda":
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                total_samples += data.size(0)
                cpu_usage.append(process.cpu_percent())
                if self.device == "cuda":
                    gpu_memory_peak = max(
                        gpu_memory_peak, torch.cuda.max_memory_allocated() / 1024**3
                    )

            if self.device == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            total_time = max(1e-9, end_time - start_time)
            avg_time_per_batch = total_time / min(num_batches, len(dataloader))
            throughput = total_samples / total_time

            return BenchmarkResult(
                params=config,
                avg_time_per_batch=avg_time_per_batch,
                throughput_samples_per_sec=throughput,
                gpu_memory_used_gb=gpu_memory_peak,
                cpu_usage_percent=float(np.mean(cpu_usage)) if cpu_usage else 0.0,
                successful=True,
            )
        except Exception as e:
            return BenchmarkResult(
                params=config,
                avg_time_per_batch=float("inf"),
                throughput_samples_per_sec=0.0,
                gpu_memory_used_gb=0.0,
                cpu_usage_percent=0.0,
                successful=False,
                error_msg=str(e),
            )

    def benchmark_model_config(self, config: Dict[str, Any], num_batches: int = 20) -> BenchmarkResult:
        try:
            model = create_model(config["model"]).to(self.device)
            optimizer = torch.optim.SGD(model.parameters(), lr=config.get("lr", 0.1))
            criterion = torch.nn.CrossEntropyLoss()

            dataset = self.create_fake_dataset(
                num_samples=config.get("fake_dataset_size", 500),
                image_size=config["model"].get("image_size", 224),
            )

            dataloader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                num_workers=config.get("num_workers", 4),
                pin_memory=True,
                shuffle=True,
            )

            model.train()
            for i, (data, target) in enumerate(dataloader):
                if i >= 2:
                    break
                data, target = data.to(self.device), target.to(self.device)
                out = model(data)
                loss = criterion(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.device == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            total_samples = 0

            process = psutil.Process()
            cpu_usage = []
            gpu_memory_peak = 0.0

            for i, (data, target) in enumerate(dataloader):
                if i >= num_batches:
                    break
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                total_samples += data.size(0)
                cpu_usage.append(process.cpu_percent())
                if self.device == "cuda":
                    gpu_memory_peak = max(
                        gpu_memory_peak, torch.cuda.max_memory_allocated() / 1024**3
                    )

            if self.device == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            total_time = max(1e-9, end_time - start_time)
            avg_time_per_batch = total_time / min(num_batches, len(dataloader))
            throughput = total_samples / total_time

            return BenchmarkResult(
                params=config,
                avg_time_per_batch=avg_time_per_batch,
                throughput_samples_per_sec=throughput,
                gpu_memory_used_gb=gpu_memory_peak,
                cpu_usage_percent=float(np.mean(cpu_usage)) if cpu_usage else 0.0,
                successful=True,
            )
        except Exception as e:
            return BenchmarkResult(
                params=config,
                avg_time_per_batch=float("inf"),
                throughput_samples_per_sec=0.0,
                gpu_memory_used_gb=0.0,
                cpu_usage_percent=0.0,
                successful=False,
                error_msg=str(e),
            )

    def optimize_dataloader_params(self) -> Dict[str, Any]:
        print("Optimizing data loader parameters")

        num_workers_options = [1, 2, 3, 4, 6, 7]
        batch_size_options = [32, 64, 128, 256, 512]
        pin_memory_options = [True, False]
        persistent_workers_options = [True, False]
        prefetch_factor_options = [1, 2, 4]

        cpu_count = psutil.cpu_count()
        num_workers_options = [w for w in num_workers_options if w <= (cpu_count or 1) * 2]

        best_result = None
        results = []

        total_combinations = (
            len(num_workers_options)
            * len(batch_size_options)
            * len(pin_memory_options)
            * len(persistent_workers_options)
            * len(prefetch_factor_options)
        )

        with tqdm(total=total_combinations, desc="DataLoader optimization") as pbar:
            for num_workers, batch_size, pin_memory, persistent_workers, prefetch_factor in product(
                num_workers_options,
                batch_size_options,
                pin_memory_options,
                persistent_workers_options,
                prefetch_factor_options,
            ):
                config = {
                    "num_workers": num_workers,
                    "batch_size": batch_size,
                    "pin_memory": pin_memory,
                    "persistent_workers": persistent_workers and num_workers > 0,
                    "prefetch_factor": prefetch_factor if num_workers > 0 else None,
                    "image_size": 224,
                    "fake_dataset_size": 1000,
                }

                result = self.benchmark_dataloader_config(config, num_batches=10)
                results.append(result)

                if result.successful and (
                    best_result is None
                    or result.throughput_samples_per_sec
                    > best_result.throughput_samples_per_sec
                ):
                    best_result = result

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "best_throughput": f"{best_result.throughput_samples_per_sec:.1f}" if best_result else "0",
                        "current": f"{result.throughput_samples_per_sec:.1f}" if result.successful else "failed",
                    }
                )

        self.dataloader_results = results

        print("   Best data loader configuration:")
        if best_result:
            print(f"   Throughput: {best_result.throughput_samples_per_sec:.1f} samples/sec")
            print(f"   Parameters: {best_result.params}")
            return best_result.params
        return {
            "num_workers": 4,
            "batch_size": 64,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "image_size": 224,
            "fake_dataset_size": 1000,
        }

    def optimize_model_params(self, dataloader_params: Dict[str, Any]) -> Dict[str, Any]:
        print("\n Optimizing model parameters")

        arch_options = ["resnet18", "resnet34", "resnet50"]
        batch_size_options = [32, 64, 128, 256]

        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb < 12:
                batch_size_options = [b for b in batch_size_options if b <= 128]

        best_result = None
        results = []

        total_combinations = len(arch_options) * len(batch_size_options)

        with tqdm(total=total_combinations, desc="Model optimization") as pbar:
            for arch, batch_size in product(arch_options, batch_size_options):
                config = {
                    "model": {
                        "arch": arch,
                        "num_classes": 1000,
                        "pretrained": False,
                        "image_size": 224,
                    },
                    "batch_size": batch_size,
                    "lr": 0.1,
                    "num_workers": dataloader_params.get("num_workers", 4),
                    "fake_dataset_size": 200,
                }

                result = self.benchmark_model_config(config, num_batches=5)
                results.append(result)

                if result.successful and (
                    best_result is None
                    or result.throughput_samples_per_sec
                    > best_result.throughput_samples_per_sec
                ):
                    best_result = result

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "best_throughput": f"{best_result.throughput_samples_per_sec:.1f}" if best_result else "0",
                        "current": f"{result.throughput_samples_per_sec:.1f}" if result.successful else "failed",
                    }
                )

        self.model_results = results

        print("   Best model configuration:")
        if best_result:
            print(f"   Throughput: {best_result.throughput_samples_per_sec:.1f} samples/sec")
            print(f"   Architecture: {best_result.params['model']['arch']}")
            print(f"   Batch size: {best_result.params['batch_size']}")
            print(f"   GPU memory: {best_result.gpu_memory_used_gb:.2f} GB")
            return best_result.params
        return {
            "model": {"arch": "resnet18", "num_classes": 1000, "pretrained": False, "image_size": 224},
            "batch_size": 64,
            "lr": 0.1,
        }

    def save_results(self, output_path: str = "optimization_results.json"):
        results = {
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / 1024**3,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
            },
            "dataloader_results": [asdict(r) for r in getattr(self, "dataloader_results", [])],
            "model_results": [asdict(r) for r in getattr(self, "model_results", [])],
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")

    def run_full_optimization(self) -> Dict[str, Any]:
        print("Starting parameter optimization")

        best_dataloader_params = self.optimize_dataloader_params()
        best_model_params = self.optimize_model_params(best_dataloader_params)

        optimal_config = {
            "data": best_dataloader_params,
            "model": best_model_params["model"],
            "training": {
                "batch_size": best_model_params["batch_size"],
                "lr": best_model_params.get("lr", 0.1),
            },
        }

        self.save_results()

        print("\nOptimization complete!")
        return optimal_config


if __name__ == "__main__":
    base_config = {"model": {"arch": "resnet50", "num_classes": 1000}, "data": {"batch_size": 256, "num_workers": 4}}
    optimizer = ParameterOptimizer(base_config)
    optimal_config = optimizer.run_full_optimization()
    print("\n Final optimal configuration:")
    print(json.dumps(optimal_config, indent=2))
