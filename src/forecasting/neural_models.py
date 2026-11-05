"""Configuration and diagnostics for reproducible NeuralForecast experiments.

The module deliberately does not import NeuralForecast at import time.  This keeps
the shared forecasting core usable in lightweight/CPU agent processes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class NeuralTrainingMode:
    name: str
    max_steps: int
    input_size: int
    batch_size: int
    windows_batch_size: int
    learning_rate: float
    val_check_steps: int
    early_stop_patience_steps: int
    validation_origins: tuple[int, ...]
    num_samples: int

    def __post_init__(self) -> None:
        if self.name not in {"fast", "research"}:
            raise ValueError("name must be 'fast' or 'research'")
        for field in ("max_steps", "input_size", "batch_size", "windows_batch_size"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")


FAST_MODE = NeuralTrainingMode(
    "fast", 20, 112, 16, 128, 1e-3, 10, 2, (1857,), 200
)

# Sized for a 4-GB GPU: smaller batches/windows than the smoke configuration,
# with enough optimizer steps and three chronological validation origins.
RESEARCH_MODE = NeuralTrainingMode(
    "research", 750, 168, 8, 32, 5e-4, 50, 5, (1801, 1829, 1857), 1000
)


def training_mode(name: str) -> NeuralTrainingMode:
    modes = {"fast": FAST_MODE, "research": RESEARCH_MODE}
    try:
        return modes[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown training mode: {name!r}") from exc


def neural_candidates(mode: str = "research") -> list[dict[str, Any]]:
    """Return predeclared validation candidates; never selected on test data."""
    candidates: list[dict[str, Any]] = [
        {"family": "NHITS", "name": "NHITS_small", "scaler": "robust",
         "mlp_units": [[128, 128], [128, 128], [128, 128]]},
        {"family": "DeepAR", "name": "DeepAR_StudentT", "scaler": "robust",
         "distribution": "StudentT", "hidden_size": 32},
        {"family": "DeepAR", "name": "DeepAR_NegativeBinomial", "scaler": "identity",
         "distribution": "NegativeBinomial", "hidden_size": 32},
    ]
    if mode.lower() == "research":
        candidates.insert(1, {"family": "NHITS", "name": "NHITS_base",
                              "scaler": "robust", "mlp_units": None})
    return candidates


def cuda_diagnostics() -> dict[str, Any]:
    """Collect interpreter, PyTorch-build, CUDA and driver diagnostics safely."""
    import torch

    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device": "cpu",
    }
    if result["cuda_available"]:
        result["device"] = "cuda"
        result["device_name"] = torch.cuda.get_device_name(0)
        result["device_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                              check=False, timeout=10)
        result["nvidia_smi_returncode"] = proc.returncode
        result["nvidia_smi_summary"] = (proc.stdout or proc.stderr).strip()[:2000]
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["nvidia_smi_returncode"] = None
        result["nvidia_smi_summary"] = f"unavailable: {exc}"
    return result


def save_training_metadata(path: str | Path, *, mode: NeuralTrainingMode,
                           candidate: dict[str, Any], diagnostics: dict[str, Any],
                           outcome: dict[str, Any]) -> Path:
    """Persist configuration/outcome/checkpoint facts without inventing loss data."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": asdict(mode), "candidate": candidate,
               "diagnostics": diagnostics, "outcome": outcome}
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return destination
