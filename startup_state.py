"""Persistent startup state and CUDA runtime diagnostics for Parakeet."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config


STATE_FILE = Path(
    os.getenv("PARAKEET_STARTUP_STATE_FILE", str(config.BASE_DIR / "startup_state.json"))
)


def update_startup_state(phase: str, message: str = "", **details: Any) -> Dict[str, Any]:
    """Atomically persist the latest startup phase for diagnostics and launchers."""
    previous_phase = None
    try:
        if STATE_FILE.is_file():
            previous_phase = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("phase")
    except (OSError, ValueError, TypeError):
        previous_phase = None

    payload: Dict[str, Any] = {
        "phase": phase,
        "message": message,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    if previous_phase and previous_phase != phase:
        payload["previous_phase"] = previous_phase
    attempt = os.getenv("PARAKEET_STARTUP_ATTEMPT")
    if attempt:
        payload["attempt"] = attempt
    if details:
        payload["details"] = details

    temp_path = STATE_FILE.with_name(f".{STATE_FILE.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_path, STATE_FILE)
    except OSError as exc:
        print(f"[startup] warning: could not write {STATE_FILE}: {exc}", flush=True)
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    summary = f"[startup] phase={phase}"
    if message:
        summary += f" message={message}"
    print(summary, flush=True)
    return payload


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "").lower()
        if name in {"torch", "torchvision", "torchaudio", "nemo-toolkit", "cuda-python"} or name.startswith(
            "nvidia-"
        ):
            versions[name] = distribution.version
    return dict(sorted(versions.items()))


def _nvidia_smi_summary() -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,compute_cap,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def collect_runtime_info(require_cuda: bool = False) -> Tuple[Dict[str, Any], Optional[str]]:
    """Collect the exact Torch/CUDA stack and exercise CUDA when it is required."""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    nvidia_smi = _nvidia_smi_summary()
    if nvidia_smi:
        info["nvidia_smi"] = nvidia_smi

    try:
        import torch
    except Exception as exc:
        error = f"PyTorch import failed: {type(exc).__name__}: {exc}"
        info["cuda_probe"] = "failed"
        return info, error

    info.update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
        }
    )

    devices: List[Dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": ".".join(str(part) for part in torch.cuda.get_device_capability(index)),
                }
            )
        info["devices"] = devices
        info["compiled_arches"] = torch.cuda.get_arch_list()

    package_families = sorted(
        {
            "cu12" if name.endswith("-cu12") else "cu13"
            for name in info["packages"]
            if name.endswith("-cu12") or name.endswith("-cu13")
        }
    )
    info["cuda_package_families"] = package_families
    if len(package_families) > 1:
        info["warning"] = "Both CUDA 12 and CUDA 13 runtime packages are installed in this virtual environment."

    if require_cuda and not torch.cuda.is_available():
        info["cuda_probe"] = "failed"
        return info, "CUDA is required for this startup mode but torch.cuda.is_available() is false."

    if require_cuda:
        try:
            probe = torch.ones(1, device="cuda")
            probe.add_(1)
            torch.cuda.synchronize()
            info["cuda_probe"] = "passed"
        except Exception as exc:
            info["cuda_probe"] = "failed"
            return info, f"CUDA tensor probe failed: {type(exc).__name__}: {exc}"
    else:
        info["cuda_probe"] = "not_required"

    return info, None


def _parse_details(values: List[str]) -> Dict[str, str]:
    details: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Detail must use key=value format: {value}")
        key, detail_value = value.split("=", 1)
        details[key] = detail_value
    return details


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Parakeet startup state")
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_parser = subparsers.add_parser("set", help="Persist a startup phase")
    set_parser.add_argument("phase")
    set_parser.add_argument("message", nargs="?", default="")
    set_parser.add_argument("--detail", action="append", default=[])

    runtime_parser = subparsers.add_parser("check-runtime", help="Record Torch and CUDA compatibility")
    runtime_parser.add_argument("--require-cuda", action="store_true")

    subparsers.add_parser("show", help="Print the latest startup state")
    args = parser.parse_args()

    if args.command == "set":
        update_startup_state(args.phase, args.message, **_parse_details(args.detail))
        return 0
    if args.command == "show":
        if not STATE_FILE.is_file():
            print(f"No startup state at {STATE_FILE}")
            return 1
        print(STATE_FILE.read_text(encoding="utf-8"), end="")
        return 0

    update_startup_state("checking_runtime", "Collecting Torch, CUDA, driver, and GPU compatibility")
    runtime_info, error = collect_runtime_info(require_cuda=args.require_cuda)
    if error:
        update_startup_state("runtime_failed", error, runtime=runtime_info)
        print(f"ERROR: {error}", flush=True)
        return 1
    update_startup_state("runtime_ready", "Runtime compatibility probe passed", runtime=runtime_info)
    print(json.dumps(runtime_info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
