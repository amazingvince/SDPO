#!/usr/bin/env python
"""Minimal preflight check for PyTorch + CUDA before training."""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass


@dataclass
class CheckResult:
    ok: bool
    message: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check torch/cuda environment readiness.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        default=True,
        help="Fail if CUDA is unavailable (default: true).",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU-only success by disabling CUDA requirement.",
    )
    parser.add_argument(
        "--min-gpus",
        type=int,
        default=1,
        help="Minimum number of visible CUDA devices required.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index used for compute checks.",
    )
    parser.add_argument(
        "--check-all-required-gpus",
        action="store_true",
        default=True,
        help="Run compute checks on GPU[0..min_gpus-1] in addition to --gpu-index.",
    )
    parser.add_argument(
        "--skip-arch-compat-check",
        action="store_true",
        help="Skip CUDA arch compatibility pre-check against torch build arch list.",
    )
    parser.add_argument(
        "--fail-on-arch-mismatch",
        action="store_true",
        help="Fail when device capability is missing from torch CUDA arch list.",
    )
    parser.add_argument(
        "--matmul-size",
        type=int,
        default=1024,
        help="Square matrix size for GPU matmul sanity test.",
    )
    return parser.parse_args()


def _print_header(title: str) -> None:
    print(f"\n== {title} ==")


def _run_checks(args: argparse.Namespace) -> CheckResult:
    if args.allow_cpu:
        args.require_cuda = False

    try:
        import torch
    except Exception as exc:
        return CheckResult(False, f"Failed to import torch: {exc}")

    _print_header("Torch")
    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_build: {torch.version.cuda}")
    print(f"cudnn_enabled: {torch.backends.cudnn.enabled}")
    print(f"cudnn_version: {torch.backends.cudnn.version()}")

    cuda_available = torch.cuda.is_available()
    print(f"cuda_available: {cuda_available}")

    if args.require_cuda and not cuda_available:
        return CheckResult(False, "CUDA is required but torch.cuda.is_available() is False.")

    if not cuda_available:
        # CPU fallback check
        _print_header("CPU Compute")
        x = torch.randn(512, 512)
        y = torch.randn(512, 512)
        z = x @ y
        if not torch.isfinite(z).all().item():
            return CheckResult(False, "CPU matmul produced non-finite values.")
        return CheckResult(True, "Torch CPU check passed.")

    _print_header("CUDA Devices")
    gpu_count = torch.cuda.device_count()
    print(f"gpu_count: {gpu_count}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_memory / (1024**3)
        print(f"gpu[{i}]: name={props.name} capability={props.major}.{props.minor} mem_gb={total_mem_gb:.2f}")

    if gpu_count < args.min_gpus:
        return CheckResult(False, f"Found {gpu_count} GPU(s), but --min-gpus={args.min_gpus}.")

    if args.gpu_index < 0 or args.gpu_index >= gpu_count:
        return CheckResult(False, f"--gpu-index={args.gpu_index} out of range for {gpu_count} GPU(s).")

    _print_header("CUDA Compute")
    supports_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if supports_bf16 else torch.float16
    print(f"compute_dtype: {dtype}")
    print(f"bf16_supported: {supports_bf16}")

    arch_list = set(torch.cuda.get_arch_list())
    def _supports_device(device_idx: int) -> bool:
        cc = torch.cuda.get_device_capability(device_idx)
        return f"sm_{cc[0]}{cc[1]}" in arch_list

    devices_to_check = {args.gpu_index}
    if args.check_all_required_gpus:
        devices_to_check.update(range(min(args.min_gpus, gpu_count)))
    devices_to_check = sorted(devices_to_check)
    print(f"devices_to_check: {devices_to_check}")

    if not args.skip_arch_compat_check:
        incompatible = [idx for idx in devices_to_check if not _supports_device(idx)]
        if incompatible:
            msg = (
                "Arch mismatch warning for GPUs "
                f"{incompatible}; torch_cuda_arch_list={sorted(arch_list)}. "
                "Will continue with runtime compute checks."
            )
            print(msg)
            if args.fail_on_arch_mismatch:
                return CheckResult(False, msg)

    size = max(128, int(args.matmul_size))
    for idx in devices_to_check:
        device = torch.device(f"cuda:{idx}")
        torch.cuda.set_device(device)
        print(f"checking_device: {device}")

        a = torch.randn((size, size), device=device, dtype=dtype)
        b = torch.randn((size, size), device=device, dtype=dtype)
        c = a @ b
        torch.cuda.synchronize(device)
        if not torch.isfinite(c).all().item():
            return CheckResult(False, f"CUDA matmul produced non-finite values on {device}.")

        # Tiny autograd sanity on GPU
        x = torch.randn((4096,), device=device, dtype=torch.float32, requires_grad=True)
        y = (x * x).mean()
        y.backward()
        torch.cuda.synchronize(device)
        if x.grad is None or not torch.isfinite(x.grad).all().item():
            return CheckResult(False, f"CUDA autograd check failed on {device}.")

    return CheckResult(True, "Torch + CUDA checks passed.")


def main() -> int:
    args = _parse_args()
    try:
        result = _run_checks(args)
    except Exception:
        _print_header("Unhandled Exception")
        traceback.print_exc()
        return 1

    _print_header("Result")
    status = "PASS" if result.ok else "FAIL"
    print(f"status: {status}")
    print(f"message: {result.message}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
