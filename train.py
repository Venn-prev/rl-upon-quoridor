"""Hardware-aware entry point for the RTX 3060 Gumbel AlphaZero run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GPU_CONFIG = ROOT / "configs" / "rtx3060_6gb.json"
CHAMPION_CHECKPOINT = ROOT / "checkpoints" / "gen85_best.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "rtx3060_6gb"


def detect() -> tuple[str, str]:
    try:
        import torch
    except ImportError:
        return "cpu", "PyTorch is not installed"
    if not torch.cuda.is_available():
        return "cpu", "CUDA is not available"
    name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return "cuda", f"{name}, {memory_gb:.1f} GB VRAM"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the Gumbel AlphaZero Quoridor model."
    )
    parser.add_argument("--config", type=Path, default=GPU_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--init",
        type=Path,
        help="weights-only starting checkpoint (defaults to gen85_best.pt)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="run on CPU instead of requiring the RTX GPU",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the selected training setup without starting training",
    )
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    device, hardware = detect()
    if args.force_cpu:
        device, hardware = "cpu", "forced by --force-cpu"
    elif device != "cuda" and not args.dry_run:
        print(
            f"CUDA GPU not available ({hardware}). Install the CUDA environment "
            "with `python -m pip install -r requirements.txt`, or use --force-cpu.",
            file=sys.stderr,
        )
        return 2

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"Training config does not exist: {config_path}", file=sys.stderr)
        return 2

    config = json.loads(config_path.read_text(encoding="utf-8"))
    init = args.init
    if init is None and config_path == GPU_CONFIG.resolve():
        init = CHAMPION_CHECKPOINT
    output = args.output.expanduser().resolve()

    print(f"Hardware: {hardware}")
    print(f"Device:   {device}")
    print(f"Config:   {config_path}")
    print(f"Output:   {output}")
    print(
        f"Network:  {config['channels']}x{config['blocks']} SE ResNet, "
        f"batch={config['batch']}, self-play games={config['games']}, "
        f"concurrent={config['concurrent_games']}"
    )
    if init is not None:
        print(f"Initial weights: {init.resolve()}")

    if args.dry_run:
        print("Dry run: training was not started.")
        return 0

    if init is not None and not init.is_file():
        print(f"Initial checkpoint does not exist: {init}", file=sys.stderr)
        return 2

    from quoridor_ai import az_train

    trainer_args = [
        "--config",
        str(config_path),
        "--output",
        str(output),
        "--device",
        device,
    ]
    if init is not None:
        trainer_args.extend(["--init", str(init.resolve())])
    sys.argv = ["az_train", *trainer_args, *extra]
    az_train.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
