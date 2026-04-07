"""
WaveSpeed / stable-diffusion.cpp launcher.
Reads a JSON config (from the GUI) and runs sd.exe with the mapped flags.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from sd_runner import build_sd_command, merge_lora_into_prompt

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_LORAS_DIR = BASE_DIR / "LoRas"


def _print_for_console(msg: str, *, stream=sys.stdout) -> None:
    """Print without crashing on Windows (cp1252) when the line contains Unicode."""
    try:
        print(msg, file=stream, flush=True)
    except UnicodeEncodeError:
        buf = getattr(stream, "buffer", None)
        if buf is not None:
            buf.write((msg + "\n").encode("utf-8", errors="replace"))
            buf.flush()
        else:
            print(msg.encode("ascii", errors="backslashreplace").decode("ascii"), file=stream, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Run sd.exe from a JSON config (GUI).")
    p.add_argument("--config-json", required=True, help="Path to JSON config file.")
    args = p.parse_args()

    path = Path(args.config_json).expanduser()
    if not path.exists():
        _print_for_console(f"[ERROR] Config not found: {path}", stream=sys.stderr)
        return 2

    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    # Merge LoRA tags into prompt
    raw_prompt = (cfg.get("prompt") or "").strip()
    lora_items = cfg.get("lora_items") or []
    pairs: list[tuple[str, float]] = []
    for item in lora_items:
        if not item:
            continue
        pairs.append((str(item["path"]), float(item["weight"])))
    cfg["prompt"] = merge_lora_into_prompt(raw_prompt, pairs)

    options = cfg.setdefault("options", {})
    # Default LoRA directory if we have Loras but no explicit dir from paths
    if pairs and not options.get("lora_model_dir"):
        if DEFAULT_LOCAL_LORAS_DIR.exists():
            options["lora_model_dir"] = str(DEFAULT_LOCAL_LORAS_DIR.resolve())

    try:
        cmd = build_sd_command(cfg)
    except ValueError as e:
        _print_for_console(f"[ERROR] {e}", stream=sys.stderr)
        return 2

    sd = Path(cmd[0])
    if not sd.exists():
        _print_for_console(f"[ERROR] sd.exe not found: {sd}", stream=sys.stderr)
        return 2

    _print_for_console("[INFO] Running: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    proc = subprocess.run(cmd, capture_output=False, text=True)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
