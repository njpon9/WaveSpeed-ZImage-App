#!/usr/bin/env python3
"""Validate bundled bin/models/LoRas: T2I + LoRA, inpaint (mask), whole img2img."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
SD = BASE / "bin" / "sd.exe"
MODEL = BASE / "models" / "stable-diffusion" / "z-image-turbo-q8_0.gguf"
AUX = BASE / "models" / "stable-diffusion" / "auxiliary"
LORAS = BASE / "LoRas"
LIGHTSLIDER = LORAS / "zimage_lightslider.safetensors"


def run_cfg(name: str, cfg: dict) -> int:
    tmp = BASE / f"_smoke_{name}.json"
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\n=== {name} ===\n")
    r = subprocess.run([sys.executable, str(BASE / "zimage_lora_app.py"), "--config-json", str(tmp)])
    tmp.unlink(missing_ok=True)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reuse-face",
        action="store_true",
        help="Skip T2I if smoke_test_face.png already exists (faster re-test of inpaint/img2img).",
    )
    args = ap.parse_args()

    if not SD.is_file():
        print("Missing", SD)
        return 1
    if not MODEL.is_file():
        print("Missing", MODEL)
        return 1
    if not LIGHTSLIDER.is_file():
        print("Missing", LIGHTSLIDER)
        return 1

    common = {
        "diffusion_model": str(MODEL),
        "llm_path": str(AUX / "Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf"),
        "vae_path": str(AUX / "ae.safetensors"),
        "lora_model_dir": str(LORAS.resolve()),
        "lora_apply_mode": "auto",
        "width": 512,
        "height": 512,
        "steps": 6,
        "cfg_scale": 1.5,
        "sampling_method": "euler",
        "scheduler": "simple",
        "negative_prompt": "blurry, low quality, deformed",
        "offload_to_cpu": True,
        "vae_on_cpu": True,
        "clip_on_cpu": True,
        "diffusion_fa": True,
        "verbose": True,
    }

    face_out = BASE / "smoke_test_face.png"
    if args.reuse_face and face_out.is_file():
        print("[INFO] Reusing existing", face_out)
    else:
        cfg_t2i = {
            "sd_exe": str(SD),
            "output": str(face_out),
            "prompt": (
                "professional headshot portrait of one adult, front-facing, neutral expression, "
                "natural skin texture, soft studio lighting, sharp eyes, plain gray background, photorealistic"
            ),
            "lora_items": [{"path": str(LIGHTSLIDER), "weight": 0.55}],
            "options": dict(common),
        }
        if run_cfg("t2i_face_lightslider", cfg_t2i) != 0:
            return 2
        if not face_out.is_file():
            print("No output from T2I")
            return 3

    img = Image.open(face_out).convert("RGB")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(mask)
    eye_y = int(h * 0.38)
    lx = int(w * 0.36)
    rx = int(w * 0.64)
    r_eye = max(12, int(min(w, h) * 0.065))
    dr.ellipse((lx - r_eye, eye_y - r_eye, lx + r_eye, eye_y + r_eye), fill=255)
    dr.ellipse((rx - r_eye, eye_y - r_eye, rx + r_eye, eye_y + r_eye), fill=255)
    mask_path = BASE / "smoke_test_face_mask.png"
    mask.save(mask_path)

    face_abs = str(face_out.resolve())
    inpaint_out = BASE / "smoke_test_inpaint_blue_eyes.png"
    cfg_inpaint = {
        "sd_exe": str(SD),
        "output": str(inpaint_out),
        "prompt": (
            "same person, identical face, realistic photograph, "
            "vivid natural blue iris color in both eyes, subtle catchlights, keep skin and pose"
        ),
        "lora_items": [{"path": str(LIGHTSLIDER), "weight": 0.44}],
        "options": {
            **common,
            "width": w,
            "height": h,
            "init_img": face_abs,
            "mask": str(mask_path.resolve()),
            "strength": 1.0,
            "steps": 16,
            "cfg_scale": 1.2,
            "img_cfg_scale": 2.0,
        },
    }
    if run_cfg("inpaint_blue_eyes", cfg_inpaint) != 0:
        return 4

    beach_out = BASE / "smoke_test_beach_bg.png"
    cfg_edit = {
        "sd_exe": str(SD),
        "output": str(beach_out),
        "prompt": (
            "same person, same pose and clothing, photorealistic, "
            "replace background with a bright sunny beach with ocean and sand, shallow depth of field"
        ),
        "lora_items": [{"path": str(LIGHTSLIDER), "weight": 0.44}],
        "options": {
            **common,
            "width": w,
            "height": h,
            "init_img": face_abs,
            "strength": 0.88,
            "steps": 14,
            "cfg_scale": 1.2,
        },
    }
    cfg_edit["options"].pop("mask", None)
    if run_cfg("img2img_beach_background", cfg_edit) != 0:
        return 5

    print("\nAll smoke steps completed OK.\nOutputs:", face_out, inpaint_out, beach_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
