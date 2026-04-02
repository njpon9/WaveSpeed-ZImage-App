"""
Build stable-diffusion.cpp (sd.exe) command lines from a JSON-friendly dict.
Used by zimage_lora_app.py and the GUI.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any


def expand_path(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return str(Path(os.path.expandvars(s)).expanduser())


def build_sd_command(cfg: dict[str, Any]) -> list[str]:
    """
    cfg keys:
      sd_exe (required), output (required), prompt (required)
      options: dict of snake_case option names (see below)
    """
    sd_exe = expand_path(cfg.get("sd_exe"))
    out = expand_path(cfg.get("output"))
    prompt = (cfg.get("prompt") or "").strip()
    if not sd_exe or not out or not prompt:
        raise ValueError("sd_exe, output, and prompt are required")

    o: dict[str, Any] = dict(cfg.get("options") or {})

    cmd: list[str] = [sd_exe]

    m = o.get("mode")
    if m:
        cmd.extend(["-M", str(m)])
    if o.get("verbose", True):
        cmd.append("-v")
    if o.get("color_log"):
        cmd.append("--color")

    # --- model stack ---
    def pair(flag: str, key: str) -> None:
        raw = o.get(key)
        if raw is None or raw is False:
            return
        s = str(raw).strip()
        if not s:
            return
        ep = expand_path(s)
        if ep:
            cmd.extend([flag, ep])

    pair("-m", "model")
    pair("--clip_l", "clip_l")
    pair("--clip_g", "clip_g")
    pair("--clip_vision", "clip_vision")
    pair("--t5xxl", "t5xxl")
    pair("--llm", "llm_path")
    pair("--llm_vision", "llm_vision")
    pair("--diffusion-model", "diffusion_model")
    pair("--high-noise-diffusion-model", "high_noise_diffusion_model")
    pair("--vae", "vae_path")
    pair("--taesd", "taesd")
    pair("--tae", "tae")
    pair("--control-net", "control_net")
    pair("--embd-dir", "embd_dir")
    pair("--lora-model-dir", "lora_model_dir")
    pair("--tensor-type-rules", "tensor_type_rules")
    pair("--photo-maker", "photo_maker")
    pair("--upscale-model", "upscale_model")

    if o.get("threads") is not None and str(o["threads"]).strip() != "":
        cmd.extend(["-t", str(int(o["threads"]))])

    pair("--vae-tile-size", "vae_tile_size")
    if o.get("vae_tile_overlap") is not None and str(o["vae_tile_overlap"]).strip() != "":
        cmd.extend(["--vae-tile-overlap", str(o["vae_tile_overlap"])])
    if o.get("vae_relative_tile_size"):
        cmd.extend(["--vae-relative-tile-size", str(o["vae_relative_tile_size"])])
    if o.get("flow_shift") is not None and str(o["flow_shift"]).strip() != "":
        cmd.extend(["--flow-shift", str(o["flow_shift"])])
    if o.get("type_override"):
        cmd.extend(["--type", str(o["type_override"])])
    if o.get("prediction"):
        cmd.extend(["--prediction", str(o["prediction"])])
    if o.get("lora_apply_mode"):
        cmd.extend(["--lora-apply-mode", str(o["lora_apply_mode"])])
    if o.get("rng"):
        cmd.extend(["--rng", str(o["rng"])])
    if o.get("sampler_rng"):
        cmd.extend(["--sampler-rng", str(o["sampler_rng"])])

    # Booleans
    for key, flag in [
        ("vae_tiling", "--vae-tiling"),
        ("offload_to_cpu", "--offload-to-cpu"),
        ("clip_on_cpu", "--clip-on-cpu"),
        ("vae_on_cpu", "--vae-on-cpu"),
        ("control_net_cpu", "--control-net-cpu"),
        ("diffusion_fa", "--diffusion-fa"),
        ("diffusion_conv_direct", "--diffusion-conv-direct"),
        ("vae_conv_direct", "--vae-conv-direct"),
        ("force_sdxl_vae_conv_scale", "--force-sdxl-vae-conv-scale"),
        ("canny", "--canny"),
        ("taesd_preview_only", "--taesd-preview-only"),
        ("preview_noisy", "--preview-noisy"),
        ("increase_ref_index", "--increase-ref-index"),
        ("disable_auto_resize_ref_image", "--disable-auto-resize-ref-image"),
        ("chroma_disable_dit_mask", "--chroma-disable-dit-mask"),
        ("chroma_enable_t5_mask", "--chroma-enable-t5-mask"),
    ]:
        if o.get(key):
            cmd.append(flag)

    # Preview
    if o.get("preview"):
        cmd.extend(["--preview", str(o["preview"])])
    pair("--preview-path", "preview_path")
    if o.get("preview_interval") is not None and str(o.get("preview_interval")).strip() != "":
        cmd.extend(["--preview-interval", str(int(o["preview_interval"]))])

    # Prompt / gen
    neg = (o.get("negative_prompt") or "").strip()
    cmd.extend(["-p", prompt])
    if neg:
        cmd.extend(["-n", neg])

    pair("-i", "init_img")
    pair("--end-img", "end_img")
    pair("--mask", "mask")
    pair("--control-image", "control_image")
    pair("--control-video", "control_video")
    pair("--pm-id-images-dir", "pm_id_images_dir")
    pair("--pm-id-embed-path", "pm_id_embed_path")

    wv, hv = o.get("width"), o.get("height")
    if wv is not None and int(wv) > 0:
        cmd.extend(["-W", str(int(wv))])
    if hv is not None and int(hv) > 0:
        cmd.extend(["-H", str(int(hv))])
    if o.get("steps") is not None and str(o.get("steps")).strip() != "":
        cmd.extend(["--steps", str(int(o["steps"]))])
    if o.get("high_noise_steps") is not None and str(o["high_noise_steps"]).strip() != "":
        cmd.extend(["--high-noise-steps", str(int(o["high_noise_steps"]))])
    if o.get("clip_skip") is not None and str(o.get("clip_skip")).strip() != "":
        cmd.extend(["--clip-skip", str(int(o["clip_skip"]))])
    if o.get("batch_count") is not None and str(o.get("batch_count")).strip() != "":
        cmd.extend(["-b", str(int(o["batch_count"]))])
    if o.get("video_frames") is not None and str(o.get("video_frames")).strip() != "":
        cmd.extend(["--video-frames", str(int(o["video_frames"]))])
    if o.get("fps") is not None and str(o.get("fps")).strip() != "":
        cmd.extend(["--fps", str(int(o["fps"]))])
    if o.get("timestep_shift") is not None and str(o.get("timestep_shift")).strip() != "":
        cmd.extend(["--timestep-shift", str(int(o["timestep_shift"]))])
    if o.get("upscale_repeats") is not None and str(o.get("upscale_repeats")).strip() != "":
        cmd.extend(["--upscale-repeats", str(int(o["upscale_repeats"]))])
    if o.get("upscale_tile_size") is not None and str(o.get("upscale_tile_size")).strip() != "":
        cmd.extend(["--upscale-tile-size", str(int(o["upscale_tile_size"]))])

    if o.get("cfg_scale") is not None:
        cmd.extend(["--cfg-scale", str(float(o["cfg_scale"]))])
    if o.get("img_cfg_scale") is not None and str(o.get("img_cfg_scale")).strip() != "":
        cmd.extend(["--img-cfg-scale", str(float(o["img_cfg_scale"]))])
    if o.get("guidance") is not None and str(o.get("guidance")).strip() != "":
        cmd.extend(["--guidance", str(float(o["guidance"]))])
    if o.get("slg_scale") is not None and str(o.get("slg_scale")).strip() != "":
        cmd.extend(["--slg-scale", str(float(o["slg_scale"]))])
    if o.get("skip_layer_start") is not None and str(o.get("skip_layer_start")).strip() != "":
        cmd.extend(["--skip-layer-start", str(float(o["skip_layer_start"]))])
    if o.get("skip_layer_end") is not None and str(o.get("skip_layer_end")).strip() != "":
        cmd.extend(["--skip-layer-end", str(float(o["skip_layer_end"]))])
    if o.get("eta") is not None and str(o.get("eta")).strip() != "":
        cmd.extend(["--eta", str(float(o["eta"]))])

    # High-noise mirrors (subset)
    for suffix, flag in [
        ("high_noise_cfg_scale", "--high-noise-cfg-scale"),
        ("high_noise_img_cfg_scale", "--high-noise-img-cfg-scale"),
        ("high_noise_guidance", "--high-noise-guidance"),
        ("high_noise_slg_scale", "--high-noise-slg-scale"),
        ("high_noise_skip_layer_start", "--high-noise-skip-layer-start"),
        ("high_noise_skip_layer_end", "--high-noise-skip-layer-end"),
        ("high_noise_eta", "--high-noise-eta"),
    ]:
        if o.get(suffix) is not None and str(o.get(suffix)).strip() != "":
            cmd.extend([flag, str(float(o[suffix]))])

    if o.get("strength") is not None and str(o.get("strength")).strip() != "":
        cmd.extend(["--strength", str(float(o["strength"]))])
    if o.get("pm_style_strength") is not None and str(o.get("pm_style_strength")).strip() != "":
        cmd.extend(["--pm-style-strength", str(float(o["pm_style_strength"]))])
    if o.get("control_strength") is not None and str(o.get("control_strength")).strip() != "":
        cmd.extend(["--control-strength", str(float(o["control_strength"]))])
    if o.get("moe_boundary") is not None and str(o.get("moe_boundary")).strip() != "":
        cmd.extend(["--moe-boundary", str(float(o["moe_boundary"]))])
    if o.get("vace_strength") is not None and str(o.get("vace_strength")).strip() != "":
        cmd.extend(["--vace-strength", str(float(o["vace_strength"]))])

    if o.get("seed") is not None and str(o.get("seed")).strip() != "":
        cmd.extend(["-s", str(int(o["seed"]))])

    if o.get("sampling_method"):
        cmd.extend(["--sampling-method", str(o["sampling_method"])])
    if o.get("high_noise_sampling_method"):
        cmd.extend(["--high-noise-sampling-method", str(o["high_noise_sampling_method"])])
    if o.get("scheduler"):
        cmd.extend(["--scheduler", str(o["scheduler"])])
    if o.get("sigmas"):
        cmd.extend(["--sigmas", str(o["sigmas"]).strip()])
    if o.get("skip_layers"):
        cmd.extend(["--skip-layers", str(o["skip_layers"]).strip()])
    if o.get("high_noise_skip_layers"):
        cmd.extend(["--high-noise-skip-layers", str(o["high_noise_skip_layers"]).strip()])
    if o.get("easycache"):
        cmd.extend(["--easycache", str(o["easycache"]).strip()])

    # Chroma / nitro (ints from help)
    if o.get("chroma_t5_mask_pad") is not None and str(o.get("chroma_t5_mask_pad")).strip() != "":
        cmd.extend(["--chroma-t5-mask-pad", str(int(o["chroma_t5_mask_pad"]))])

    # ref-image (repeatable)
    refs = o.get("ref_images") or []
    if isinstance(refs, str):
        refs = [line.strip() for line in refs.splitlines() if line.strip()]
    for r in refs:
        rp = expand_path(str(r))
        if rp:
            cmd.extend(["-r", rp])

    # LoRA tags appended to prompt are handled by caller (merged into prompt before this)

    cmd.extend(["-o", out])

    # Extra raw args from user
    extra = o.get("extra_cli", "")
    if isinstance(extra, str) and extra.strip():
        cmd.extend(shlex.split(extra, posix=False))

    return cmd


def merge_lora_into_prompt(prompt: str, lora_items: list[tuple[str, float]]) -> str:
    """lora_items: list of (path_or_name, weight)."""
    parts = [prompt.strip()]
    for ref, w in lora_items:
        if w == 0:
            continue
        p = Path(ref)
        name = p.stem if p.exists() else (Path(ref).stem if ref.lower().endswith((".safetensors", ".ckpt")) else ref)
        parts.append(f"<lora:{name}:{w}>")
    return " ".join(parts).strip()
