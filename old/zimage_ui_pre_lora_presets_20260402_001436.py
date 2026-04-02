"""
Z-Image / sd.exe launcher GUI — scrollable options, JSON-backed run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageFilter, ImageTk

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "zimage_prompt_history.json"
SECOND_PASS_PRESET_PATH = BASE_DIR / "zimage_second_pass_preset.json"
HISTORY_MAX = 15
SD_EXE_DEFAULT = Path(os.path.expandvars(r"%USERPROFILE%\AppData\Roaming\wavespeed-desktop\sd-bin\sd.exe"))
MODELS_DIR = Path(os.path.expandvars(r"%USERPROFILE%\AppData\Roaming\wavespeed-desktop\models\stable-diffusion"))
LORAS_DIR = BASE_DIR / "LoRas"
AUX_DIR = MODELS_DIR / "auxiliary"

NEGATIVE_DEFAULT = (
    "blurry, bad quality, low resolution, watermark, distorted, ugly, deformed, "
    "extra limbs, poorly drawn, bad anatomy"
)
SAMPLING_METHODS = [
    "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
    "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd",
]
SCHEDULERS = [
    "simple", "discrete", "karras", "exponential", "ays", "gits", "smoothstep",
    "sgm_uniform", "lcm",
]
SPEED_HELP_TEXT = """Fastest possible

- Resolution: 768x768 or 720x720
- Steps: 4
- CFG scale: 1.0
- Sampling: euler
- Scheduler: simple
- diffusion_fa: on
- Low VRAM / offload-to-cpu / clip-on-cpu / vae-on-cpu: off if your GPU can fit the model
- Second pass: off

Balanced quality / speed

- Resolution: 896x896 or 1024x1024
- Steps: 6 to 8
- CFG scale: 1.5 to 2.5
- Sampling: euler or dpm++2m
- Scheduler: simple or karras
- diffusion_fa: on
- Low VRAM / CPU offload: off if possible
- Second pass: only when needed, with 4 to 6 steps and strength around 0.25 to 0.40

Low VRAM

- Resolution: 640x640 to 768x768
- Steps: 4 to 6
- CFG scale: 1.0 to 2.0
- Sampling: euler
- Scheduler: simple
- Low VRAM: on
- offload-to-cpu / clip-on-cpu / vae-on-cpu: on as needed
- VAE tiling: on only if decode runs out of memory
- Second pass: off unless necessary

General tips

- Lower width and height first for the biggest speedup.
- CPU offload helps memory but usually slows generation.
- Keep steps low before changing sampler settings.
- If you use the automated second pass, keep its strength moderate so it refines instead of redoing the whole image.
"""

# Optional numeric string fields: empty => omit
INT_OPTION_KEYS = frozenset({
    "threads", "high_noise_steps", "clip_skip", "batch_count", "video_frames", "fps",
    "timestep_shift", "upscale_repeats", "upscale_tile_size", "preview_interval",
    "chroma_t5_mask_pad",
})
FLOAT_OPTION_KEYS = frozenset({
    "img_cfg_scale", "guidance", "slg_scale", "skip_layer_start", "skip_layer_end", "eta",
    "high_noise_cfg_scale", "high_noise_img_cfg_scale", "high_noise_guidance",
    "high_noise_slg_scale", "high_noise_skip_layer_start", "high_noise_skip_layer_end",
    "high_noise_eta", "pm_style_strength", "control_strength", "moe_boundary", "vace_strength",
})


def list_models() -> list[tuple[str, str]]:
    if not MODELS_DIR.exists():
        return [("z_image_turbo-Q4_K.gguf", str(MODELS_DIR / "z_image_turbo-Q4_K.gguf"))]
    found = sorted(MODELS_DIR.glob("z_image_turbo-*.gguf"))
    if not found:
        return [("z_image_turbo-Q4_K.gguf", str(MODELS_DIR / "z_image_turbo-Q4_K.gguf"))]
    return [(p.name, str(p)) for p in found]


def list_all_diffusion_models() -> list[tuple[str, str]]:
    """
    Include canonical z-image models plus additional UNet models.
    Labels are relative to MODELS_DIR when possible.
    """
    exts = {".gguf", ".safetensors", ".ckpt"}
    candidates: list[Path] = []
    if MODELS_DIR.exists():
        candidates.extend(p for p in MODELS_DIR.glob("*") if p.is_file() and p.suffix.lower() in exts)
        unet_dir = MODELS_DIR / "unet"
        if unet_dir.exists():
            candidates.extend(p for p in unet_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not candidates:
        return [("z_image_turbo-Q4_K.gguf", str(MODELS_DIR / "z_image_turbo-Q4_K.gguf"))]

    dedup: dict[str, Path] = {}
    for p in candidates:
        dedup[str(p.resolve())] = p
    out: list[tuple[str, str]] = []
    for p in sorted(dedup.values(), key=lambda x: x.name.lower()):
        try:
            label = str(p.relative_to(MODELS_DIR))
        except ValueError:
            label = p.name
        out.append((label, str(p)))
    return out


def list_loras() -> list[str]:
    if not LORAS_DIR.exists():
        return [""]
    names = sorted(p.name for p in LORAS_DIR.glob("*.safetensors"))
    return names if names else [""]


def default_llm() -> str:
    return str(AUX_DIR / "Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf")


def default_vae() -> str:
    return str(AUX_DIR / "ae.safetensors")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Z-Image Local UI — sd.exe")
        self.geometry("1200x860")

        self._proc: subprocess.Popen | None = None
        self._run_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._active_phase = ""
        self._history: list[dict] = []
        self.lora_rows: list[dict] = []
        self.edit_lora_rows: list[dict] = []
        self.second_pass_lora_rows: list[dict] = []
        self._str_opts: dict[str, tk.StringVar] = {}
        self._bool_opts: dict[str, tk.BooleanVar] = {}
        self._edit_canvas_image: Image.Image | None = None
        self._edit_canvas_mask: Image.Image | None = None
        self._edit_canvas_mask_draw: ImageDraw.ImageDraw | None = None
        self._edit_canvas_preview: ImageTk.PhotoImage | None = None
        self._edit_canvas_base_preview: ImageTk.PhotoImage | None = None
        self._edit_canvas_scale = 1.0
        self._edit_canvas_zoom = 1.0
        self._edit_canvas_source_path = ""
        self._edit_mask_dirty = False
        self._edit_mask_temp_path = ""
        self._edit_mask_history: list[Image.Image] = []
        self._edit_mask_stroke_active = False

        models = list_models()
        all_models = list_all_diffusion_models()
        loras = list_loras()

        self.model_var = tk.StringVar(value=models[0][0])
        self.model_path_var = tk.StringVar(value=models[0][1])
        self._model_map = {name: path for name, path in all_models}
        self._all_model_map = {name: path for name, path in all_models}
        self.edit_model_var = tk.StringVar(value=models[0][0])

        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        left_outer = ttk.Frame(root, width=480)
        left_outer.pack(side=tk.LEFT, fill=tk.BOTH)
        left_outer.pack_propagate(False)

        # Defaults so “Simple run” matches the legacy app without opening Advanced
        self.sd_exe_var = tk.StringVar(value=str(SD_EXE_DEFAULT))
        self.verbose_var = tk.BooleanVar(value=True)
        self.color_log_var = tk.BooleanVar(value=False)
        self.high_noise_sampling_var = tk.StringVar(value="")
        self._str_opt("llm_path", default_llm())
        self._str_opt("vae_path", default_vae())
        self._str_opt("lora_model_dir", str(LORAS_DIR.resolve()) if LORAS_DIR.exists() else "")
        # Legacy zimage_lora_app defaults: always --offload-to-cpu, --diffusion-fa, --lora-apply-mode auto
        self._bool_opt("offload_to_cpu", True)
        self._bool_opt("diffusion_fa", True)
        self._str_opt("lora_apply_mode", "auto")

        self.left_notebook = ttk.Notebook(left_outer)
        self.left_notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_simple = ttk.Frame(self.left_notebook)
        self.tab_adv = ttk.Frame(self.left_notebook)
        self.tab_edit = ttk.Frame(self.left_notebook)
        self.left_notebook.add(self.tab_simple, text="Simple run")
        self.left_notebook.add(self.tab_adv, text="Advanced")
        self.left_notebook.add(self.tab_edit, text="Image edit")

        self._build_simple_tab(self.tab_simple, all_models, loras, all_models)
        self._build_advanced_tab(self.tab_adv)
        self._build_edit_tab(self.tab_edit, all_models)

        self.tab_history = ttk.Frame(self.left_notebook)
        self.left_notebook.add(self.tab_history, text="History")
        self._build_history_tab(self.tab_history)
        self._load_history_from_disk()
        self._load_second_pass_preset()

        run_row = ttk.Frame(left_outer)
        run_row.pack(fill=tk.X, pady=(10, 0))
        self.run_btn = ttk.Button(run_row, text="Run", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))
        self.stop_btn = ttk.Button(run_row, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))
        self.help_btn = ttk.Button(run_row, text="Help", command=self.show_speed_help)
        self.help_btn.pack(side=tk.LEFT)

        right = ttk.Frame(root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))
        ttk.Label(right, text="Log").pack(anchor="w")
        self.log = tk.Text(right, wrap="none", padx=6, pady=6)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.status = tk.StringVar(value="Idle.")
        ttk.Label(right, textvariable=self.status).pack(anchor="w", pady=(4, 0))

    def _str_opt(self, key: str, default: str = "") -> tk.StringVar:
        if key not in self._str_opts:
            self._str_opts[key] = tk.StringVar(value=default)
        return self._str_opts[key]

    def _bool_opt(self, key: str, default: bool = False) -> tk.BooleanVar:
        if key not in self._bool_opts:
            self._bool_opts[key] = tk.BooleanVar(value=default)
        return self._bool_opts[key]

    def _row_entry(self, parent: ttk.Widget, label: str, key: str, default: str = "", browse: str | None = None) -> None:
        fr = ttk.Frame(parent)
        fr.pack(fill=tk.X, pady=2)
        ttk.Label(fr, text=label, width=24).pack(side=tk.LEFT, anchor="nw")
        var = self._str_opt(key, default)
        ttk.Entry(fr, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        if browse == "file":
            ttk.Button(fr, text="…", width=3, command=lambda: self._browse_file(var)).pack(side=tk.LEFT, padx=(4, 0))
        elif browse == "dir":
            ttk.Button(fr, text="…", width=3, command=lambda: self._browse_dir(var)).pack(side=tk.LEFT, padx=(4, 0))

    def _browse_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _browse_dir(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _bind_canvas_mousewheel(self, canvas: tk.Canvas) -> None:
        def wheel(e: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        def bind_wheel(_e: tk.Event | None = None) -> None:
            canvas.bind_all("<MouseWheel>", wheel)

        def unbind_wheel(_e: tk.Event | None = None) -> None:
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", bind_wheel)
        canvas.bind("<Leave>", unbind_wheel)

    def _build_simple_tab(self, tab: ttk.Frame, models: list, loras: list[str], all_models: list[tuple[str, str]]) -> None:
        scroll_wrap = ttk.Frame(tab)
        scroll_wrap.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(scroll_wrap, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(scroll_wrap, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas, padding=(8, 8, 8, 4))
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _sync_simple(_e=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _sync_simple)

        def _fill_simple_width(e) -> None:
            canvas.itemconfigure(win_id, width=max(e.width - 4, 1))

        canvas.bind("<Configure>", _fill_simple_width)
        self._bind_canvas_mousewheel(canvas)

        q = ttk.LabelFrame(inner, text="Model & output", padding=8)
        q.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(q, text="Diffusion / UNet model").pack(anchor="w")
        model_cb = ttk.Combobox(q, state="readonly", values=[m[0] for m in models], textvariable=self.model_var, width=40)
        model_cb.pack(fill=tk.X)
        model_cb.bind("<<ComboboxSelected>>", lambda _e: self.model_path_var.set(self._model_map.get(self.model_var.get(), self.model_path_var.get())))
        ttk.Label(
            q,
            text="Includes z-image models plus any additional diffusion or UNet models discovered in the unet folder.",
            wraplength=420,
        ).pack(anchor="w", pady=(6, 0))

        self.out_var = tk.StringVar(value=str(BASE_DIR / "zimage_output.png"))
        ofr = ttk.Frame(q)
        ofr.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(ofr, text="Output image").pack(side=tk.LEFT)
        ttk.Entry(ofr, textvariable=self.out_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        ttk.Button(ofr, text="Browse…", command=self.pick_output).pack(side=tk.LEFT)
        ttk.Label(q, text="Uses WaveSpeed llm/vae paths and sd.exe by default (see Advanced).", wraplength=420).pack(anchor="w", pady=(8, 0))

        pr = ttk.LabelFrame(inner, text="Prompts", padding=8)
        pr.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(pr, text="Prompt").pack(anchor="w")
        self.prompt = self.make_resizable_text(pr, height=4, width=44)
        self.prompt.insert("1.0", "portrait lighting, cinematic, ultra detailed")
        ttk.Label(pr, text="Negative prompt").pack(anchor="w", pady=(8, 0))
        self.neg_prompt = self.make_resizable_text(pr, height=3, width=44)
        self.neg_prompt.insert("1.0", NEGATIVE_DEFAULT)

        lr = ttk.LabelFrame(inner, text="LoRA (merged into prompt)", padding=8)
        lr.pack(fill=tk.X, pady=(0, 8))
        self.lora_container = ttk.Frame(lr)
        self.lora_container.pack(fill=tk.X, pady=(4, 0))
        self._lora_names = loras
        lora_btns = ttk.Frame(lr)
        lora_btns.pack(fill=tk.X, pady=(6, 0))
        self.add_lora_btn = ttk.Button(lora_btns, text="Add LoRA", command=lambda: self.add_lora_row(self.lora_rows[-1]["name_var"].get(), 0.8))
        self.add_lora_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.remove_lora_btn = ttk.Button(lora_btns, text="Remove last", command=self.remove_last_lora_row, state=tk.DISABLED)
        self.remove_lora_btn.pack(side=tk.LEFT)
        self.add_lora_row(default_name="zimage_lightslider.safetensors" if "zimage_lightslider.safetensors" in loras else loras[0], default_weight=0.8)

        sz = ttk.LabelFrame(inner, text="Size & presets", padding=8)
        sz.pack(fill=tk.X, pady=(0, 8))
        size_row = ttk.Frame(sz)
        size_row.pack(fill=tk.X)
        self.width_var = tk.IntVar(value=1024)
        self.height_var = tk.IntVar(value=1024)
        ttk.Label(size_row, text="W").pack(side=tk.LEFT)
        ttk.Entry(size_row, textvariable=self.width_var, width=8).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(size_row, text="H").pack(side=tk.LEFT)
        ttk.Entry(size_row, textvariable=self.height_var, width=8).pack(side=tk.LEFT, padx=(4, 0))
        preset_row = ttk.Frame(sz)
        preset_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(preset_row, text="Presets:").pack(side=tk.LEFT)
        ttk.Button(preset_row, text="1:1", command=lambda: self.set_dimensions(1024, 1024)).pack(side=tk.LEFT, padx=(6, 4))
        ttk.Button(preset_row, text="16:9 720p", command=lambda: self.set_dimensions(1280, 720)).pack(side=tk.LEFT, padx=4)
        ttk.Button(preset_row, text="9:16", command=lambda: self.set_dimensions(720, 1280)).pack(side=tk.LEFT, padx=4)
        ttk.Button(preset_row, text="4:3", command=lambda: self.set_dimensions(1024, 768)).pack(side=tk.LEFT, padx=4)

        sm = ttk.LabelFrame(inner, text="Sampling", padding=8)
        sm.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(sm, text="Steps").pack(anchor="w")
        self.steps_var = tk.IntVar(value=4)
        self.steps_lbl = tk.StringVar(value="4")
        steps_row = ttk.Frame(sm)
        steps_row.pack(fill=tk.X)
        ttk.Scale(steps_row, from_=1, to=50, orient="horizontal", variable=self.steps_var, command=self.on_steps).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(steps_row, textvariable=self.steps_lbl, width=5).pack(side=tk.LEFT, padx=(6, 0))
        self.steps_entry_var = tk.StringVar(value="4")
        se = ttk.Entry(steps_row, textvariable=self.steps_entry_var, width=6)
        se.pack(side=tk.LEFT, padx=(6, 0))
        se.bind("<Return>", self.on_steps_entry)
        se.bind("<FocusOut>", self.on_steps_entry)

        ttk.Label(sm, text="CFG scale").pack(anchor="w", pady=(6, 0))
        self.cfg_var = tk.DoubleVar(value=1.0)
        self.cfg_lbl = tk.StringVar(value="1.00")
        cfg_row = ttk.Frame(sm)
        cfg_row.pack(fill=tk.X)
        ttk.Scale(cfg_row, from_=1.0, to=20.0, orient="horizontal", variable=self.cfg_var, command=self.on_cfg).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(cfg_row, textvariable=self.cfg_lbl, width=5).pack(side=tk.LEFT, padx=(6, 0))
        self.cfg_entry_var = tk.StringVar(value="1.00")
        ce = ttk.Entry(cfg_row, textvariable=self.cfg_entry_var, width=6)
        ce.pack(side=tk.LEFT, padx=(6, 0))
        ce.bind("<Return>", self.on_cfg_entry)
        ce.bind("<FocusOut>", self.on_cfg_entry)

        ttk.Label(sm, text="Sampling method").pack(anchor="w", pady=(6, 0))
        self.sampling_var = tk.StringVar(value="euler")
        ttk.Combobox(sm, state="readonly", values=SAMPLING_METHODS, textvariable=self.sampling_var, width=38).pack(fill=tk.X)

        ttk.Label(sm, text="Scheduler").pack(anchor="w", pady=(6, 0))
        self.scheduler_var = tk.StringVar(value="simple")
        ttk.Combobox(sm, state="readonly", values=SCHEDULERS, textvariable=self.scheduler_var, width=38).pack(fill=tk.X)

        ttk.Label(sm, text="Seed (optional)").pack(anchor="w", pady=(6, 0))
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(sm, textvariable=self.seed_var).pack(fill=tk.X)

        i2 = ttk.LabelFrame(inner, text="Image-to-image", padding=8)
        i2.pack(fill=tk.X, pady=(0, 8))
        self.img2img_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(i2, text="Enable init image (--init-img)", variable=self.img2img_enabled, onvalue=True, offvalue=False).pack(anchor="w")
        init_row = ttk.Frame(i2)
        init_row.pack(fill=tk.X, pady=(4, 0))
        self.init_img_var = tk.StringVar(value="")
        ttk.Entry(init_row, textvariable=self.init_img_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(init_row, text="Pick image…", command=self.pick_init_image).pack(side=tk.LEFT)
        ttk.Label(i2, text="Strength").pack(anchor="w", pady=(6, 0))
        strength_row = ttk.Frame(i2)
        strength_row.pack(fill=tk.X)
        self.strength_var = tk.DoubleVar(value=0.75)
        self.strength_lbl = tk.StringVar(value="0.75")
        ttk.Scale(strength_row, from_=0.0, to=1.0, orient="horizontal", variable=self.strength_var, command=self.on_strength).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(strength_row, textvariable=self.strength_lbl, width=6).pack(side=tk.LEFT, padx=(6, 0))

        sp = ttk.LabelFrame(inner, text="Automated second pass", padding=8)
        sp.pack(fill=tk.X, pady=(0, 8))
        self.second_pass_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            sp,
            text="Enable automated second pass after Simple run",
            variable=self.second_pass_enabled,
            onvalue=True,
            offvalue=False,
        ).pack(anchor="w")
        ttk.Label(
            sp,
            text="Pass 1 generates the base image. Pass 2 automatically reuses that image as --init-img and a reference, then saves a separate refined output.",
            wraplength=420,
        ).pack(anchor="w", pady=(4, 0))

        spb = ttk.Frame(sp)
        spb.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(spb, text="Save preset", command=lambda: self.save_second_pass_preset(True)).pack(side=tk.LEFT)
        ttk.Button(spb, text="Reload saved", command=self._load_second_pass_preset).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(spb, text="Output suffix").pack(side=tk.LEFT, padx=(12, 4))
        self.second_pass_suffix_var = tk.StringVar(value="_realism")
        ttk.Entry(spb, textvariable=self.second_pass_suffix_var, width=14).pack(side=tk.LEFT)

        ttk.Label(sp, text="Edit model").pack(anchor="w", pady=(8, 0))
        self.second_pass_model_var = tk.StringVar(value=models[0][0])
        ttk.Combobox(
            sp,
            state="readonly",
            values=[m[0] for m in all_models],
            textvariable=self.second_pass_model_var,
            width=38,
        ).pack(fill=tk.X)
        self.second_pass_inherit_base_settings = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            sp,
            text="Preserve pass 1 model, size, sampler, and global stack for pass 2 stability",
            variable=self.second_pass_inherit_base_settings,
        ).pack(anchor="w", pady=(6, 0))
        self.second_pass_use_ref_image = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            sp,
            text="Also pass the pass 1 image as a reference image (-r)",
            variable=self.second_pass_use_ref_image,
        ).pack(anchor="w")

        ttk.Label(sp, text="Second-pass prompt").pack(anchor="w", pady=(8, 0))
        self.second_pass_prompt = self.make_resizable_text(sp, height=4, width=44)
        self.second_pass_prompt.insert("1.0", "add realism, natural skin texture, lifelike lighting, photographic detail")
        ttk.Label(sp, text="Second-pass negative prompt").pack(anchor="w", pady=(8, 0))
        self.second_pass_negative = self.make_resizable_text(sp, height=3, width=44)
        self.second_pass_negative.insert("1.0", NEGATIVE_DEFAULT)

        sps = ttk.LabelFrame(sp, text="Second-pass settings", padding=8)
        sps.pack(fill=tk.X, pady=(8, 0))
        self.second_pass_width = tk.IntVar(value=1024)
        self.second_pass_height = tk.IntVar(value=1024)
        wh2 = ttk.Frame(sps)
        wh2.pack(fill=tk.X)
        ttk.Label(wh2, text="W").pack(side=tk.LEFT)
        ttk.Entry(wh2, textvariable=self.second_pass_width, width=8).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(wh2, text="H").pack(side=tk.LEFT)
        ttk.Entry(wh2, textvariable=self.second_pass_height, width=8).pack(side=tk.LEFT, padx=(4, 0))
        self.second_pass_steps = tk.IntVar(value=6)
        self.second_pass_cfg = tk.DoubleVar(value=1.5)
        self.second_pass_sampling = tk.StringVar(value="euler")
        self.second_pass_scheduler = tk.StringVar(value="simple")
        self.second_pass_strength = tk.DoubleVar(value=0.35)
        self.second_pass_strength_lbl = tk.StringVar(value="0.35")
        ttk.Label(sps, text="Steps").pack(anchor="w", pady=(8, 0))
        ttk.Entry(sps, textvariable=self.second_pass_steps, width=8).pack(anchor="w")
        ttk.Label(sps, text="CFG scale").pack(anchor="w", pady=(8, 0))
        ttk.Entry(sps, textvariable=self.second_pass_cfg, width=8).pack(anchor="w")
        ttk.Label(sps, text="Sampling").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(sps, state="readonly", values=SAMPLING_METHODS, textvariable=self.second_pass_sampling, width=30).pack(fill=tk.X)
        ttk.Label(sps, text="Scheduler").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(sps, state="readonly", values=SCHEDULERS, textvariable=self.second_pass_scheduler, width=30).pack(fill=tk.X)
        ttk.Label(sps, text="Strength").pack(anchor="w", pady=(8, 0))
        sp_strength_row = ttk.Frame(sps)
        sp_strength_row.pack(fill=tk.X)
        ttk.Scale(
            sp_strength_row,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.second_pass_strength,
            command=self.on_second_pass_strength,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(sp_strength_row, textvariable=self.second_pass_strength_lbl, width=6).pack(side=tk.LEFT, padx=(6, 0))

        spl = ttk.LabelFrame(sp, text="Second-pass LoRA (merged into prompt)", padding=8)
        spl.pack(fill=tk.X, pady=(8, 0))
        self.second_pass_lora_container = ttk.Frame(spl)
        self.second_pass_lora_container.pack(fill=tk.X, pady=(4, 0))
        sp_lora_btns = ttk.Frame(spl)
        sp_lora_btns.pack(fill=tk.X, pady=(6, 0))
        self.add_second_pass_lora_btn = ttk.Button(
            sp_lora_btns,
            text="Add LoRA",
            command=lambda: self.add_second_pass_lora_row(
                self.second_pass_lora_rows[-1]["name_var"].get() if self.second_pass_lora_rows else "",
                0.8,
            ),
        )
        self.add_second_pass_lora_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.remove_second_pass_lora_btn = ttk.Button(
            sp_lora_btns,
            text="Remove last",
            command=self.remove_last_second_pass_lora_row,
            state=tk.DISABLED,
        )
        self.remove_second_pass_lora_btn.pack(side=tk.LEFT)
        self.add_second_pass_lora_row(default_name="", default_weight=0.0)

        ttk.Label(
            sp,
            text="Speed tips: lower width/height first, keep steps low, keep Euler/simple and diffusion_fa on, and disable CPU offload if your GPU can already fit the run.",
            wraplength=420,
        ).pack(anchor="w", pady=(8, 0))

        vm = ttk.LabelFrame(inner, text="VRAM", padding=8)
        vm.pack(fill=tk.X, pady=(0, 8))
        self.low_vram = tk.BooleanVar(value=True)
        self.vae_tiling = tk.BooleanVar(value=False)
        ttk.Checkbutton(vm, text="Low VRAM (CLIP on CPU)", variable=self.low_vram).pack(anchor="w")
        ttk.Checkbutton(vm, text="VAE tiling (optional; can look blocky)", variable=self.vae_tiling).pack(anchor="w")
        ttk.Label(vm, text="Legacy WaveSpeed launcher also used --offload-to-cpu, --diffusion-fa, and --lora-apply-mode auto (defaults on; see Advanced).", wraplength=420).pack(anchor="w", pady=(6, 0))

    def _build_advanced_tab(self, tab: ttk.Frame) -> None:
        scroll_wrap = ttk.Frame(tab)
        scroll_wrap.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(scroll_wrap, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(scroll_wrap, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas, padding=(8, 8, 8, 0))
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _sync_adv(_e=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _sync_adv)

        def _fill_adv_width(e) -> None:
            canvas.itemconfigure(win_id, width=max(e.width - 4, 1))

        canvas.bind("<Configure>", _fill_adv_width)
        self._bind_canvas_mousewheel(canvas)

        ex = ttk.LabelFrame(inner, text="Executable & logging", padding=8)
        ex.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(ex, text="sd.exe").pack(anchor="w")
        sdfr = ttk.Frame(ex)
        sdfr.pack(fill=tk.X)
        ttk.Entry(sdfr, textvariable=self.sd_exe_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(sdfr, text="Browse…", width=10, command=self.pick_sd_exe).pack(side=tk.LEFT, padx=(6, 0))
        self._row_entry(ex, "Pipeline mode (-M)", "mode", "")
        ttk.Checkbutton(ex, text="Verbose (-v)", variable=self.verbose_var).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(ex, text="Color log (--color)", variable=self.color_log_var).pack(anchor="w")

        es = ttk.LabelFrame(inner, text="Extended sampling & schedules", padding=8)
        es.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(es, text="High-noise sampling method").pack(anchor="w")
        ttk.Combobox(es, values=[""] + SAMPLING_METHODS, textvariable=self.high_noise_sampling_var, width=38).pack(fill=tk.X)
        ttk.Label(es, text="Sigmas / layers / cache (optional)").pack(anchor="w", pady=(6, 0))
        self._row_entry(es, "Sigmas", "sigmas", "")
        self._row_entry(es, "Skip layers", "skip_layers", "")
        self._row_entry(es, "High-noise skip layers", "high_noise_skip_layers", "")
        self._row_entry(es, "Easycache", "easycache", "")

        vmx = ttk.LabelFrame(inner, text="VRAM / CPU (extra)", padding=8)
        vmx.pack(fill=tk.X, pady=(0, 8))
        self._bool_chk(vmx, "Offload diffusion to CPU", "offload_to_cpu")
        self._bool_chk(vmx, "VAE on CPU", "vae_on_cpu")
        self._bool_chk(vmx, "ControlNet on CPU", "control_net_cpu")
        self._row_entry(vmx, "Thread count (-t)", "threads", "")
        self._row_entry(vmx, "VAE tile size", "vae_tile_size", "")
        self._row_entry(vmx, "VAE tile overlap", "vae_tile_overlap", "")
        self._row_entry(vmx, "VAE relative tile size", "vae_relative_tile_size", "")
        self._row_entry(vmx, "Flow shift", "flow_shift", "")

        ms = ttk.LabelFrame(inner, text="Model stack (paths)", padding=8)
        ms.pack(fill=tk.X, pady=(0, 8))
        self._row_entry(ms, "LLM (--llm)", "llm_path", default_llm(), browse="file")
        self._row_entry(ms, "LLM vision", "llm_vision", "", browse="file")
        self._row_entry(ms, "VAE (--vae)", "vae_path", default_vae(), browse="file")
        self._row_entry(ms, "Main -m weights", "model", "", browse="file")
        self._row_entry(ms, "CLIP-L", "clip_l", "", browse="file")
        self._row_entry(ms, "CLIP-G", "clip_g", "", browse="file")
        self._row_entry(ms, "CLIP vision", "clip_vision", "", browse="file")
        self._row_entry(ms, "T5-XXL", "t5xxl", "", browse="file")
        self._row_entry(ms, "High-noise diffusion", "high_noise_diffusion_model", "", browse="file")
        self._row_entry(ms, "TAESD", "taesd", "", browse="file")
        self._row_entry(ms, "TAE", "tae", "", browse="file")
        self._row_entry(ms, "LoRA search dir", "lora_model_dir", str(LORAS_DIR.resolve()) if LORAS_DIR.exists() else "", browse="dir")
        self._row_entry(ms, "Embeddings dir", "embd_dir", "", browse="dir")
        self._row_entry(ms, "Tensor type rules file", "tensor_type_rules", "", browse="file")
        self._row_entry(ms, "PhotoMaker", "photo_maker", "", browse="file")
        self._row_entry(ms, "Upscale model", "upscale_model", "", browse="file")
        self._row_entry(ms, "Type override (--type)", "type_override", "")
        self._row_entry(ms, "Prediction", "prediction", "")
        self._row_entry(ms, "LoRA apply mode", "lora_apply_mode", "")

        gx = ttk.LabelFrame(inner, text="Generation extras", padding=8)
        gx.pack(fill=tk.X, pady=(0, 8))
        self._row_entry(gx, "High-noise steps", "high_noise_steps", "")
        self._row_entry(gx, "CLIP skip", "clip_skip", "")
        self._row_entry(gx, "Batch count (-b)", "batch_count", "")
        self._row_entry(gx, "Video frames", "video_frames", "")
        self._row_entry(gx, "FPS", "fps", "")
        self._row_entry(gx, "Timestep shift", "timestep_shift", "")
        self._row_entry(gx, "Upscale repeats", "upscale_repeats", "")
        self._row_entry(gx, "Upscale tile size", "upscale_tile_size", "")
        for lbl, k in [
            ("Image CFG scale", "img_cfg_scale"),
            ("Guidance", "guidance"),
            ("SLG scale", "slg_scale"),
            ("Skip layer start", "skip_layer_start"),
            ("Skip layer end", "skip_layer_end"),
            ("Eta (noise)", "eta"),
            ("H.N. CFG scale", "high_noise_cfg_scale"),
            ("H.N. img CFG", "high_noise_img_cfg_scale"),
            ("H.N. guidance", "high_noise_guidance"),
            ("H.N. SLG scale", "high_noise_slg_scale"),
            ("H.N. skip layer start", "high_noise_skip_layer_start"),
            ("H.N. skip layer end", "high_noise_skip_layer_end"),
            ("H.N. eta", "high_noise_eta"),
            ("PhotoMaker style str.", "pm_style_strength"),
            ("Control strength", "control_strength"),
            ("MoE boundary", "moe_boundary"),
            ("VACE strength", "vace_strength"),
        ]:
            self._row_entry(gx, lbl, k, "")

        ct = ttk.LabelFrame(inner, text="Control / masks / extra I/O", padding=8)
        ct.pack(fill=tk.X, pady=(0, 8))
        self._row_entry(ct, "ControlNet weights", "control_net", "", browse="file")
        self._row_entry(ct, "Control image", "control_image", "", browse="file")
        self._row_entry(ct, "Control video", "control_video", "", browse="file")
        self._row_entry(ct, "Inpaint / mask", "mask", "", browse="file")
        self._row_entry(ct, "End frame image", "end_img", "", browse="file")
        self._row_entry(ct, "PhotoMaker ID images dir", "pm_id_images_dir", "", browse="dir")
        self._row_entry(ct, "PhotoMaker ID embed", "pm_id_embed_path", "", browse="file")

        prg = ttk.LabelFrame(inner, text="Preview & RNG", padding=8)
        prg.pack(fill=tk.X, pady=(0, 8))
        self._row_entry(prg, "Preview mode", "preview", "")
        self._row_entry(prg, "Preview output path", "preview_path", "", browse="file")
        self._row_entry(prg, "Preview interval", "preview_interval", "")
        self._row_entry(prg, "RNG type", "rng", "")
        self._row_entry(prg, "Sampler RNG", "sampler_rng", "")
        self._row_entry(prg, "Chroma T5 mask pad", "chroma_t5_mask_pad", "")

        rf = ttk.LabelFrame(inner, text="Reference images (-r, one path per line)", padding=8)
        rf.pack(fill=tk.X, pady=(0, 8))
        self.ref_images_txt = tk.Text(rf, height=4, width=44, wrap="word")
        self.ref_images_txt.pack(fill=tk.X)

        xc = ttk.LabelFrame(inner, text="Extra CLI", padding=8)
        xc.pack(fill=tk.X, pady=(0, 8))
        self._row_entry(xc, "Extra arguments", "extra_cli", "")

        pf = ttk.LabelFrame(inner, text="sd.cpp performance / misc flags", padding=8)
        pf.pack(fill=tk.X, pady=(0, 8))
        grid = ttk.Frame(pf)
        grid.pack(fill=tk.X)
        col1 = ttk.Frame(grid)
        col2 = ttk.Frame(grid)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pairs = [
            ("Diffusion FA", "diffusion_fa"),
            ("Diffusion conv direct", "diffusion_conv_direct"),
            ("VAE conv direct", "vae_conv_direct"),
            ("Force SDXL VAE conv scale", "force_sdxl_vae_conv_scale"),
            ("Canny", "canny"),
            ("TAESD preview only", "taesd_preview_only"),
            ("Preview noisy", "preview_noisy"),
            ("Increase ref index", "increase_ref_index"),
            ("Disable auto-resize ref", "disable_auto_resize_ref_image"),
            ("Chroma disable DiT mask", "chroma_disable_dit_mask"),
            ("Chroma enable T5 mask", "chroma_enable_t5_mask"),
        ]
        for i, (lbl, k) in enumerate(pairs):
            parent = col1 if i % 2 == 0 else col2
            ttk.Checkbutton(parent, text=lbl, variable=self._bool_opt(k)).pack(anchor="w")

    def _build_edit_tab(self, tab: ttk.Frame, models: list[tuple[str, str]]) -> None:
        scroll_wrap = ttk.Frame(tab)
        scroll_wrap.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(scroll_wrap, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(scroll_wrap, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas, padding=(8, 8, 8, 4))
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _sync_edit(_e=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _sync_edit)

        def _fill_edit_width(e) -> None:
            canvas.itemconfigure(win_id, width=max(e.width - 4, 1))

        canvas.bind("<Configure>", _fill_edit_width)
        self._bind_canvas_mousewheel(canvas)

        mm = ttk.LabelFrame(inner, text="Edit model", padding=8)
        mm.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(mm, text="Diffusion / UNet model").pack(anchor="w")
        names = [m[0] for m in models]
        mcb = ttk.Combobox(mm, state="readonly", values=names, textvariable=self.edit_model_var, width=46)
        mcb.pack(fill=tk.X)
        self.edit_mode_var = tk.StringVar(value="whole")
        self.edit_preset_var = tk.StringVar(value="Balanced")
        self.edit_seed_var = tk.StringVar(value="")
        mode_row = ttk.Frame(mm)
        mode_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(mode_row, text="Workflow").pack(side=tk.LEFT)
        self.edit_mode_menu = ttk.Combobox(
            mode_row,
            state="readonly",
            values=["Whole Image Edit", "Masked Edit"],
            width=20,
        )
        self.edit_mode_menu.pack(side=tk.LEFT, padx=(8, 12))
        self.edit_mode_menu.set("Whole Image Edit")
        self.edit_mode_menu.bind("<<ComboboxSelected>>", self.on_edit_mode_changed)
        ttk.Label(mode_row, text="Preset").pack(side=tk.LEFT)
        self.edit_preset_menu = ttk.Combobox(
            mode_row,
            state="readonly",
            values=["Subtle", "Balanced", "Strong"],
            textvariable=self.edit_preset_var,
            width=12,
        )
        self.edit_preset_menu.pack(side=tk.LEFT, padx=(8, 0))
        self.edit_preset_menu.bind("<<ComboboxSelected>>", self.apply_edit_preset)
        ttk.Label(
            mm,
            text="Includes z-image and additional models discovered in the unet folder.",
            wraplength=430,
        ).pack(anchor="w", pady=(6, 0))

        refs = ttk.LabelFrame(inner, text="Reference photos (multi-select)", padding=8)
        refs.pack(fill=tk.X, pady=(0, 8))
        self.edit_refs_list = tk.Listbox(refs, height=6, selectmode=tk.EXTENDED)
        self.edit_refs_list.pack(fill=tk.X)
        ref_btns = ttk.Frame(refs)
        ref_btns.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(ref_btns, text="Add photos…", command=self.add_edit_refs).pack(side=tk.LEFT)
        ttk.Button(ref_btns, text="Remove selected", command=self.remove_edit_refs).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(
            refs,
            text="First item is used as --init-img. Reference-image passthrough (-r) is optional and off by default for the most stable edit behavior.",
            wraplength=430,
        ).pack(anchor="w", pady=(6, 0))
        self.edit_use_ref_images = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            refs,
            text="Also pass the selected photos as reference images (-r)",
            variable=self.edit_use_ref_images,
        ).pack(anchor="w", pady=(0, 6))

        pp = ttk.LabelFrame(inner, text="Edit prompt", padding=8)
        pp.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(pp, text="Prompt").pack(anchor="w")
        self.edit_prompt = self.make_resizable_text(pp, height=4, width=44)
        self.edit_prompt.insert("1.0", "preserve identity, improve lighting, realistic details")
        ttk.Label(pp, text="Negative prompt").pack(anchor="w", pady=(8, 0))
        self.edit_negative = self.make_resizable_text(pp, height=3, width=44)
        self.edit_negative.insert("1.0", NEGATIVE_DEFAULT)

        el = ttk.LabelFrame(inner, text="Edit LoRAs", padding=8)
        el.pack(fill=tk.X, pady=(0, 8))
        self.edit_lora_container = ttk.Frame(el)
        self.edit_lora_container.pack(fill=tk.X)
        edit_lora_btns = ttk.Frame(el)
        edit_lora_btns.pack(fill=tk.X, pady=(6, 0))
        self.add_edit_lora_btn = ttk.Button(
            edit_lora_btns,
            text="Add LoRA",
            command=lambda: self.add_edit_lora_row(default_name="", default_weight=0.0),
        )
        self.add_edit_lora_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.remove_edit_lora_btn = ttk.Button(
            edit_lora_btns,
            text="Remove last",
            command=self.remove_last_edit_lora_row,
            state=tk.DISABLED,
        )
        self.remove_edit_lora_btn.pack(side=tk.LEFT)
        self.add_edit_lora_row(default_name="", default_weight=0.0)

        es = ttk.LabelFrame(inner, text="Edit settings", padding=8)
        es.pack(fill=tk.X, pady=(0, 8))
        self.edit_width = tk.IntVar(value=1024)
        self.edit_height = tk.IntVar(value=1024)
        wh = ttk.Frame(es)
        wh.pack(fill=tk.X)
        ttk.Label(wh, text="W").pack(side=tk.LEFT)
        ttk.Entry(wh, textvariable=self.edit_width, width=8).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(wh, text="H").pack(side=tk.LEFT)
        ttk.Entry(wh, textvariable=self.edit_height, width=8).pack(side=tk.LEFT, padx=(4, 0))
        seed_row = ttk.Frame(es)
        seed_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(seed_row, text="Seed").pack(side=tk.LEFT)
        ttk.Entry(seed_row, textvariable=self.edit_seed_var, width=18).pack(side=tk.LEFT, padx=(8, 0))

        self.edit_steps = tk.IntVar(value=4)
        self.edit_cfg = tk.DoubleVar(value=1.0)
        self.edit_sampling = tk.StringVar(value="euler")
        self.edit_scheduler = tk.StringVar(value="simple")
        self.edit_strength = tk.DoubleVar(value=0.55)
        self.edit_strength_lbl = tk.StringVar(value="0.55")

        ttk.Label(es, text="Steps").pack(anchor="w", pady=(8, 0))
        ttk.Entry(es, textvariable=self.edit_steps, width=8).pack(anchor="w")
        ttk.Label(es, text="CFG scale").pack(anchor="w", pady=(8, 0))
        ttk.Entry(es, textvariable=self.edit_cfg, width=8).pack(anchor="w")
        ttk.Label(es, text="Sampling").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(es, state="readonly", values=SAMPLING_METHODS, textvariable=self.edit_sampling, width=30).pack(fill=tk.X)
        ttk.Label(es, text="Scheduler").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(es, state="readonly", values=SCHEDULERS, textvariable=self.edit_scheduler, width=30).pack(fill=tk.X)
        ttk.Label(es, text="Strength").pack(anchor="w", pady=(8, 0))
        srow = ttk.Frame(es)
        srow.pack(fill=tk.X)
        ttk.Scale(
            srow,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.edit_strength,
            command=self.on_edit_strength,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(srow, textvariable=self.edit_strength_lbl, width=6).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(
            es,
            text="Use Subtle for small corrections, Balanced for most edits, and Strong for larger transformations.",
            wraplength=430,
        ).pack(anchor="w", pady=(6, 0))

        mask = ttk.LabelFrame(inner, text="Inpainting workspace", padding=8)
        mask.pack(fill=tk.X, pady=(0, 8))
        self.edit_mask_status = tk.StringVar(value="Mask mode disabled for whole-image edits.")
        mask_top = ttk.Frame(mask)
        mask_top.pack(fill=tk.X)
        ttk.Label(mask_top, text="Brush size").pack(side=tk.LEFT)
        self.edit_mask_brush_size = tk.IntVar(value=28)
        ttk.Scale(
            mask_top,
            from_=4,
            to=96,
            orient="horizontal",
            variable=self.edit_mask_brush_size,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self.edit_mask_brush_label = tk.StringVar(value="28 px")
        ttk.Label(mask_top, textvariable=self.edit_mask_brush_label, width=7).pack(side=tk.LEFT)
        self.edit_mask_brush_size.trace_add("write", self._on_edit_brush_size_change)
        ttk.Label(mask_top, text="Zoom").pack(side=tk.LEFT, padx=(10, 0))
        self.edit_mask_zoom = tk.DoubleVar(value=1.0)
        ttk.Scale(
            mask_top,
            from_=1.0,
            to=3.0,
            orient="horizontal",
            variable=self.edit_mask_zoom,
            command=self._on_edit_zoom_change,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self.edit_mask_zoom_label = tk.StringVar(value="1.0x")
        ttk.Label(mask_top, textvariable=self.edit_mask_zoom_label, width=5).pack(side=tk.LEFT)
        mask_tools = ttk.Frame(mask)
        mask_tools.pack(fill=tk.X, pady=(6, 0))
        self.edit_mask_tool_var = tk.StringVar(value="paint")
        ttk.Radiobutton(mask_tools, text="Paint", variable=self.edit_mask_tool_var, value="paint").pack(side=tk.LEFT)
        ttk.Radiobutton(mask_tools, text="Erase", variable=self.edit_mask_tool_var, value="erase").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(mask_tools, text="Undo", command=self.undo_edit_mask).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(mask_tools, text="Clear mask", command=self.clear_edit_mask).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(mask_tools, text="Invert mask", command=self.invert_edit_mask).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(mask_tools, text="Fit view", command=self.reset_edit_zoom).pack(side=tk.LEFT, padx=(6, 0))
        mask_refine = ttk.Frame(mask)
        mask_refine.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(mask_refine, text="Grow").pack(side=tk.LEFT)
        self.edit_mask_grow = tk.IntVar(value=12)
        ttk.Entry(mask_refine, textvariable=self.edit_mask_grow, width=6).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(mask_refine, text="Feather").pack(side=tk.LEFT)
        self.edit_mask_feather = tk.IntVar(value=10)
        ttk.Entry(mask_refine, textvariable=self.edit_mask_feather, width=6).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(mask_refine, text="px", width=4).pack(side=tk.LEFT)
        ttk.Label(
            mask,
            text="Best workflow: choose Masked Edit, zoom in, paint only the region to change, then use a precise prompt.",
            wraplength=430,
        ).pack(anchor="w", pady=(6, 0))
        self.edit_mask_canvas = tk.Canvas(mask, width=512, height=512, bg="#202020", highlightthickness=1, highlightbackground="#666")
        self.edit_mask_canvas.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.edit_mask_canvas.bind("<ButtonPress-1>", self._start_paint_edit_mask)
        self.edit_mask_canvas.bind("<B1-Motion>", self._paint_edit_mask)
        self.edit_mask_canvas.bind("<ButtonRelease-1>", self._end_paint_edit_mask)
        ttk.Label(mask, textvariable=self.edit_mask_status, wraplength=430).pack(anchor="w", pady=(8, 0))

        est = ttk.LabelFrame(inner, text="Edit model stack overrides", padding=8)
        est.pack(fill=tk.X, pady=(0, 8))
        self.edit_use_stack_overrides = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            est,
            text="Use dedicated edit paths instead of the global Advanced model stack",
            variable=self.edit_use_stack_overrides,
        ).pack(anchor="w")
        ttk.Label(
            est,
            text="Useful for SDXL editing so Image edit does not inherit the z-image default VAE and encoder stack.",
            wraplength=430,
        ).pack(anchor="w", pady=(4, 6))
        self.edit_main_model_override = tk.StringVar(value="")
        self.edit_vae_override = tk.StringVar(value="")
        self.edit_clip_l_override = tk.StringVar(value="")
        self.edit_clip_g_override = tk.StringVar(value="")
        self.edit_t5xxl_override = tk.StringVar(value="")
        self.edit_llm_override = tk.StringVar(value="")
        self.edit_llm_vision_override = tk.StringVar(value="")
        self._row_var_entry(est, "Main -m weights", self.edit_main_model_override, browse="file")
        self._row_var_entry(est, "VAE (--vae)", self.edit_vae_override, browse="file")
        self._row_var_entry(est, "CLIP-L", self.edit_clip_l_override, browse="file")
        self._row_var_entry(est, "CLIP-G", self.edit_clip_g_override, browse="file")
        self._row_var_entry(est, "T5-XXL", self.edit_t5xxl_override, browse="file")
        self._row_var_entry(est, "LLM (--llm)", self.edit_llm_override, browse="file")
        self._row_var_entry(est, "LLM vision", self.edit_llm_vision_override, browse="file")
        self._on_edit_brush_size_change()
        self.apply_edit_preset()
        self._update_edit_mode_ui()

    def add_edit_refs(self) -> None:
        files = filedialog.askopenfilenames(
            title="Choose reference photos",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All files", "*.*")],
        )
        for f in files:
            if f:
                self.edit_refs_list.insert("end", f)
        if self.edit_refs_list.size() > 0:
            self._load_edit_canvas_image(self.edit_refs_list.get(0))

    def remove_edit_refs(self) -> None:
        sel = list(self.edit_refs_list.curselection())
        for idx in reversed(sel):
            self.edit_refs_list.delete(idx)
        if self.edit_refs_list.size() > 0:
            self._load_edit_canvas_image(self.edit_refs_list.get(0))
        else:
            self._clear_edit_canvas()

    def add_edit_lora_row(self, default_name: str, default_weight: float) -> None:
        self._add_lora_row_to(
            self.edit_lora_container,
            self.edit_lora_rows,
            self.add_edit_lora_btn,
            self.remove_edit_lora_btn,
            default_name,
            default_weight,
        )

    def remove_last_edit_lora_row(self) -> None:
        self._remove_last_lora_row_from(
            self.edit_lora_rows,
            self.add_edit_lora_btn,
            self.remove_edit_lora_btn,
        )

    def on_edit_mode_changed(self, _event=None) -> None:
        label = self.edit_mode_menu.get().strip()
        self.edit_mode_var.set("masked" if label == "Masked Edit" else "whole")
        self._update_edit_mode_ui()

    def on_edit_strength(self, value: str) -> None:
        self.edit_strength_lbl.set(f"{float(value):.2f}")

    def apply_edit_preset(self, _event=None) -> None:
        preset = self.edit_preset_var.get().strip() or "Balanced"
        values = {
            "Subtle": {"strength": 0.35, "steps": 4, "cfg": 1.0},
            "Balanced": {"strength": 0.55, "steps": 6, "cfg": 1.2},
            "Strong": {"strength": 0.8, "steps": 8, "cfg": 1.8},
        }.get(preset, {"strength": 0.55, "steps": 6, "cfg": 1.2})
        self.edit_strength.set(values["strength"])
        self.edit_steps.set(values["steps"])
        self.edit_cfg.set(values["cfg"])
        self.edit_sampling.set("euler")
        self.edit_scheduler.set("simple")
        self.on_edit_strength(str(values["strength"]))

    def _update_edit_mode_ui(self) -> None:
        masked = self.edit_mode_var.get() == "masked"
        self.edit_mask_canvas.configure(state=(tk.NORMAL if masked else tk.DISABLED))
        if masked:
            if self._edit_canvas_image is None and self.edit_refs_list.size() > 0:
                self._load_edit_canvas_image(self.edit_refs_list.get(0))
            self.edit_mask_status.set("Masked edit active. Paint white where the model should modify the image.")
        else:
            self.edit_mask_status.set("Mask mode disabled for whole-image edits.")
        self._refresh_edit_canvas_preview()

    def _on_edit_brush_size_change(self, *_args) -> None:
        self.edit_mask_brush_label.set(f"{int(float(self.edit_mask_brush_size.get()))} px")

    def _on_edit_zoom_change(self, _value=None) -> None:
        self._edit_canvas_zoom = float(self.edit_mask_zoom.get())
        self.edit_mask_zoom_label.set(f"{self._edit_canvas_zoom:.1f}x")
        self._refresh_edit_canvas_preview()

    def reset_edit_zoom(self) -> None:
        self.edit_mask_zoom.set(1.0)
        self._on_edit_zoom_change()

    def _push_edit_mask_history(self) -> None:
        if self._edit_canvas_mask is None:
            return
        self._edit_mask_history.append(self._edit_canvas_mask.copy())
        self._edit_mask_history = self._edit_mask_history[-20:]

    def undo_edit_mask(self) -> None:
        if not self._edit_mask_history or self._edit_canvas_image is None:
            return
        self._edit_canvas_mask = self._edit_mask_history.pop()
        self._edit_canvas_mask_draw = ImageDraw.Draw(self._edit_canvas_mask)
        self._edit_mask_dirty = True
        self._refresh_edit_canvas_preview()
        self.edit_mask_status.set("Undid the last mask change.")

    def _clear_edit_canvas(self) -> None:
        self._edit_canvas_image = None
        self._edit_canvas_mask = None
        self._edit_canvas_mask_draw = None
        self._edit_canvas_preview = None
        self._edit_canvas_base_preview = None
        self._edit_canvas_source_path = ""
        self._edit_mask_dirty = False
        self._edit_mask_history = []
        if hasattr(self, "edit_mask_canvas"):
            self.edit_mask_canvas.delete("all")
        if hasattr(self, "edit_mask_status"):
            self.edit_mask_status.set("Load a source image to enable the built-in mask editor.")

    def _load_edit_canvas_image(self, path: str) -> None:
        try:
            img = Image.open(path).convert("RGBA")
        except OSError as ex:
            self._clear_edit_canvas()
            messagebox.showerror("Mask editor", f"Could not load image for mask editing.\n\n{ex}")
            return
        self._edit_canvas_image = img
        self._edit_canvas_mask = Image.new("L", img.size, 0)
        self._edit_canvas_mask_draw = ImageDraw.Draw(self._edit_canvas_mask)
        self._edit_canvas_source_path = path
        self._edit_mask_dirty = False
        self._edit_mask_history = []
        self._refresh_edit_canvas_preview()
        if self.edit_mode_var.get() == "masked":
            self.edit_mask_status.set("Masked edit active. Paint white where the model should modify the image.")

    def _refresh_edit_canvas_preview(self) -> None:
        if not hasattr(self, "edit_mask_canvas"):
            return
        self.edit_mask_canvas.delete("all")
        if self._edit_canvas_image is None:
            self.edit_mask_canvas.create_text(
                256,
                256,
                text="Add a source photo to preview and paint a mask.",
                fill="#dddddd",
                width=320,
            )
            return
        src = self._edit_canvas_image
        target_size = 512
        scale = min(target_size / max(src.width, 1), target_size / max(src.height, 1)) * max(self._edit_canvas_zoom, 1.0)
        scale = min(scale, 6.0)
        self._edit_canvas_scale = scale
        disp_w = max(1, int(src.width * scale))
        disp_h = max(1, int(src.height * scale))
        x0 = (target_size - disp_w) // 2
        y0 = (target_size - disp_h) // 2
        base = src.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        preview = base.copy()
        if self._edit_canvas_mask is not None:
            mask_resized = self._edit_canvas_mask.resize((disp_w, disp_h), Image.Resampling.NEAREST)
            if self.edit_mode_var.get() == "masked":
                overlay = Image.new("RGBA", (disp_w, disp_h), (255, 48, 48, 0))
                overlay.putalpha(mask_resized.point(lambda px: int(px * 0.45)))
                preview = Image.alpha_composite(preview, overlay)
        self._edit_canvas_preview = ImageTk.PhotoImage(preview)
        self.edit_mask_canvas.create_image(x0, y0, anchor="nw", image=self._edit_canvas_preview)
        self.edit_mask_canvas.create_rectangle(x0, y0, x0 + disp_w, y0 + disp_h, outline="#6b6b6b")

    def _start_paint_edit_mask(self, event) -> None:
        if self.edit_mode_var.get() != "masked":
            return
        self._push_edit_mask_history()
        self._edit_mask_stroke_active = True
        self._paint_edit_mask(event)

    def _end_paint_edit_mask(self, _event=None) -> None:
        self._edit_mask_stroke_active = False

    def _paint_edit_mask(self, event) -> None:
        if self.edit_mode_var.get() != "masked":
            return
        if self._edit_canvas_image is None or self._edit_canvas_mask_draw is None:
            return
        src = self._edit_canvas_image
        target_size = 512
        disp_w = max(1, int(src.width * self._edit_canvas_scale))
        disp_h = max(1, int(src.height * self._edit_canvas_scale))
        x0 = (target_size - disp_w) // 2
        y0 = (target_size - disp_h) // 2
        if not (x0 <= event.x < x0 + disp_w and y0 <= event.y < y0 + disp_h):
            return
        img_x = int((event.x - x0) / self._edit_canvas_scale)
        img_y = int((event.y - y0) / self._edit_canvas_scale)
        radius = max(1, int(float(self.edit_mask_brush_size.get()) / max(self._edit_canvas_scale, 0.001) / 2))
        bbox = (img_x - radius, img_y - radius, img_x + radius, img_y + radius)
        fill = 255 if self.edit_mask_tool_var.get() == "paint" else 0
        self._edit_canvas_mask_draw.ellipse(bbox, fill=fill)
        self._edit_mask_dirty = True
        self._refresh_edit_canvas_preview()
        self.edit_mask_status.set("Mask updated. White/red regions will be passed as the edit mask.")

    def clear_edit_mask(self) -> None:
        if self._edit_canvas_image is None:
            return
        self._push_edit_mask_history()
        self._edit_canvas_mask = Image.new("L", self._edit_canvas_image.size, 0)
        self._edit_canvas_mask_draw = ImageDraw.Draw(self._edit_canvas_mask)
        self._edit_mask_dirty = False
        self._refresh_edit_canvas_preview()
        self.edit_mask_status.set("Mask cleared.")

    def invert_edit_mask(self) -> None:
        if self._edit_canvas_mask is None or self._edit_canvas_image is None:
            return
        self._push_edit_mask_history()
        self._edit_canvas_mask = self._edit_canvas_mask.point(lambda px: 255 - px)
        self._edit_canvas_mask_draw = ImageDraw.Draw(self._edit_canvas_mask)
        self._edit_mask_dirty = True
        self._refresh_edit_canvas_preview()
        self.edit_mask_status.set("Mask inverted.")

    def _save_edit_mask_if_needed(self, output_path: str) -> str:
        if self._edit_canvas_mask is None:
            return ""
        out_target = Path(output_path).expanduser()
        out_target.parent.mkdir(parents=True, exist_ok=True)
        mask_path = out_target.with_name(f"{out_target.stem}_mask.png")
        processed_mask = self._edit_canvas_mask.copy()
        try:
            grow_px = max(0, int(self.edit_mask_grow.get()))
        except (TypeError, ValueError, tk.TclError):
            grow_px = 0
        try:
            feather_px = max(0, int(self.edit_mask_feather.get()))
        except (TypeError, ValueError, tk.TclError):
            feather_px = 0
        if grow_px > 0:
            size = max(3, (grow_px * 2) + 1)
            processed_mask = processed_mask.filter(ImageFilter.MaxFilter(size=size))
        if feather_px > 0:
            processed_mask = processed_mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
        processed_mask.save(mask_path)
        self._edit_mask_temp_path = str(mask_path)
        self._edit_mask_dirty = False
        return self._edit_mask_temp_path

    def on_second_pass_strength(self, value: str) -> None:
        self.second_pass_strength_lbl.set(f"{float(value):.2f}")

    def _build_history_tab(self, tab: ttk.Frame) -> None:
        outer = ttk.Frame(tab, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text="The last 15 successful runs are kept (older entries are dropped). "
            "Each entry stores prompts, LoRA names/weights, and the output path. "
            "Select a row and click Apply, or double-click the row.",
            wraplength=430,
        ).pack(anchor="w")

        mid = ttk.Frame(outer)
        mid.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        vsb = ttk.Scrollbar(mid, orient="vertical")
        self.history_listbox = tk.Listbox(mid, height=14, yscrollcommand=vsb.set, exportselection=False)
        vsb.config(command=self.history_listbox.yview)
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_listbox.bind("<<ListboxSelect>>", self._on_history_select)
        self.history_listbox.bind("<Double-Button-1>", lambda _e: self.apply_selected_history())

        detail_fr = ttk.LabelFrame(outer, text="Selected entry", padding=6)
        detail_fr.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
        self.history_detail = tk.Text(detail_fr, height=10, wrap="word", state="disabled")
        self.history_detail.pack(fill=tk.BOTH, expand=True)

        btns = ttk.Frame(outer)
        btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btns, text="Apply to form", command=self.apply_selected_history).pack(side=tk.LEFT)
        ttk.Button(btns, text="Open image", command=self.open_selected_history_image).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btns, text="Open folder", command=self.open_selected_history_folder).pack(side=tk.LEFT, padx=(8, 0))

    def _history_row_label(self, e: dict) -> str:
        mode = e.get("mode") or "simple"
        p = (e.get("prompt") or "").replace("\n", " ").strip()
        if len(p) > 52:
            p = p[:49] + "…"
        when = e.get("created") or ""
        if isinstance(when, str) and "T" in when:
            when = when.split("T", 1)[0]
        tail = Path(e.get("output") or "").name or "?"
        if mode == "edit":
            tag = "edit"
        elif mode == "second_pass":
            tag = "pass2"
        else:
            tag = "gen"
        loras = e.get("lora_items")
        lora_hint = ""
        if isinstance(loras, list) and loras:
            lora_hint = f"  [{len(loras)} LoRA]"
        return f"[{tag}] {p}  →  {tail}{lora_hint}  ({when})"

    def _refresh_history_listbox(self) -> None:
        if not hasattr(self, "history_listbox"):
            return
        self.history_listbox.delete(0, tk.END)
        for e in self._history:
            self.history_listbox.insert(tk.END, self._history_row_label(e))

    def _on_history_select(self, _event: tk.Event | None = None) -> None:
        sel = self.history_listbox.curselection()
        self.history_detail.configure(state="normal")
        self.history_detail.delete("1.0", "end")
        if not sel:
            self.history_detail.configure(state="disabled")
            return
        e = self._history[sel[0]]
        parts = [
            f"Mode: {e.get('mode', 'simple')}",
            f"Output: {e.get('output', '')}",
            "",
            "Prompt:",
            e.get("prompt") or "",
            "",
            "Negative:",
            e.get("negative") or "",
        ]
        if e.get("mode") == "second_pass":
            parts.extend(["", f"Edit model: {e.get('model', '')}"])
        loras = e.get("lora_items")
        if isinstance(loras, list) and loras:
            lines: list[str] = []
            for it in loras:
                if not isinstance(it, dict):
                    continue
                nm = (it.get("name") or "").strip()
                if not nm:
                    continue
                try:
                    w = float(it.get("weight", 0))
                except (TypeError, ValueError):
                    w = 0.0
                lines.append(f"  {nm} @ {w:.2f}")
            if lines:
                parts.extend(["", "LoRAs:", *lines])
            elif e.get("mode") != "edit":
                parts.extend(["", "LoRAs:", "  (none)"])
        elif e.get("mode") != "edit":
            parts.extend(["", "LoRAs:", "  (none)"])
        refs = e.get("ref_images")
        if refs:
            parts.extend(["", "Reference images:", "\n".join(str(r) for r in refs)])
        settings = e.get("settings")
        if isinstance(settings, dict):
            for section_name, section_values in settings.items():
                if not isinstance(section_values, dict) or not section_values:
                    continue
                parts.extend(["", f"{section_name}:", *self._format_history_settings(section_values)])
        self.history_detail.insert("1.0", "\n".join(parts))
        self.history_detail.configure(state="disabled")

    def _format_history_settings(self, data: dict, *, indent: str = "  ") -> list[str]:
        lines: list[str] = []
        for key, value in data.items():
            label = str(key).replace("_", " ")
            if isinstance(value, dict):
                lines.append(f"{indent}{label}:")
                lines.extend(self._format_history_settings(value, indent=indent + "  "))
            elif isinstance(value, list):
                if value and all(isinstance(item, dict) for item in value):
                    lines.append(f"{indent}{label}:")
                    for item in value:
                        summary = ", ".join(f"{k}={v}" for k, v in item.items())
                        lines.append(f"{indent}  - {summary}")
                elif value:
                    lines.append(f"{indent}{label}: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"{indent}{label}: []")
            else:
                lines.append(f"{indent}{label}: {value}")
        return lines

    def _load_history_from_disk(self) -> None:
        self._history = []
        if HISTORY_PATH.exists():
            try:
                raw = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
                full: list = []
                if isinstance(raw, list):
                    full = raw
                elif isinstance(raw, dict):
                    full = list(raw.get("entries") or [])
                self._history = full[:HISTORY_MAX]
                if len(full) > HISTORY_MAX:
                    self._persist_history()
            except (json.JSONDecodeError, OSError):
                self._history = []
        self._refresh_history_listbox()

    def _second_pass_preset_payload(self) -> dict:
        return {
            "version": 1,
            "enabled": bool(self.second_pass_enabled.get()),
            "inherit_base_settings": bool(self.second_pass_inherit_base_settings.get()),
            "use_ref_image": bool(self.second_pass_use_ref_image.get()),
            "suffix": self.second_pass_suffix_var.get().strip(),
            "model": self.second_pass_model_var.get().strip(),
            "prompt": self.second_pass_prompt.get("1.0", "end").strip(),
            "negative": self.second_pass_negative.get("1.0", "end").strip(),
            "width": int(self.second_pass_width.get()),
            "height": int(self.second_pass_height.get()),
            "steps": int(self.second_pass_steps.get()),
            "cfg_scale": float(self.second_pass_cfg.get()),
            "sampling_method": self.second_pass_sampling.get().strip(),
            "scheduler": self.second_pass_scheduler.get().strip(),
            "strength": float(self.second_pass_strength.get()),
            "lora_items": self._collect_lora_snapshot(self.second_pass_lora_rows),
        }

    def save_second_pass_preset(self, show_message: bool = False) -> None:
        payload = self._second_pass_preset_payload()
        try:
            SECOND_PASS_PRESET_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            if show_message:
                messagebox.showinfo("Second pass", f"Saved preset to:\n{SECOND_PASS_PRESET_PATH}")
        except OSError as ex:
            messagebox.showerror("Second pass", str(ex))

    def _load_second_pass_preset(self) -> None:
        if not SECOND_PASS_PRESET_PATH.exists():
            return
        try:
            raw = json.loads(SECOND_PASS_PRESET_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(raw, dict):
            return
        self._apply_second_pass_snapshot(raw)

    def _apply_second_pass_snapshot(self, data: dict) -> None:
        self.second_pass_enabled.set(bool(data.get("enabled", self.second_pass_enabled.get())))
        self.second_pass_inherit_base_settings.set(bool(data.get("inherit_base_settings", self.second_pass_inherit_base_settings.get())))
        self.second_pass_use_ref_image.set(bool(data.get("use_ref_image", self.second_pass_use_ref_image.get())))
        self.second_pass_suffix_var.set(str(data.get("suffix") or self.second_pass_suffix_var.get()))
        model = str(data.get("model") or "")
        if model and model in self._all_model_map:
            self.second_pass_model_var.set(model)
        self._set_text_widget(self.second_pass_prompt, str(data.get("prompt") or ""))
        self._set_text_widget(self.second_pass_negative, str(data.get("negative") or ""))
        try:
            self.second_pass_width.set(int(data.get("width", self.second_pass_width.get())))
            self.second_pass_height.set(int(data.get("height", self.second_pass_height.get())))
            self.second_pass_steps.set(int(data.get("steps", self.second_pass_steps.get())))
            self.second_pass_cfg.set(float(data.get("cfg_scale", self.second_pass_cfg.get())))
            self.second_pass_strength.set(float(data.get("strength", self.second_pass_strength.get())))
        except (TypeError, ValueError):
            pass
        self.second_pass_sampling.set(str(data.get("sampling_method") or self.second_pass_sampling.get()))
        self.second_pass_scheduler.set(str(data.get("scheduler") or self.second_pass_scheduler.get()))
        self.second_pass_strength_lbl.set(f"{float(self.second_pass_strength.get()):.2f}")
        self._apply_lora_snapshot(
            data.get("lora_items"),
            self.second_pass_lora_rows,
            self.add_second_pass_lora_row,
            self.remove_last_second_pass_lora_row,
        )

    def _persist_history(self) -> None:
        try:
            self._history = self._history[:HISTORY_MAX]
            payload = {"version": 1, "entries": self._history}
            HISTORY_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def _append_history(self, entry: dict) -> None:
        self._history.insert(0, entry)
        self._history = self._history[:HISTORY_MAX]
        self._persist_history()
        self._refresh_history_listbox()

    def _set_text_widget(self, widget: tk.Text, content: str) -> None:
        widget.delete("1.0", "end")
        widget.insert("1.0", content)

    def _sync_lora_row_widgets(self, row: dict, name: str, weight: float) -> None:
        row["name_var"].set(name)
        row["weight_var"].set(weight)
        row["weight_lbl"].set(f"{weight:.2f}")
        row["weight_entry_var"].set(f"{weight:.2f}")
        try:
            row["scale"].set(weight)
        except tk.TclError:
            pass

    def _apply_history_loras(self, items: object) -> None:
        self._apply_lora_snapshot(items, self.lora_rows, self.add_lora_row, self.remove_last_lora_row)

    def _apply_edit_history_loras(self, items: object) -> None:
        self._apply_lora_snapshot(items, self.edit_lora_rows, self.add_edit_lora_row, self.remove_last_edit_lora_row)

    def _apply_lora_snapshot(self, items: object, row_list: list[dict], add_fn, remove_fn) -> None:
        if not isinstance(items, list):
            return
        parsed: list[tuple[str, float]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            n = (it.get("name") or "").strip()
            if not n:
                continue
            try:
                w = float(it.get("weight", 0))
            except (TypeError, ValueError):
                w = 0.0
            if w == 0:
                continue
            parsed.append((n, w))
        while len(row_list) > 1:
            remove_fn()
        if not row_list:
            add_fn("", 0.0)
        if not parsed:
            self._sync_lora_row_widgets(row_list[0], "", 0.0)
            return
        self._sync_lora_row_widgets(row_list[0], parsed[0][0], parsed[0][1])
        for name, w in parsed[1:]:
            add_fn(default_name=name, default_weight=w)

    def apply_selected_history(self) -> None:
        sel = self.history_listbox.curselection()
        if not sel:
            messagebox.showinfo("History", "Select a history row first.")
            return
        e = self._history[sel[0]]
        out = (e.get("output") or "").strip()
        if out:
            self.out_var.set(out)
        mode = e.get("mode") or "simple"
        if mode == "edit":
            self.left_notebook.select(self.tab_edit)
            self._set_text_widget(self.edit_prompt, e.get("prompt") or "")
            self._set_text_widget(self.edit_negative, e.get("negative") or "")
            self._apply_edit_history_loras(e.get("lora_items"))
            refs = e.get("ref_images")
            if isinstance(refs, list) and refs:
                self.edit_refs_list.delete(0, tk.END)
                for r in refs:
                    if r:
                        self.edit_refs_list.insert(tk.END, str(r))
                self._load_edit_canvas_image(str(refs[0]))
            settings = e.get("settings") or {}
            edit_settings = settings.get("Image edit settings") if isinstance(settings, dict) else {}
            if isinstance(edit_settings, dict):
                mode_name = str(edit_settings.get("workflow_mode") or "")
                self.edit_mode_menu.set("Masked Edit" if mode_name == "masked" else "Whole Image Edit")
                self.on_edit_mode_changed()
                preset = str(edit_settings.get("preset") or "")
                if preset in {"Subtle", "Balanced", "Strong"}:
                    self.edit_preset_var.set(preset)
        elif mode == "second_pass":
            self.left_notebook.select(self.tab_simple)
            self._apply_second_pass_snapshot(e)
        else:
            self.left_notebook.select(self.tab_simple)
            self._set_text_widget(self.prompt, e.get("prompt") or "")
            self._set_text_widget(self.neg_prompt, e.get("negative") or "")
            self._apply_history_loras(e.get("lora_items"))

    def open_selected_history_image(self) -> None:
        sel = self.history_listbox.curselection()
        if not sel:
            messagebox.showinfo("History", "Select a history row first.")
            return
        self._open_file_path(self._history[sel[0]].get("output") or "")

    def open_selected_history_folder(self) -> None:
        sel = self.history_listbox.curselection()
        if not sel:
            messagebox.showinfo("History", "Select a history row first.")
            return
        outp = self._history[sel[0]].get("output") or ""
        folder = str(Path(outp).expanduser().resolve().parent) if outp else ""
        if folder and Path(folder).is_dir():
            self._open_file_path(folder)
        else:
            messagebox.showwarning("History", "Could not open folder for that output path.")

    def _open_file_path(self, path: str) -> None:
        path = str(Path(path).expanduser())
        if not path:
            return
        p = Path(path)
        if not p.exists():
            messagebox.showwarning("History", f"Path does not exist:\n{path}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except OSError as ex:
            messagebox.showerror("Open", str(ex))

    def _kill_proc_tree(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        pid = proc.pid
        try:
            if sys.platform == "win32":
                kwargs: dict = {}
                cr = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if cr:
                    kwargs["creationflags"] = cr
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    **kwargs,
                )
            else:
                proc.kill()
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass

    def _bool_chk(self, parent: ttk.Widget, text: str, key: str) -> None:
        ttk.Checkbutton(parent, text=text, variable=self._bool_opt(key)).pack(anchor="w")

    def _row_var_entry(self, parent: ttk.Widget, label: str, var: tk.StringVar, browse: str | None = None) -> None:
        fr = ttk.Frame(parent)
        fr.pack(fill=tk.X, pady=2)
        ttk.Label(fr, text=label, width=24).pack(side=tk.LEFT, anchor="nw")
        ttk.Entry(fr, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        if browse == "file":
            ttk.Button(fr, text="…", width=3, command=lambda: self._browse_file(var)).pack(side=tk.LEFT, padx=(4, 0))
        elif browse == "dir":
            ttk.Button(fr, text="…", width=3, command=lambda: self._browse_dir(var)).pack(side=tk.LEFT, padx=(4, 0))

    def show_speed_help(self) -> None:
        messagebox.showinfo("Generation speed help", SPEED_HELP_TEXT)

    def make_resizable_text(self, parent: ttk.Widget, *, height: int, width: int) -> tk.Text:
        wrap = ttk.Frame(parent)
        wrap.pack(fill=tk.X)
        txt = tk.Text(wrap, height=height, width=width, wrap="word")
        txt.pack(fill=tk.X, expand=True)
        grip = ttk.Label(wrap, text="◢", cursor="size_nw_se")
        grip.place(relx=1.0, rely=1.0, anchor="se")
        drag_state = {"x": 0, "y": 0, "w": width, "h": height}

        def on_press(event) -> None:
            drag_state["x"] = event.x_root
            drag_state["y"] = event.y_root
            drag_state["w"] = int(txt.cget("width"))
            drag_state["h"] = int(txt.cget("height"))

        def on_drag(event) -> None:
            dx = event.x_root - drag_state["x"]
            dy = event.y_root - drag_state["y"]
            new_w = max(20, drag_state["w"] + int(dx / 7))
            new_h = max(2, drag_state["h"] + int(dy / 18))
            txt.configure(width=new_w, height=new_h)

        grip.bind("<ButtonPress-1>", on_press)
        grip.bind("<B1-Motion>", on_drag)
        return txt

    def set_dimensions(self, width: int, height: int) -> None:
        self.width_var.set(width)
        self.height_var.set(height)

    def on_steps(self, value: str) -> None:
        v = int(float(value))
        self.steps_var.set(v)
        self.steps_lbl.set(str(v))
        self.steps_entry_var.set(str(v))

    def on_cfg(self, value: str) -> None:
        v = float(value)
        self.cfg_var.set(v)
        self.cfg_lbl.set(f"{v:.2f}")
        self.cfg_entry_var.set(f"{v:.2f}")

    def on_steps_entry(self, _event=None) -> None:
        try:
            v = int(float(self.steps_entry_var.get().strip()))
        except ValueError:
            self.steps_entry_var.set(str(self.steps_var.get()))
            return
        v = max(1, min(50, v))
        self.steps_var.set(v)
        self.steps_lbl.set(str(v))
        self.steps_entry_var.set(str(v))

    def on_cfg_entry(self, _event=None) -> None:
        try:
            v = float(self.cfg_entry_var.get().strip())
        except ValueError:
            self.cfg_entry_var.set(f"{self.cfg_var.get():.2f}")
            return
        v = max(1.0, min(20.0, v))
        self.cfg_var.set(v)
        self.cfg_lbl.set(f"{v:.2f}")
        self.cfg_entry_var.set(f"{v:.2f}")

    def pick_output(self) -> None:
        f = filedialog.asksaveasfilename(
            title="Choose output PNG",
            defaultextension=".png",
            filetypes=[("PNG images", "*.png"), ("All files", "*.*")],
            initialfile=Path(self.out_var.get()).name if self.out_var.get() else "zimage_output.png",
        )
        if f:
            self.out_var.set(f)

    def pick_sd_exe(self) -> None:
        f = filedialog.askopenfilename(title="Choose sd.exe", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if f:
            self.sd_exe_var.set(f)

    def pick_init_image(self) -> None:
        f = filedialog.askopenfilename(
            title="Choose init image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All files", "*.*")],
        )
        if f:
            self.init_img_var.set(f)

    def on_strength(self, value: str) -> None:
        v = float(value)
        self.strength_var.set(v)
        self.strength_lbl.set(f"{v:.2f}")

    def unique_output_path(self, p: str) -> str:
        target = Path(p).expanduser()
        if target.suffix == "":
            target = target.with_suffix(".png")
        if not target.exists():
            return str(target)
        i = 1
        while True:
            cand = target.with_name(f"{target.stem} ({i}){target.suffix}")
            if not cand.exists():
                return str(cand)
            i += 1

    def _collect_lora_snapshot(self, row_list: list[dict]) -> list[dict]:
        out: list[dict] = []
        for row in row_list:
            name = row["name_var"].get().strip()
            w = float(row["weight_var"].get())
            if name and w != 0:
                out.append({"name": name, "weight": w})
        return out

    def _add_lora_row_to(
        self,
        container: ttk.Frame,
        row_list: list[dict],
        add_btn: ttk.Button,
        remove_btn: ttk.Button,
        default_name: str,
        default_weight: float,
    ) -> None:
        rowf = ttk.Frame(container)
        rowf.pack(fill=tk.X, pady=(0, 5))
        name_var = tk.StringVar(value=default_name)
        weight_var = tk.DoubleVar(value=default_weight)
        weight_lbl = tk.StringVar(value=f"{default_weight:.2f}")
        weight_entry_var = tk.StringVar(value=f"{default_weight:.2f}")
        min_var = tk.DoubleVar(value=0.0)
        max_var = tk.DoubleVar(value=2.0)
        ttk.Combobox(rowf, state="readonly", values=self._lora_names, textvariable=name_var, width=22).pack(side=tk.LEFT, fill=tk.X, expand=True)
        scale = ttk.Scale(
            rowf,
            from_=0.0,
            to=2.0,
            value=default_weight,
            orient="horizontal",
            command=lambda v, _w=weight_var, _l=weight_lbl, _e=weight_entry_var: (
                _w.set(float(v)),
                _l.set(f"{float(v):.2f}"),
                _e.set(f"{float(v):.2f}"),
            ),
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        ttk.Label(rowf, textvariable=weight_lbl, width=7).pack(side=tk.LEFT)
        weight_entry = ttk.Entry(rowf, textvariable=weight_entry_var, width=6)
        weight_entry.pack(side=tk.LEFT, padx=(6, 0))

        row = {
            "frame": rowf,
            "name_var": name_var,
            "weight_var": weight_var,
            "weight_lbl": weight_lbl,
            "weight_entry_var": weight_entry_var,
            "scale": scale,
            "min_var": min_var,
            "max_var": max_var,
        }

        def clamp_and_apply() -> None:
            lo = float(row["min_var"].get())
            hi = float(row["max_var"].get())
            row["scale"].configure(from_=lo, to=hi)
            try:
                w = float(row["weight_var"].get())
            except (TypeError, ValueError):
                w = 0.0
            w = max(lo, min(hi, w))
            row["weight_var"].set(w)
            row["weight_lbl"].set(f"{w:.2f}")
            row["weight_entry_var"].set(f"{w:.2f}")
            try:
                row["scale"].set(w)
            except tk.TclError:
                pass

        def on_weight_entry(_e=None) -> None:
            try:
                row["weight_var"].set(float(row["weight_entry_var"].get().strip()))
            except ValueError:
                pass
            clamp_and_apply()

        def extend_min() -> None:
            row["min_var"].set(float(row["min_var"].get()) - 0.25)
            clamp_and_apply()

        def extend_max() -> None:
            row["max_var"].set(float(row["max_var"].get()) + 0.25)
            clamp_and_apply()

        weight_entry.bind("<Return>", on_weight_entry)
        weight_entry.bind("<FocusOut>", on_weight_entry)

        ttk.Button(rowf, text="-0.25", width=6, command=extend_min).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(rowf, text="+0.25", width=6, command=extend_max).pack(side=tk.LEFT, padx=(4, 0))

        row_list.append(row)
        clamp_and_apply()
        remove_btn.config(state=(tk.NORMAL if len(row_list) > 1 else tk.DISABLED))
        add_btn.config(state=(tk.NORMAL if len(row_list) < 6 else tk.DISABLED))

    def _remove_last_lora_row_from(self, row_list: list[dict], add_btn: ttk.Button, remove_btn: ttk.Button) -> None:
        if len(row_list) <= 1:
            return
        row = row_list.pop()
        row["frame"].destroy()
        remove_btn.config(state=(tk.NORMAL if len(row_list) > 1 else tk.DISABLED))
        add_btn.config(state=(tk.NORMAL if len(row_list) < 6 else tk.DISABLED))

    def add_lora_row(self, default_name: str, default_weight: float) -> None:
        self._add_lora_row_to(
            self.lora_container,
            self.lora_rows,
            self.add_lora_btn,
            self.remove_lora_btn,
            default_name,
            default_weight,
        )

    def remove_last_lora_row(self) -> None:
        self._remove_last_lora_row_from(self.lora_rows, self.add_lora_btn, self.remove_lora_btn)

    def add_second_pass_lora_row(self, default_name: str, default_weight: float) -> None:
        self._add_lora_row_to(
            self.second_pass_lora_container,
            self.second_pass_lora_rows,
            self.add_second_pass_lora_btn,
            self.remove_second_pass_lora_btn,
            default_name,
            default_weight,
        )

    def remove_last_second_pass_lora_row(self) -> None:
        self._remove_last_lora_row_from(
            self.second_pass_lora_rows,
            self.add_second_pass_lora_btn,
            self.remove_second_pass_lora_btn,
        )

    def append_log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    def _coerce_option_value(self, key: str, raw: str) -> object:
        s = raw.strip()
        if key in INT_OPTION_KEYS:
            return int(float(s))
        if key in FLOAT_OPTION_KEYS:
            return float(s)
        return s

    def _collect_options(self) -> dict:
        o: dict[str, object] = {}

        o["diffusion_model"] = self.model_path_var.get().strip()
        o["negative_prompt"] = self.neg_prompt.get("1.0", "end").strip()

        w, h = int(self.width_var.get()), int(self.height_var.get())
        if w > 0:
            o["width"] = w
        if h > 0:
            o["height"] = h
        o["steps"] = int(self.steps_var.get())
        o["cfg_scale"] = float(self.cfg_var.get())
        o["sampling_method"] = self.sampling_var.get().strip()
        hn = self.high_noise_sampling_var.get().strip()
        if hn:
            o["high_noise_sampling_method"] = hn
        o["scheduler"] = self.scheduler_var.get().strip()

        seed = self.seed_var.get().strip()
        if seed:
            o["seed"] = int(seed)

        if self.low_vram.get():
            o["clip_on_cpu"] = True
            o["vae_on_cpu"] = True
        # Keep VAE tiling opt-in only; avoids blocky/pixelated outputs.
        if self.vae_tiling.get():
            o["vae_tiling"] = True

        if self.img2img_enabled.get():
            ii = self.init_img_var.get().strip()
            if ii:
                o["init_img"] = ii
                o["strength"] = float(self.strength_var.get())

        if self.verbose_var.get():
            o["verbose"] = True
        else:
            o["verbose"] = False
        if self.color_log_var.get():
            o["color_log"] = True

        for key, var in self._str_opts.items():
            if key == "extra_cli":
                continue
            s = var.get().strip()
            if not s:
                continue
            try:
                o[key] = self._coerce_option_value(key, s)
            except ValueError:
                o[key] = s

        for key, var in self._bool_opts.items():
            if var.get():
                o[key] = True

        refs = [ln.strip() for ln in self.ref_images_txt.get("1.0", "end").splitlines() if ln.strip()]
        if refs:
            o["ref_images"] = refs

        extra = self._str_opts.get("extra_cli")
        if extra and extra.get().strip():
            o["extra_cli"] = extra.get().strip()

        return o

    def _collect_active_loras_for_history(self) -> list[dict]:
        return self._collect_lora_snapshot(self.lora_rows)

    def _collect_lora_items(self, row_list: list[dict]) -> list[dict]:
        out = []
        for row in row_list:
            name = row["name_var"].get().strip()
            w = float(row["weight_var"].get())
            if name and w != 0:
                out.append({"path": str(LORAS_DIR / name), "weight": w})
        return out

    def _history_advanced_settings(self, options: dict) -> dict:
        keys = [
            "llm_path", "llm_vision", "vae_path", "model", "clip_l", "clip_g", "clip_vision",
            "t5xxl", "high_noise_diffusion_model", "taesd", "tae", "lora_model_dir", "embd_dir",
            "tensor_type_rules", "photo_maker", "upscale_model", "type_override", "prediction",
            "lora_apply_mode", "mode", "verbose", "color_log", "high_noise_sampling_method",
            "sigmas", "skip_layers", "high_noise_skip_layers", "easycache", "offload_to_cpu",
            "vae_on_cpu", "control_net_cpu", "threads", "vae_tile_size", "vae_tile_overlap",
            "vae_relative_tile_size", "flow_shift", "high_noise_steps", "clip_skip", "batch_count",
            "video_frames", "fps", "timestep_shift", "upscale_repeats", "upscale_tile_size",
            "img_cfg_scale", "guidance", "slg_scale", "skip_layer_start", "skip_layer_end", "eta",
            "high_noise_cfg_scale", "high_noise_img_cfg_scale", "high_noise_guidance",
            "high_noise_slg_scale", "high_noise_skip_layer_start", "high_noise_skip_layer_end",
            "high_noise_eta", "pm_style_strength", "control_strength", "moe_boundary", "vace_strength",
            "control_net", "control_image", "control_video", "mask", "end_img", "pm_id_images_dir",
            "pm_id_embed_path", "preview", "preview_path", "preview_interval", "rng", "sampler_rng",
            "chroma_t5_mask_pad", "ref_images", "extra_cli", "vae_tiling", "clip_on_cpu",
            "diffusion_fa", "diffusion_conv_direct", "vae_conv_direct", "force_sdxl_vae_conv_scale",
            "canny", "taesd_preview_only", "preview_noisy", "increase_ref_index",
            "disable_auto_resize_ref_image", "chroma_disable_dit_mask", "chroma_enable_t5_mask",
        ]
        return {key: options[key] for key in keys if key in options}

    def _history_simple_settings(self, cfg: dict) -> dict:
        options = dict(cfg.get("options") or {})
        simple_keys = [
            "diffusion_model", "width", "height", "steps", "cfg_scale", "sampling_method",
            "scheduler", "seed", "negative_prompt", "init_img", "strength",
        ]
        settings = {
            "Simple run settings": {key: options[key] for key in simple_keys if key in options},
            "Simple run LoRAs": {"items": self._collect_active_loras_for_history()},
            "Advanced settings": self._history_advanced_settings(options),
        }
        if self.second_pass_enabled.get():
            preview_second_cfg = self.build_second_pass_config("__history_preview__.png", "__history_preview_input__.png", cfg)
            settings.update(self._history_second_pass_settings(preview_second_cfg))
        return settings

    def _history_edit_settings(self, cfg: dict) -> dict:
        options = dict(cfg.get("options") or {})
        edit_keys = [
            "diffusion_model", "width", "height", "steps", "cfg_scale", "sampling_method",
            "scheduler", "negative_prompt", "init_img", "strength", "ref_images", "mask", "seed",
        ]
        settings = {
            "Image edit settings": {key: options[key] for key in edit_keys if key in options},
            "Image edit LoRAs": {"items": self._collect_lora_snapshot(self.edit_lora_rows)},
        }
        settings["Image edit settings"]["workflow_mode"] = self.edit_mode_var.get()
        settings["Image edit settings"]["preset"] = self.edit_preset_var.get().strip()
        settings["Image edit settings"]["use_ref_images"] = bool(self.edit_use_ref_images.get())
        settings["Image edit settings"]["mask_grow"] = int(self.edit_mask_grow.get())
        settings["Image edit settings"]["mask_feather"] = int(self.edit_mask_feather.get())
        if self.edit_use_stack_overrides.get():
            settings["Image edit stack overrides"] = {
                "enabled": True,
                "model": self.edit_main_model_override.get().strip(),
                "vae_path": self.edit_vae_override.get().strip(),
                "clip_l": self.edit_clip_l_override.get().strip(),
                "clip_g": self.edit_clip_g_override.get().strip(),
                "t5xxl": self.edit_t5xxl_override.get().strip(),
                "llm_path": self.edit_llm_override.get().strip(),
                "llm_vision": self.edit_llm_vision_override.get().strip(),
            }
        return settings

    def _history_second_pass_settings(self, cfg: dict) -> dict:
        options = dict(cfg.get("options") or {})
        pass2_keys = [
            "diffusion_model", "width", "height", "steps", "cfg_scale", "sampling_method",
            "scheduler", "negative_prompt", "init_img", "strength", "ref_images",
        ]
        return {
            "Second pass settings": {
                "inherit_base_settings": bool(self.second_pass_inherit_base_settings.get()),
                "use_ref_image": bool(self.second_pass_use_ref_image.get()),
                **{key: options[key] for key in pass2_keys if key in options},
            },
            "Second pass LoRAs": {"items": self._collect_lora_snapshot(self.second_pass_lora_rows)},
            "Advanced settings": self._history_advanced_settings(options),
        }

    def _validate_edit_model_requirements(self) -> str | None:
        if self.edit_mode_var.get() == "masked":
            if self.edit_refs_list.size() == 0:
                return "Masked edit requires a source image."
            if self._edit_canvas_mask is None:
                return "Masked edit requires a source image loaded into the built-in mask editor."
            if self._edit_canvas_mask.getbbox() is None:
                return "Masked edit requires a painted mask. Paint the regions you want to change."
        try:
            int(self.edit_mask_grow.get())
            int(self.edit_mask_feather.get())
        except (TypeError, ValueError, tk.TclError):
            return "Mask grow and feather must be whole numbers."
        model_label = self.edit_model_var.get().strip().lower()
        if "flux-2-klein-9b" not in model_label:
            return None
        llm_path = (
            self.edit_llm_override.get().strip()
            if self.edit_use_stack_overrides.get() and self.edit_llm_override.get().strip()
            else self._str_opts.get("llm_path", tk.StringVar()).get().strip()
        ).lower()
        vae_path = (
            self.edit_vae_override.get().strip()
            if self.edit_use_stack_overrides.get() and self.edit_vae_override.get().strip()
            else self._str_opts.get("vae_path", tk.StringVar()).get().strip()
        ).lower()
        missing: list[str] = []
        if "qwen3-8b" not in llm_path:
            missing.append("a Qwen3-8B LLM encoder for Flux 2 klein 9B")
        if "flux2_ae" not in vae_path and "flux2-ae" not in vae_path:
            missing.append("the Flux2 VAE (flux2_ae.safetensors)")
        if not missing:
            return None
        return (
            "Flux 2 klein 9B needs a different auxiliary stack than the default z-image setup.\n\n"
            "Missing or incompatible assets detected:\n- "
            + "\n- ".join(missing)
            + "\n\nInstall those files and point Image edit to them with the edit stack overrides before running this model."
        )

    def build_config(self, output_path: str, prompt: str) -> dict:
        return {
            "sd_exe": self.sd_exe_var.get().strip(),
            "output": output_path,
            "prompt": prompt,
            "lora_items": self._collect_lora_items(self.lora_rows),
            "options": self._collect_options(),
        }

    def build_edit_config(self, output_path: str) -> dict:
        refs = [self.edit_refs_list.get(i) for i in range(self.edit_refs_list.size())]
        opts = self._collect_options()
        for key in ("init_img", "strength", "negative_prompt", "ref_images", "mask", "control_image", "control_video", "end_img"):
            opts.pop(key, None)
        opts["diffusion_model"] = self._all_model_map.get(self.edit_model_var.get(), self.model_path_var.get())
        opts["negative_prompt"] = self.edit_negative.get("1.0", "end").strip()
        opts["width"] = int(self.edit_width.get())
        opts["height"] = int(self.edit_height.get())
        opts["steps"] = int(self.edit_steps.get())
        opts["cfg_scale"] = float(self.edit_cfg.get())
        opts["sampling_method"] = self.edit_sampling.get().strip()
        opts["scheduler"] = self.edit_scheduler.get().strip()
        opts["init_img"] = refs[0]
        opts["strength"] = float(self.edit_strength.get())
        seed = self.edit_seed_var.get().strip()
        if seed:
            opts["seed"] = int(seed)
        else:
            opts.pop("seed", None)
        if self.edit_mode_var.get() == "masked":
            opts["mask"] = self._save_edit_mask_if_needed(output_path)
        else:
            opts.pop("mask", None)
        if self.edit_use_ref_images.get():
            opts["ref_images"] = refs
        else:
            opts.pop("ref_images", None)
        if self.edit_use_stack_overrides.get():
            for key in ("model", "vae_path", "clip_l", "clip_g", "t5xxl", "llm_path", "llm_vision"):
                opts.pop(key, None)
            override_pairs = [
                ("model", self.edit_main_model_override.get().strip()),
                ("vae_path", self.edit_vae_override.get().strip()),
                ("clip_l", self.edit_clip_l_override.get().strip()),
                ("clip_g", self.edit_clip_g_override.get().strip()),
                ("t5xxl", self.edit_t5xxl_override.get().strip()),
                ("llm_path", self.edit_llm_override.get().strip()),
                ("llm_vision", self.edit_llm_vision_override.get().strip()),
            ]
            for key, value in override_pairs:
                if value:
                    opts[key] = value
        return {
            "sd_exe": self.sd_exe_var.get().strip(),
            "output": output_path,
            "prompt": self.edit_prompt.get("1.0", "end").strip(),
            "lora_items": self._collect_lora_items(self.edit_lora_rows),
            "options": opts,
        }

    def build_second_pass_config(self, output_path: str, input_path: str, base_cfg: dict) -> dict:
        base_opts = dict(base_cfg.get("options") or {})
        opts = base_opts if self.second_pass_inherit_base_settings.get() else self._collect_options()
        if not self.second_pass_inherit_base_settings.get():
            opts["diffusion_model"] = self._all_model_map.get(self.second_pass_model_var.get(), self.model_path_var.get())
            opts["width"] = int(self.second_pass_width.get())
            opts["height"] = int(self.second_pass_height.get())
            opts["steps"] = int(self.second_pass_steps.get())
            opts["cfg_scale"] = float(self.second_pass_cfg.get())
            opts["sampling_method"] = self.second_pass_sampling.get().strip()
            opts["scheduler"] = self.second_pass_scheduler.get().strip()
        opts["init_img"] = input_path
        opts["strength"] = float(self.second_pass_strength.get())
        if self.second_pass_use_ref_image.get():
            opts["ref_images"] = [input_path]
        else:
            opts.pop("ref_images", None)
        opts["negative_prompt"] = self.second_pass_negative.get("1.0", "end").strip()
        return {
            "sd_exe": self.sd_exe_var.get().strip(),
            "output": output_path,
            "prompt": self.second_pass_prompt.get("1.0", "end").strip(),
            "lora_items": self._collect_lora_items(self.second_pass_lora_rows),
            "options": opts,
        }

    def second_pass_output_path(self, p: str) -> str:
        target = Path(p).expanduser()
        suffix = self.second_pass_suffix_var.get().strip() or "_pass2"
        if target.suffix == "":
            target = target.with_suffix(".png")
        return self.unique_output_path(str(target.with_name(f"{target.stem}{suffix}{target.suffix}")))

    def _run_config(self, cfg: dict, *, step_label: str, use_i2: bool, init_src: str = "", strength: float = 0.0) -> int:
        tmp_path: str | None = None
        out = str(cfg.get("output") or "")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
                json.dump(cfg, tf, indent=2, ensure_ascii=False)
                tmp_path = tf.name
            cmd = [sys.executable, str(BASE_DIR / "zimage_lora_app.py"), "--config-json", tmp_path]
            self._active_phase = step_label
            self.status.set(f"Running {step_label}…")
            self.append_log(f"\n[INFO] {step_label}\n")
            if use_i2:
                self.append_log(f"[INFO] Mode: image-to-image (init={init_src}, strength={strength:.2f})\n")
            else:
                self.append_log("[INFO] Mode: text-to-image\n")
            self.append_log(f"[INFO] Output: {out}\n")
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self.append_log(line if line.endswith("\n") else f"{line}\n")
            rc = self._proc.wait()
            self.append_log(f"\n[DONE] {step_label} exit code: {rc}\n")
            return rc
        finally:
            self._proc = None
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def on_stop(self) -> None:
        self._stop_requested.set()
        if self._proc and self._proc.poll() is None:
            self.append_log("[INFO] Stop: forcing subprocess tree to exit (hard stop)…\n")
            self._kill_proc_tree(self._proc)
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status.set(f"Stopped during {self._active_phase}." if self._active_phase else "Stopped.")

    def on_run(self) -> None:
        active_tab = str(self.left_notebook.select())
        is_edit_tab = active_tab == str(self.tab_edit)
        prompt = self.prompt.get("1.0", "end").strip()
        if not is_edit_tab and not prompt:
            messagebox.showerror("Missing prompt", "Please enter a prompt.")
            return
        if is_edit_tab:
            ep = self.edit_prompt.get("1.0", "end").strip()
            if not ep:
                messagebox.showerror("Missing prompt", "Please enter an edit prompt.")
                return
            if self.edit_refs_list.size() == 0:
                messagebox.showerror("Missing photos", "Add at least one reference photo in Image edit.")
                return
            req_error = self._validate_edit_model_requirements()
            if req_error:
                messagebox.showerror("Image edit requirements", req_error)
                return

        out = self.out_var.get().strip()
        if not out:
            messagebox.showerror("Missing output path", "Please choose where to save output.")
            return
        out = self.unique_output_path(out)

        seed = self.seed_var.get().strip()
        if seed:
            try:
                int(seed)
            except ValueError:
                messagebox.showerror("Invalid seed", "Seed must be an integer.")
                return
        if is_edit_tab:
            edit_seed = self.edit_seed_var.get().strip()
            if edit_seed:
                try:
                    int(edit_seed)
                except ValueError:
                    messagebox.showerror("Invalid edit seed", "Edit seed must be an integer.")
                    return

        if (not is_edit_tab) and self.img2img_enabled.get():
            init_img = self.init_img_var.get().strip()
            if not init_img:
                messagebox.showerror("Image-to-image", "Enable init image mode requires a file path.")
                return
        if (not is_edit_tab) and self.second_pass_enabled.get():
            sp_prompt = self.second_pass_prompt.get("1.0", "end").strip()
            if not sp_prompt:
                messagebox.showerror("Second pass", "Please enter a second-pass prompt or disable automated second pass.")
                return

        cfg = self.build_edit_config(out) if is_edit_tab else self.build_config(out, prompt)

        if is_edit_tab:
            history_snapshot: dict = {
                "mode": "edit",
                "prompt": self.edit_prompt.get("1.0", "end").strip(),
                "negative": self.edit_negative.get("1.0", "end").strip(),
                "output": out,
                "ref_images": [self.edit_refs_list.get(i) for i in range(self.edit_refs_list.size())],
                "lora_items": self._collect_lora_snapshot(self.edit_lora_rows),
                "settings": self._history_edit_settings(cfg),
            }
        else:
            history_snapshot = {
                "mode": "simple",
                "prompt": prompt,
                "negative": self.neg_prompt.get("1.0", "end").strip(),
                "output": out,
                "lora_items": self._collect_active_loras_for_history(),
                "settings": self._history_simple_settings(cfg),
            }

        self.log.delete("1.0", "end")
        self.status.set("Running…")
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._stop_requested.clear()
        self._active_phase = ""

        def worker() -> None:
            try:
                use_i2 = bool(self.img2img_enabled.get()) or is_edit_tab
                if is_edit_tab:
                    init_src = self.edit_refs_list.get(0)
                    strength = float(self.edit_strength.get())
                else:
                    init_src = self.init_img_var.get().strip()
                    strength = float(self.strength_var.get())

                rc = self._run_config(
                    cfg,
                    step_label=(
                        "Step 1/1 - image edit"
                        if is_edit_tab
                        else ("Step 1/2 - base generation" if self.second_pass_enabled.get() else "Step 1/1 - base generation")
                    ),
                    use_i2=use_i2,
                    init_src=init_src,
                    strength=strength,
                )
                base_exists = Path(out).expanduser().exists()
                if rc == 0 and base_exists:
                    hist = {**history_snapshot, "created": datetime.now().isoformat(timespec="seconds")}
                    self.after(0, lambda h=hist: self._append_history(h))
                auto_second_pass = (not is_edit_tab) and bool(self.second_pass_enabled.get())
                if self._stop_requested.is_set() or not auto_second_pass or rc != 0 or not base_exists:
                    self.status.set("Stopped." if self._stop_requested.is_set() else f"Done (exit {rc}).")
                    return

                second_out = self.second_pass_output_path(out)
                second_cfg = self.build_second_pass_config(second_out, out, cfg)
                second_history = {
                    "mode": "second_pass",
                    "enabled": bool(self.second_pass_enabled.get()),
                    "inherit_base_settings": bool(self.second_pass_inherit_base_settings.get()),
                    "prompt": self.second_pass_prompt.get("1.0", "end").strip(),
                    "negative": self.second_pass_negative.get("1.0", "end").strip(),
                    "output": second_out,
                    "ref_images": [out],
                    "lora_items": self._collect_lora_snapshot(self.second_pass_lora_rows),
                    "model": (
                        self.model_var.get().strip()
                        if self.second_pass_inherit_base_settings.get()
                        else self.second_pass_model_var.get().strip()
                    ),
                    "suffix": self.second_pass_suffix_var.get().strip(),
                    "width": int((second_cfg.get("options") or {}).get("width", self.second_pass_width.get())),
                    "height": int((second_cfg.get("options") or {}).get("height", self.second_pass_height.get())),
                    "steps": int((second_cfg.get("options") or {}).get("steps", self.second_pass_steps.get())),
                    "cfg_scale": float((second_cfg.get("options") or {}).get("cfg_scale", self.second_pass_cfg.get())),
                    "sampling_method": str((second_cfg.get("options") or {}).get("sampling_method", self.second_pass_sampling.get())),
                    "scheduler": str((second_cfg.get("options") or {}).get("scheduler", self.second_pass_scheduler.get())),
                    "strength": float(self.second_pass_strength.get()),
                    "settings": self._history_second_pass_settings(second_cfg),
                }
                rc2 = self._run_config(
                    second_cfg,
                    step_label="Step 2/2 - automated second pass",
                    use_i2=True,
                    init_src=out,
                    strength=float(self.second_pass_strength.get()),
                )
                if rc2 == 0 and Path(second_out).expanduser().exists():
                    hist2 = {**second_history, "created": datetime.now().isoformat(timespec="seconds")}
                    self.after(0, lambda h=hist2: self._append_history(h))
                self.status.set("Stopped." if self._stop_requested.is_set() else f"Done (pass 2 exit {rc2}).")
            except Exception as e:
                if self._stop_requested.is_set():
                    self.status.set("Stopped.")
                else:
                    self.append_log(f"[ERROR] {e}\n")
                    self.status.set("Error.")
            finally:
                self._active_phase = ""
                self.run_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)

        self._run_thread = threading.Thread(target=worker, daemon=True)
        self._run_thread.start()


if __name__ == "__main__":
    App().mainloop()
