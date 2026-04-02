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
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BASE_DIR = Path(__file__).resolve().parent
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


def list_loras() -> list[str]:
    if not LORAS_DIR.exists():
        return [""]
    names = sorted(p.name for p in LORAS_DIR.glob("*.safetensors"))
    return names if names else [""]


def default_llm() -> str:
    p = AUX_DIR / "Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf"
    return str(p) if p.exists() else ""


def default_vae() -> str:
    p = AUX_DIR / "ae.safetensors"
    return str(p) if p.exists() else ""


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Z-Image Local UI — sd.exe")
        self.geometry("1200x860")

        self._proc: subprocess.Popen | None = None
        self._run_thread: threading.Thread | None = None
        self.lora_rows: list[dict] = []
        self._str_opts: dict[str, tk.StringVar] = {}
        self._bool_opts: dict[str, tk.BooleanVar] = {}

        models = list_models()
        loras = list_loras()

        self.model_var = tk.StringVar(value=models[0][0])
        self.model_path_var = tk.StringVar(value=models[0][1])
        self._model_map = {name: path for name, path in models}

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

        notebook = ttk.Notebook(left_outer)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_simple = ttk.Frame(notebook)
        tab_adv = ttk.Frame(notebook)
        notebook.add(tab_simple, text="Simple run")
        notebook.add(tab_adv, text="Advanced")

        self._build_simple_tab(tab_simple, models, loras)
        self._build_advanced_tab(tab_adv)

        run_row = ttk.Frame(left_outer)
        run_row.pack(fill=tk.X, pady=(10, 0))
        self.run_btn = ttk.Button(run_row, text="Run", command=self.on_run)
        self.run_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))
        self.stop_btn = ttk.Button(run_row, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

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

    def _build_simple_tab(self, tab: ttk.Frame, models: list, loras: list[str]) -> None:
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

        ttk.Label(q, text="Z-Image diffusion (.gguf)").pack(anchor="w")
        model_cb = ttk.Combobox(q, state="readonly", values=[m[0] for m in models], textvariable=self.model_var, width=40)
        model_cb.pack(fill=tk.X)
        model_cb.bind("<<ComboboxSelected>>", lambda _e: self.model_path_var.set(self._model_map.get(self.model_var.get(), self.model_path_var.get())))

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

    def _bool_chk(self, parent: ttk.Widget, text: str, key: str) -> None:
        ttk.Checkbutton(parent, text=text, variable=self._bool_opt(key)).pack(anchor="w")

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

    def add_lora_row(self, default_name: str, default_weight: float) -> None:
        rowf = ttk.Frame(self.lora_container)
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

        self.lora_rows.append(row)
        clamp_and_apply()
        self.remove_lora_btn.config(state=(tk.NORMAL if len(self.lora_rows) > 1 else tk.DISABLED))
        self.add_lora_btn.config(state=(tk.NORMAL if len(self.lora_rows) < 6 else tk.DISABLED))

    def remove_last_lora_row(self) -> None:
        if len(self.lora_rows) <= 1:
            return
        row = self.lora_rows.pop()
        row["frame"].destroy()
        self.remove_lora_btn.config(state=(tk.NORMAL if len(self.lora_rows) > 1 else tk.DISABLED))
        self.add_lora_btn.config(state=(tk.NORMAL if len(self.lora_rows) < 6 else tk.DISABLED))

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

    def build_config(self, output_path: str, prompt: str) -> dict:
        lora_items = []
        for row in self.lora_rows:
            name = row["name_var"].get().strip()
            w = float(row["weight_var"].get())
            if name and w != 0:
                lora_items.append({"path": str(LORAS_DIR / name), "weight": w})

        return {
            "sd_exe": self.sd_exe_var.get().strip(),
            "output": output_path,
            "prompt": prompt,
            "lora_items": lora_items,
            "options": self._collect_options(),
        }

    def on_stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status.set("Stopped.")

    def on_run(self) -> None:
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            messagebox.showerror("Missing prompt", "Please enter a prompt.")
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

        if self.img2img_enabled.get():
            init_img = self.init_img_var.get().strip()
            if not init_img:
                messagebox.showerror("Image-to-image", "Enable init image mode requires a file path.")
                return

        cfg = self.build_config(out, prompt)

        self.log.delete("1.0", "end")
        self.status.set("Running…")
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        def worker() -> None:
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
                    json.dump(cfg, tf, indent=2, ensure_ascii=False)
                    tmp_path = tf.name
                cmd = [sys.executable, str(BASE_DIR / "zimage_lora_app.py"), "--config-json", tmp_path]
                use_i2 = bool(self.img2img_enabled.get())
                if use_i2:
                    self.append_log(
                        f"[INFO] Mode: image-to-image (init={self.init_img_var.get().strip()}, "
                        f"strength={float(self.strength_var.get()):.2f})\n"
                    )
                else:
                    self.append_log("[INFO] Mode: text-to-image\n")
                self.append_log(f"[INFO] Output: {out}\n")
                self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    self.append_log(line if line.endswith("\n") else f"{line}\n")
                rc = self._proc.wait()
                self.append_log(f"\n[DONE] Exit code: {rc}\n")
                self.status.set(f"Done (exit {rc}).")
            except Exception as e:
                self.append_log(f"[ERROR] {e}\n")
                self.status.set("Error.")
            finally:
                if tmp_path:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except OSError:
                        pass
                self.run_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)

        self._run_thread = threading.Thread(target=worker, daemon=True)
        self._run_thread.start()


if __name__ == "__main__":
    App().mainloop()
