# WaveSpeed Z-Image App (local UI)

**Repository:** [github.com/njpon9/WaveSpeed-ZImage-App](https://github.com/njpon9/WaveSpeed-ZImage-App)

Tkinter front-end for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (`sd.exe`) with bundled paths under this repo: `bin/`, `models/stable-diffusion/`, `LoRas/`.

## Why the big files are not in Git

GitHub rejects blobs **> 100 MB**; Z-Image GGUF + Qwen + VAE + `stable-diffusion.dll` are **multi‑GB**. This repo tracks **code only**; `.gitignore` excludes `bin/`, `models/`, and `LoRas/*.safetensors`. After clone, restore those folders locally (script below or manual copy).

## Quick start (Windows, after assets exist)

1. Install **Python 3.10+** (include **tcl/tk** / IDLE on Windows so `tkinter` works).
2. `pip install pillow`
3. Populate **`bin/`**, **`models/stable-diffusion/`** (with **`auxiliary/`**), and **`LoRas/`** — see [Bundle layout](#bundle-layout).
4. Double-click **`start-ui.bat`** or: `py -3 zimage_ui.py`
5. Optional check: `py -3 smoke_bundle.py` (full) or `py -3 smoke_bundle.py --reuse-face` (inpaint/img2img only).

## Bundle layout

| Path | Contents |
|------|-----------|
| `bin/` | Full **sd-bin** from WaveSpeed: `sd.exe`, `stable-diffusion.dll`, `sd-cli.exe`, etc. |
| `models/stable-diffusion/` | e.g. `z-image-turbo-q8_0.gguf` (or other z-image `*.gguf`) |
| `models/stable-diffusion/auxiliary/` | `ae.safetensors` (VAE), `Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf` (or matching Qwen3 GGUF name) |
| `LoRas/` | Your `.safetensors` LoRAs (e.g. `zimage_lightslider.safetensors`) |

The UI defaults to `BASE_DIR / "bin" / "sd.exe"` and `BASE_DIR / "models" / "stable-diffusion"` — no AppData paths.

## Replicate the bundle on a Windows PC (WaveSpeed Desktop installed)

From repo root in **PowerShell**:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force   # once, if scripts are blocked
.\scripts\copy_wavespeed_bundle.ps1
```

That copies `%AppData%\wavespeed-desktop\sd-bin` → `.\bin` and `...\models\stable-diffusion` → `.\models\stable-diffusion`. **LoRAs** are not copied by default (large/custom); add yours under `LoRas\` or extend the script.

## Replicate without WaveSpeed (manual / other machine)

1. **Binary** — Build or download stable-diffusion.cpp for your OS from [releases](https://github.com/leejet/stable-diffusion.cpp/releases). On Windows, place `sd.exe` + shipped DLLs in `bin\`. On Linux/macOS use the platform binary and either name it `sd` and adjust **Advanced → sd.exe path** in the UI, or symlink as needed.

2. **Weights** — Official pointers are in upstream docs, e.g. [Z-Image (stable-diffusion.cpp)](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/z_image.md): Z-Image-Turbo GGUF, Qwen3 4B text encoder, and VAE (`ae`) paths listed there. Download into `models\stable-diffusion\` and `models\stable-diffusion\auxiliary\` to match filenames your UI expects (or pick files in **Advanced**).

3. **LoRAs** — Download `.safetensors` into `LoRas\` (Civitai / Hugging Face / your own training).

## For Cursor / AI assistants

When the user clones this repo and wants a **working** app:

1. **Do not** expect `bin/`, `models/`, or LoRA weights in the repo; they are gitignored.
2. On **Windows** with WaveSpeed Desktop: run `.\scripts\copy_wavespeed_bundle.ps1` from the repo root, then ensure `LoRas\` contains required `.safetensors`.
3. Otherwise: follow **Replicate without WaveSpeed** and populate the [Bundle layout](#bundle-layout).
4. **Z-Image image edit:** keep **“Also pass … (-r)”** **off** for Z-Image (sd.cpp can assert with `-r` + Z-Image). Use higher **strength** / **steps** for inpaint and whole-image edits; masked runs set `img_cfg_scale` in code when Advanced does not override it.

## Optional: Git LFS

If you control a private repo and want binaries in Git, use [Git LFS](https://git-lfs.com/) for `bin/*` and large `*.gguf` / `*.safetensors`. Public GitHub LFS bandwidth is limited; many teams instead attach a **release .zip** with `bin` + `models` and keep the code repo small.

## License / third-party

Application code here is project-specific. `sd.exe`, models, and LoRAs are **third-party**; keep their respective licenses when redistributing binaries or weights.
