# pngmodel — Logo Refinement (Real-ESRGAN fine-tune + vectorize)

Take a low-quality logo → output a **high-resolution PNG** and a clean **SVG vector**.

This project fine-tunes **Real-ESRGAN** (a super-resolution / restoration GAN) specifically on
logos, then runs an inference pipeline that:

1. Upscales + cleans the input logo with the fine-tuned model → high-res PNG
2. Traces the cleaned raster into a scalable SVG (vtracer)

Tuned for a single **NVIDIA RTX 3070 Ti (8 GB VRAM)**.

> ⚠️ **Use Python 3.10 or 3.11.** PyTorch 2.1.2 and `basicsr` do **not** ship wheels for
> Python 3.12+. Your system Python (3.14) will fail to install this stack. Create a venv with a
> 3.10/3.11 interpreter:
> ```bash
> py -3.10 -m venv .venv      # or: python3.10 -m venv .venv
> .venv\Scripts\activate
> ```
> If you don't have 3.10/3.11, install it from python.org first.

---

## Why Real-ESRGAN for logos

Real-ESRGAN learns to reverse realistic image degradation (downscaling, blur, JPEG noise).
By default it is trained on photos. Logos have flat colors, hard edges and text, so we
**fine-tune** the official `RealESRGAN_x4plus` weights on a logo dataset. Fine-tuning instead of
training from scratch means we need far less data and GPU time.

We train in two stages, exactly like the official recipe:

| Stage | Model            | Loss            | Purpose                                   | VRAM |
|-------|------------------|-----------------|-------------------------------------------|------|
| 1     | RealESRNet (PSNR)| L1              | Stable base, learns logo structure        | low  |
| 2     | RealESRGAN (GAN) | L1 + percep + GAN | Adds sharp, crisp detail                | high |

Stage 2 is the memory-heavy one (it adds a U-Net discriminator + VGG perceptual loss), which is
why the configs use small patch sizes and batch sizes.

---

## Project layout

```
pngmodel/
├── README.md
├── requirements.txt
├── configs/
│   ├── finetune_realesrnet_logos.yml   # Stage 1 (L1 only)
│   └── finetune_realesrgan_logos.yml   # Stage 2 (GAN)
├── scripts/
│   ├── download_weights.py             # fetch pretrained x4plus weights
│   ├── prepare_data.py                 # build HR dataset + meta_info
│   ├── render_svgs.py                  # (optional) make HR logos from free SVG sets
│   └── train.py                        # launches basicsr training
├── src/
│   ├── infer.py                        # upscale → high-res PNG (+ optional SVG)
│   └── vectorize.py                    # raster PNG → SVG
├── app.py                              # Gradio UI: drop logo → get PNG + SVG
├── data/
│   ├── logos_hr/                       # your high-quality source logos go here
│   └── meta_info/                      # generated file lists
└── weights/                            # pretrained + your fine-tuned .pth files
```

---

## End-to-end run (summary)

> Full step-by-step with exact commands is in the "How to run" section below.

```bash
# 0. install
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 1. get pretrained weights
python scripts/download_weights.py

# 2. put high-quality logos in data/logos_hr/ (or render some)
python scripts/render_svgs.py            # optional, builds a starter dataset
python scripts/prepare_data.py           # makes multiscale crops + meta_info

# 3. fine-tune  (stage 1 then stage 2)
python scripts/train.py --config configs/finetune_realesrnet_logos.yml
python scripts/train.py --config configs/finetune_realesrgan_logos.yml

# 4. infer: low-quality logo -> high-res PNG + SVG
python src/infer.py -i path/to/logo.png -o results/ --svg

# or the UI
python app.py
```

---

## How to run (detailed)

### 1. Environment
- Install the CUDA build of PyTorch that matches your driver (see `requirements.txt` notes).
- Verify the GPU is visible:
  ```bash
  python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
  ```

### 2. Data
You need **high-quality** logos as the "ground truth". Real-ESRGAN generates the low-quality
versions on the fly during training, so you only supply the good ones.

- Drop PNG/JPG logos into `data/logos_hr/`. A few hundred is enough to start; a few thousand is
  better. Prefer large, clean images (≥ 512 px).
- `scripts/render_svgs.py` can bootstrap a dataset by rendering free SVG icon/brand sets to
  high-res PNGs (great GT because vectors are perfectly crisp).
- `scripts/prepare_data.py` creates multi-scale copies and a `meta_info` list the trainer reads.

### 3. Fine-tune (8 GB VRAM)
- Stage 1 builds a stable base; stage 2 adds the crisp GAN detail.
- If you hit CUDA out-of-memory, lower `batch_size_per_gpu` (then `gt_size`) in the config.
- Checkpoints land in `experiments/`. Watch progress with TensorBoard.

### 4. Inference
- `src/infer.py` loads your fine-tuned generator, upscales the input (default x4), saves a
  high-res PNG, and with `--svg` also traces an SVG via vtracer.

### Tips for 8 GB
- Start from the configs as-is; they are already conservative.
- Close other GPU apps (browsers with HW accel, games).
- Use `--no-half` only if you see color/precision artifacts; fp16 saves memory.

See the per-file comments for details on each parameter.
# smallLLM
