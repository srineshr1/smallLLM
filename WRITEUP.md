# pngmodel — AI Logo Refiner: Full Project Writeup

> A video-friendly walkthrough of what this project is, what was built, the models used,
> and the debugging journey from a failed run to a working one.

**Goal:** take a low-quality logo and produce a **high-resolution PNG** + a clean **SVG vector**,
by fine-tuning a super-resolution AI model specifically on logos.

**Hardware:** a single **NVIDIA RTX 3070 Ti (8 GB VRAM)**.

---

## 1. The core technology

The model is **Real-ESRGAN** — a super-resolution GAN that learns to reverse real-world image
degradation (downscaling, blur, JPEG noise). It's normally trained on photos, so this project is
about **fine-tuning** it on logos instead.

- Training framework: **BasicSR**
- **PyTorch 2.1.2 (CUDA 12.1)** inside a **Python 3.11** virtual environment
  - (System Python is 3.14, which is too new for this stack — that's why we use `.venv`.)

---

## 2. The models downloaded

From the official Real-ESRGAN release page (`xinntao/Real-ESRGAN`), into `weights/`:

| File | Size | Role |
|------|------|------|
| `RealESRNet_x4plus.pth` | 64 MB | PSNR-oriented generator — the **base** for Stage 1 fine-tuning |
| `RealESRGAN_x4plus.pth` | 64 MB | The GAN generator (also the fallback inference model) |

The discriminator was **trained from scratch** (`pretrain_network_d: ~`); no pretrained netD needed
for fine-tuning.

---

## 3. The architecture

- **Generator: RRDBNet** — Residual-in-Residual Dense Blocks. 23 blocks, 64 feature channels,
  **×4** upscaling. This network does the actual upscaling.
- **Discriminator: U-Net with Spectral Normalization** — judges "real vs. fake" per pixel, pushing
  the generator toward sharp detail.
- **Three loss functions** combined in Stage 2:
  - **L1 (pixel) loss** — keep colors/shapes faithful
  - **VGG19 perceptual loss** — match high-level features
  - **GAN loss** — add crispness

---

## 4. The two-stage training recipe

Mirrors the official Real-ESRGAN recipe:

| Stage | Model | Loss | Notes | VRAM settings |
|-------|-------|------|-------|---------------|
| 1 | RealESRNet | L1 only | Stable base | batch 6, 256px patches |
| 2 | RealESRGAN | L1 + VGG perceptual + GAN | Adds crisp detail | batch 4, 256px patches |

Each stage: ~20,000 iterations, roughly **4–5 hours** on the 3070 Ti.

**Pipeline:**
`download_weights.py` → `render_svgs.py` → `prepare_data.py` → `train.py` (×2 stages) →
`infer.py` / `app.py` (Gradio web UI).

---

## 5. The dataset journey (the real story)

### Attempt 1 — Simple Icons (FAILED)
- Used the Simple Icons set: **3,443 logos → ~10,300 training images** (multiscale augmentation).
- Both stages trained successfully... but output was **grayscale with a weird embossed/metallic
  texture**.
- **Diagnosis:** checked the data — **60/60 sampled logos were monochrome** (Simple Icons are
  single-color black silhouettes *by design*). Training was healthy; the model just faithfully
  learned "logos = grayscale." It had **never seen color**, so it desaturated everything. The
  embossed texture was the GAN inventing fake detail on flat shapes.
- **Key lesson:** *the training data is the ceiling.* The model can only output what it was shown.

### Attempt 2 — scraped colored JPGs (REJECTED before training)
- A folder of ~2,000 colored logo JPGs. Rejected because: JPEG compression artifacts baked in, plus
  collages, mockups, "COMPANY NAME" template text, and esports-style glows. Bad ground truth.

### Attempt 3 — gilbarbara/logos color vectors (THE FIX)
- Cloned **gilbarbara/logos**: **1,863 full-COLOR brand SVGs**.
- Rendered to clean PNGs (aspect ratio preserved, padded on white) → **1,860 images** →
  **5,580 training images** with multiscale.
- Verified: **75% colored**, 25% genuinely monochrome brand logos — a realistic, healthy mix.
- **Retuned Stage 2 for flat logo art** (old weights were tuned for photos):

  | Loss | Before | After | Why |
  |------|--------|-------|-----|
  | Pixel L1 | 1.0 | **1.5** | trust the real colors more |
  | Perceptual | 1.0 | **0.5** | less texture invention |
  | GAN | 0.1 | **0.03** | stops the embossed-texture hallucination |

---

## 6. Current state / next step

Colored dataset is built and both configs updated. Next: **retrain** both stages on the color data.

```powershell
.venv\Scripts\activate

# (optional) archive the old monochrome-trained runs first
Rename-Item experiments\finetune_RealESRNetx4plus_logos  finetune_RealESRNetx4plus_logos_mono
Rename-Item experiments\finetune_RealESRGANx4plus_logos  finetune_RealESRGANx4plus_logos_mono

python scripts/train.py --config configs/finetune_realesrnet_logos.yml   # Stage 1
python scripts/train.py --config configs/finetune_realesrgan_logos.yml   # Stage 2
```

**Expected result:** color is preserved, and no more embossed texture.

Quick checkpoint after Stage 1 (no GAN yet, so it'll be clean but a touch soft):

```powershell
python src/infer.py -i your_logo.png -o results\ -m experiments\finetune_RealESRNetx4plus_logos\models\net_g_20000.pth
```

---

## The one-line takeaway

> It didn't fail because the AI was broken — it failed because it was fed black-and-white logos and
> asked to produce color. The fix wasn't a better model, it was **better data.**

---

## Dataset sources & licenses (mention in the video)

- **gilbarbara/logos** — full-color brand SVGs (check the repo's license/terms before redistribution).
- **Simple Icons** — monochrome brand icons (the cause of Attempt 1's failure; good for mono tasks,
  not color).
- Pretrained weights — **Real-ESRGAN** by Xintao Wang et al. (BSD-3-Clause).
