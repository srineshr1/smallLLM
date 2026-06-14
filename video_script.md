# pngmodel — AI Logo Refiner: LinkedIn Video Script
# ~2.5–3 minutes

────────────────────────────────────────────────────────────────
## HOOK (0:00–0:10)
────────────────────────────────────────────────────────────────

You know when you have a logo and it's tiny, blurry, full of JPEG artifacts,
and you just need a clean high-res version? That's what I built an AI to do.

────────────────────────────────────────────────────────────────
## THE PROBLEM (0:10–0:30)
────────────────────────────────────────────────────────────────

Super-resolution models like Real-ESRGAN exist. They take blurry images and
upscale them 4x. The catch — they're trained on photos. Landscapes, faces,
textures. Feed them a logo and they hallucinate fake fabric grains and noise
onto your flat color shapes. That's not what a logo needs.

So I decided: fine-tune Real-ESRGAN specifically on logos. Make it understand
flat colors, hard edges, and text. Then add a vectorizer on top so the output
is an actual SVG — infinitely scalable, no pixels.

────────────────────────────────────────────────────────────────
## THE STACK (0:30–0:50)
────────────────────────────────────────────────────────────────

Here's what I used:
- Real-ESRGAN as the base model — it already knows how to reverse blur, noise,
  and downscaling. I'm not training from scratch, I'm teaching it a new domain.
- BasicSR as the training framework — handles data loading, degradation
  simulation, checkpointing.
- PyTorch with CUDA on a single RTX 3070 Ti with 8 GB of VRAM.
- CairoSVG to render clean vector logos into training data, and vtracer + OpenCV
  to trace the upscaled output back into SVG.

All of this runs on consumer hardware. No cloud, no A100s.

────────────────────────────────────────────────────────────────
## THE MODEL (0:50–1:10)
────────────────────────────────────────────────────────────────

The generator is an RRDBNet — 23 residual-in-residual dense blocks with about
17 million parameters doing 4x upscaling. Stage two adds a U-Net discriminator
with spectral normalization — ~4 million parameters that judges pixel-by-pixel
whether the output is real or fake.

I train in two stages, exactly like the official recipe:
Stage 1 — L1 loss only. Builds a stable, structurally correct base.
Stage 2 — Three losses combined: L1 for color fidelity, VGG perceptual for
high-level feature matching, and GAN loss for crispness.

────────────────────────────────────────────────────────────────
## THE JOURNEY (1:10–1:50)
────────────────────────────────────────────────────────────────

This is where it gets interesting. My first dataset was 3,400 icons from
Simple Icons. Over 10,000 training images after multi-scale augmentation.
Trained both stages — and the output was grayscale with this weird embossed
metallic texture.

I was frustrated. But here's the thing: the model wasn't broken. It worked
exactly as trained. Simple Icons are all single-color black silhouettes by
design. Every single image in the training set was monochrome. The model
faithfully learned "logos = grayscale." It never saw color, so it couldn't
produce it. And the GAN — give it a flat shape with no texture, it invents
texture. Hence the embossed metal look.

Second attempt: scraped JPGs. Rejected immediately — full of compression
artifacts, mockups, template text. Bad ground truth means bad output.

Third attempt — and the one that worked: I cloned the gilbarbara/logos repo —
1,863 full-color brand SVGs. Rendered them to clean PNGs. Verified the mix:
75% colored, 25% genuinely monochrome. That's a realistic distribution.

Then I retuned the loss weights. This is important — the default weights are
for photos. For logos I raised the pixel loss weight to 1.5 to enforce color
fidelity, dropped perceptual to 0.5 so it stops inventing texture, and cut
GAN loss to 0.03 to kill the embossed effect. Now the model upscales logos
as logos — sharp, clean, color-accurate.

────────────────────────────────────────────────────────────────
## TRAINING STATS (1:50–2:10)
────────────────────────────────────────────────────────────────

Numbers, because people ask:
- 1,860 source logos → 5,580 training images with multi-scale augmentation
- Stage 1: 20,000 iterations — 6 hours 21 minutes
- Stage 2: 20,000 iterations — 4 hours 37 minutes
- Total training: about 11 hours on a single GPU
- Pixel loss went from 0.025 at start to 0.030 at the end — GAN training
  is volatile by nature, but the visual quality tells the real story.

────────────────────────────────────────────────────────────────
## OUTPUT (2:10–2:30)
────────────────────────────────────────────────────────────────

The end-to-end pipeline: drop in a low-quality logo → the model upscales it
4x with clean edges and faithful colors → then the vectorizer traces it into
a scalable SVG using k-means color quantization and contour detection.

I also built a Gradio web UI so anyone can drag and drop a logo and get both
the high-res PNG and the SVG back without touching a terminal.

────────────────────────────────────────────────────────────────
## THE TAKEAWAY (2:30–2:50)
────────────────────────────────────────────────────────────────

Here's what I want you to take from this:

The model didn't fail because the AI was broken. It failed because I fed it
black-and-white logos and asked for color. The fix wasn't a better architecture
or more hyperparameter tuning. The fix was better data.

Your model is only as good as what you show it. That's true whether you're
fine-tuning a GAN or training an LLM. The data ceiling is real.

────────────────────────────────────────────────────────────────
## CLOSE (2:50–3:00)
────────────────────────────────────────────────────────────────

If you're working on image restoration, domain adaptation, or just curious
about making GANs do something they weren't designed for — drop a comment.
Happy to share more details about the configs, the training runs, or the code.

────────────────────────────────────────────────────────────────

## B-ROLL / VISUAL SUGGESTIONS BY TIMESTAMP

0:00  → Show a blurry, artifact-filled logo next to the clean output
0:15  → Side-by-side: stock Real-ESRGAN output (textured, wrong) vs fine-tuned (clean)
0:40  → Scroll through requirements.txt quickly, then highlight the key packages
0:55  → Show architecture diagram (RRDBNet blocks, x4 upscaling, U-Net discriminator)
1:10  → Show the grayscale + embossed output from Attempt 1, then the correct color output
1:45  → Overlay: the three loss weight changes (1.0→1.5, 1.0→0.5, 0.1→0.03)
2:00  → TensorBoard loss curves going down
2:15  → Screen recording: drag logo into Gradio UI → PNG + SVG appear
2:35  → Text on screen: "The data ceiling is real."
2:50  → Project repo URL / GitHub link

## KEY NUMBERS TO FLASH ON SCREEN (lower thirds)

- 16.7M generator parameters
- 5,580 training images
- 11 hours total training
- 4x upscaling
- 8 GB VRAM (consumer GPU)
- 3 dataset attempts, 1 working
