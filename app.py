"""
app.py — Gradio UI for logo refinement.

Drop a low-quality logo, get back a high-res PNG and (optionally) an SVG vector.

Run:
    python app.py
then open the printed local URL (default http://127.0.0.1:7860).

By default it loads weights/RealESRGAN_x4plus.pth. Point MODEL_PATH (env var or the
textbox in the UI) at your fine-tuned generator for logo-specialized results, e.g.
    experiments/finetune_RealESRGANx4plus_logos/models/net_g_20000.pth
"""

import os
import os.path as osp
import sys
import tempfile

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "src"))

import cv2
import gradio as gr

from infer import build_upsampler

FINE_TUNED_MODEL = osp.join(
    osp.dirname(osp.abspath(__file__)),
    "experiments",
    "finetune_RealESRGANx4plus_logos",
    "models",
    "net_g_20000.pth",
)

DEFAULT_MODEL = FINE_TUNED_MODEL if osp.isfile(FINE_TUNED_MODEL) else osp.join(
    osp.dirname(osp.abspath(__file__)), "weights", "RealESRGAN_x4plus.pth"
)
from vectorize import raster_to_svg

# Cache one upsampler per model path so we don't reload weights every click.
_UPSAMPLER_CACHE = {}


def get_upsampler(model_path: str, tile: int, no_half: bool):
    key = (model_path, tile, no_half)
    if key not in _UPSAMPLER_CACHE:
        _UPSAMPLER_CACHE[key] = build_upsampler(
            model_path, scale=4, tile=tile, half=not no_half
        )
    return _UPSAMPLER_CACHE[key]


def refine(image, model_path, outscale, tile, make_svg, no_half):
    if image is None:
        raise gr.Error("Please upload a logo image first.")
    model_path = (model_path or "").strip() or DEFAULT_MODEL
    if not osp.isfile(model_path):
        raise gr.Error(
            f"Model not found: {model_path}\n"
            f"Run scripts/download_weights.py or point to your fine-tuned net_g_*.pth."
        )

    upsampler = get_upsampler(model_path, int(tile), bool(no_half))

    # Gradio gives RGB(A); cv2 / RealESRGANer work in BGR(A).
    if image.shape[-1] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        output, _ = upsampler.enhance(bgr, outscale=float(outscale))
    except RuntimeError as e:
        raise gr.Error(f"Upscale failed (try a smaller tile size). {e}")

    tmpdir = tempfile.mkdtemp(prefix="pngmodel_")
    png_path = osp.join(tmpdir, "refined.png")
    cv2.imwrite(png_path, output)

    # Convert back to RGB(A) for display in the UI.
    if output.shape[-1] == 4:
        preview = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
    else:
        preview = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    svg_path = None
    if make_svg:
        svg_path = osp.join(tmpdir, "refined.svg")
        raster_to_svg(png_path, svg_path)

    return preview, png_path, svg_path


def build_ui():
    with gr.Blocks(title="pngmodel — Logo Refiner") as demo:
        gr.Markdown(
            "# Logo Refiner\n"
            "Upload a low-quality logo → get a high-res **PNG** and optional **SVG** vector.\n"
            "Fine-tuned Real-ESRGAN (x4) + vtracer."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label="Input logo", type="numpy", image_mode="RGBA")
                model_path = gr.Textbox(
                    label="Model weights (.pth)",
                    value=os.environ.get("MODEL_PATH", DEFAULT_MODEL),
                    info="Use your fine-tuned net_g_*.pth for best logo results.",
                )
                with gr.Row():
                    outscale = gr.Slider(1, 4, value=4, step=1, label="Upscale factor")
                    tile = gr.Slider(0, 512, value=0, step=64,
                                     label="Tile (0=off; raise if OOM)")
                with gr.Row():
                    make_svg = gr.Checkbox(value=True, label="Also produce SVG")
                    no_half = gr.Checkbox(value=False, label="Disable fp16 (--no-half)")
                btn = gr.Button("Refine", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Refined (high-res)")
                out_png = gr.File(label="Download PNG")
                out_svg = gr.File(label="Download SVG")

        btn.click(
            refine,
            inputs=[inp, model_path, outscale, tile, make_svg, no_half],
            outputs=[out_img, out_png, out_svg],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
