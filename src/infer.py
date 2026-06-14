import argparse
import os
import os.path as osp
import sys
import types

import cv2


def _patch_torchvision_functional_tensor():
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ModuleNotFoundError:
        import torchvision.transforms.functional as F

        m = types.ModuleType("torchvision.transforms.functional_tensor")
        m.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = m


_patch_torchvision_functional_tensor()

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

DEFAULT_MODEL = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "weights",
    "RealESRGAN_x4plus.pth",
)


def build_upsampler(
    model_path: str,
    scale: int = 4,
    tile: int = 0,
    half: bool = True,
):
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )
    return RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=tile,
        half=half,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Upscale a logo to high-res PNG (optionally trace SVG)."
    )
    parser.add_argument("-i", "--input", required=True, help="Input logo path")
    parser.add_argument("-o", "--output", default="results", help="Output directory")
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, help="Generator .pth path"
    )
    parser.add_argument("--svg", action="store_true", help="Also trace SVG via vtracer")
    parser.add_argument("--scale", type=float, default=4, help="Upscale factor")
    parser.add_argument("--tile", type=int, default=0, help="Tile size (0=off)")
    parser.add_argument("--no-half", action="store_true", help="Disable fp16")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    upsampler = build_upsampler(args.model, scale=args.scale, tile=args.tile, half=not args.no_half)

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read {args.input}")
        sys.exit(1)

    output, _ = upsampler.enhance(img, outscale=args.scale)

    stem = osp.splitext(osp.basename(args.input))[0]
    png_path = osp.join(args.output, f"{stem}_x{args.scale}.png")
    cv2.imwrite(png_path, output)
    print(f"PNG -> {png_path}")

    if args.svg:
        from vectorize import raster_to_svg

        svg_path = osp.join(args.output, f"{stem}.svg")
        raster_to_svg(png_path, svg_path)
        print(f"SVG -> {svg_path}")


if __name__ == "__main__":
    main()
