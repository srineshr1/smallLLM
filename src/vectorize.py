import sys
import cv2
import numpy as np


def raster_to_svg(png_path: str, svg_path: str, n_colors: int = 16, **kwargs):
    img = cv2.imread(png_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {png_path}")
    h, w = img.shape[:2]

    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    labels = labels.flatten()

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for i in range(n_colors):
        b, g, r = centers[i].tolist()
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        if hex_color == "#ffffff":
            continue

        mask = (labels == i).reshape(h, w).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 3:
                continue
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            path_d = "M " + " L ".join(f"{p[0][0]},{p[0][1]}" for p in approx) + " Z"
            svg_parts.append(f'<path d="{path_d}" fill="{hex_color}" stroke="{hex_color}" stroke-width="0.3"/>')

    svg_parts.append("</svg>")
    with open(svg_path, "w") as f:
        f.write("\n".join(svg_parts))


if __name__ == "__main__":
    raster_to_svg(sys.argv[1], sys.argv[2])
