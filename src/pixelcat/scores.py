import numpy as np
from PIL import Image, ImageFilter


def directed_palette_distance(results_a, results_b):
    if not results_a or not results_b:
        return None

    distance = 0.0
    for color_a in results_a:
        rgb_a = np.array(color_a["rgb"], dtype=float)
        nearest = min(
            np.linalg.norm(rgb_a - np.array(color_b["rgb"], dtype=float))
            for color_b in results_b
        )
        distance += nearest * (color_a["percent"] / 100)

    return float(distance)


def palette_distance(results_a, results_b):
    forward = directed_palette_distance(results_a, results_b)
    backward = directed_palette_distance(results_b, results_a)
    if forward is None or backward is None:
        return None

    return (forward + backward) / 2


def distance_to_similarity(distance):
    if distance is None:
        return None

    return max(0.0, 100 - distance / 441.67 * 100)


def color_mismatch_score(cat_results, jacket_results):
    distance = palette_distance(cat_results, jacket_results)
    if distance is None:
        return None

    return min(100, distance / 441.67 * 100)


def loaf_score(mask):
    if not mask.any():
        return {"score": 0.0, "circularity": 0.0, "aspect": 0.0, "coverage": 0.0}

    rows, cols = np.where(mask)
    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1
    crop = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    area = int(crop.sum())

    mask_img = Image.fromarray((crop * 255).astype(np.uint8), mode="L")
    edges = np.asarray(mask_img.filter(ImageFilter.FIND_EDGES)) > 0
    perimeter = max(1, int(edges.sum()))

    circularity = min(1.0, (4 * np.pi * area) / (perimeter * perimeter))
    aspect_ratio = width / max(1, height)
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 1.55) / 1.55)
    coverage = area / max(1, width * height)
    coverage_score = min(1.0, coverage / 0.75)

    score = 10 * (0.45 * circularity + 0.35 * aspect_score + 0.20 * coverage_score)
    return {
        "score": round(float(score), 1),
        "circularity": round(float(circularity), 3),
        "aspect": round(float(aspect_ratio), 2),
        "coverage": round(float(coverage), 3),
    }

