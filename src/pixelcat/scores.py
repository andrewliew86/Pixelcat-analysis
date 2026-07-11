import numpy as np


def rgb_to_lab(rgb):
    """Convert sRGB values to CIE Lab (D65), where distance is perceptual."""
    values = np.asarray(rgb, dtype=float) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array(
        [[0.4124564, 0.2126729, 0.0193339],
         [0.3575761, 0.7151522, 0.1191920],
         [0.1804375, 0.0721750, 0.9503041]]
    )
    scaled = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(scaled > 0.008856, np.cbrt(scaled), 7.787 * scaled + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def directed_palette_distance(results_a, results_b):
    if not results_a or not results_b:
        return None

    distance = 0.0
    for color_a in results_a:
        lab_a = rgb_to_lab(color_a["rgb"])
        nearest = min(
            np.linalg.norm(lab_a - rgb_to_lab(color_b["rgb"]))
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

    return float(100 * np.exp(-((distance / 32) ** 1.35)))


def color_mismatch_score(cat_results, jacket_results):
    # Visibility is directional: every cat-fur color needs a close fabric color;
    # colors present only in the jacket should not make the fur look riskier.
    distance = directed_palette_distance(cat_results, jacket_results)
    if distance is None:
        return None

    return float(100 * (1 - np.exp(-((distance / 28) ** 1.35))))


def loaf_score(mask):
    if not mask.any():
        return {
            "score": 0.0,
            "circularity": 0.0,
            "aspect": 0.0,
            "coverage": 0.0,
            "compactness": 0.0,
        }

    rows, cols = np.where(mask)
    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1
    crop = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    area = int(crop.sum())

    padded = np.pad(crop, 1, constant_values=False)
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    vertical = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    perimeter = max(1, horizontal + vertical)

    circularity = min(1.0, (4 * np.pi * area) / (perimeter * perimeter))
    # PCA makes the aspect ratio independent of camera rotation.
    points = np.column_stack((cols, rows)).astype(float)
    eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
    aspect_ratio = float(np.sqrt(max(eigenvalues) / max(1e-9, min(eigenvalues))))
    aspect_score = float(np.exp(-((aspect_ratio - 1.45) / 0.65) ** 2))
    coverage = area / max(1, width * height)
    coverage_score = min(1.0, coverage / 0.72)

    # A loaf has a filled-in body. Thin legs and tails create rows/columns with
    # much shorter spans than the main body, which this robust score penalizes.
    row_spans = np.array([np.ptp(np.flatnonzero(row)) + 1 for row in crop if row.any()])
    col_spans = np.array([np.ptp(np.flatnonzero(col)) + 1 for col in crop.T if col.any()])
    span_score = min(
        np.percentile(row_spans, 25) / max(1, np.percentile(row_spans, 75)),
        np.percentile(col_spans, 25) / max(1, np.percentile(col_spans, 75)),
    )

    score = 10 * (
        0.25 * min(1.0, circularity / 0.55)
        + 0.30 * aspect_score
        + 0.25 * coverage_score
        + 0.20 * span_score
    )
    return {
        "score": round(float(score), 1),
        "circularity": round(float(circularity), 3),
        "aspect": round(float(aspect_ratio), 2),
        "coverage": round(float(coverage), 3),
        "compactness": round(float(span_score), 3),
    }

