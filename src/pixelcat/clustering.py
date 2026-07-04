import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from .config import MAX_CLUSTER_PIXELS, MAX_SCATTER_PIXELS


def rgb_to_hex(rgb):
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def sample_indices(row_count, max_rows, seed=42):
    max_rows = max(1, int(max_rows))
    if row_count <= max_rows:
        return np.arange(row_count)

    rng = np.random.default_rng(seed)
    return rng.choice(row_count, size=max_rows, replace=False)


def sample_rows(values, max_rows, seed=42):
    return values[sample_indices(len(values), max_rows, seed)]


def cluster_pixels(
    pixels,
    num_colors=3,
    method="Fast K-Means",
    max_cluster_pixels=MAX_CLUSTER_PIXELS,
):
    if len(pixels) == 0:
        return [], np.empty((0, 3), dtype=np.uint8), np.array([], dtype=int)

    fit_pixels = sample_rows(pixels, max_cluster_pixels)
    unique_count = len(np.unique(fit_pixels, axis=0))
    n_clusters = max(1, min(num_colors, unique_count, len(fit_pixels)))

    if method == "Gaussian Mixture":
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=42,
            reg_covar=1e-3,
            max_iter=150,
        )
        labels = model.fit_predict(fit_pixels)
        centers = model.means_
    else:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=3,
            batch_size=4096,
        )
        labels = model.fit_predict(fit_pixels)
        centers = model.cluster_centers_

    counts = np.bincount(labels, minlength=n_clusters)
    total = counts.sum()
    order = np.argsort(counts)[::-1]

    results = []
    sorted_labels = np.zeros_like(labels)
    for rank, cluster_index in enumerate(order):
        rgb = np.clip(np.rint(centers[cluster_index]), 0, 255).astype(int)
        percent = float(counts[cluster_index] / total * 100) if total else 0.0
        results.append(
            {
                "rgb": tuple(int(c) for c in rgb),
                "hex": rgb_to_hex(rgb),
                "percent": percent,
            }
        )
        sorted_labels[labels == cluster_index] = rank

    scatter_indices = sample_indices(len(fit_pixels), MAX_SCATTER_PIXELS)
    scatter_pixels = fit_pixels[scatter_indices]
    scatter_labels = sorted_labels[scatter_indices]
    return results, scatter_pixels, scatter_labels

