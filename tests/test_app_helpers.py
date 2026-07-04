import numpy as np

from app import cluster_pixels, sample_indices


def test_cluster_pixels_returns_sorted_percentages():
    dark = np.tile(np.array([[20, 20, 20]], dtype=np.uint8), (80, 1))
    light = np.tile(np.array([[220, 220, 220]], dtype=np.uint8), (20, 1))
    results, scatter_pixels, scatter_labels = cluster_pixels(
        np.vstack([dark, light]),
        num_colors=2,
    )

    assert len(results) == 2
    assert results[0]["percent"] > results[1]["percent"]
    assert round(sum(item["percent"] for item in results), 3) == 100
    assert len(scatter_pixels) == len(scatter_labels)


def test_cluster_pixels_handles_empty_input():
    results, scatter_pixels, scatter_labels = cluster_pixels(
        np.empty((0, 3), dtype=np.uint8),
        num_colors=3,
    )

    assert results == []
    assert scatter_pixels.shape == (0, 3)
    assert len(scatter_labels) == 0


def test_sample_indices_caps_rows_reproducibly():
    first = sample_indices(100, 10)
    second = sample_indices(100, 10)

    assert len(first) == 10
    assert np.array_equal(first, second)
