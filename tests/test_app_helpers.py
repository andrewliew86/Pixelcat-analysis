import numpy as np

from src.pixelcat.clustering import cluster_pixels, sample_indices
from src.pixelcat.scores import (
    color_mismatch_score,
    distance_to_similarity,
    loaf_score,
    palette_distance,
)
from src.pixelcat.ui.components import palette_table_html


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


def test_cluster_pixels_supports_gaussian_mixture():
    dark = np.tile(np.array([[20, 20, 20]], dtype=np.uint8), (30, 1))
    light = np.tile(np.array([[220, 220, 220]], dtype=np.uint8), (30, 1))
    results, scatter_pixels, scatter_labels = cluster_pixels(
        np.vstack([dark, light]),
        num_colors=2,
        method="Gaussian Mixture",
    )

    assert len(results) == 2
    assert round(sum(item["percent"] for item in results), 3) == 100
    assert len(scatter_pixels) == len(scatter_labels)


def test_palette_distance_is_zero_for_same_palette():
    palette = [{"rgb": (100, 120, 140), "hex": "#64788c", "percent": 100.0}]

    assert palette_distance(palette, palette) == 0
    assert color_mismatch_score(palette, palette) == 0
    assert distance_to_similarity(0) == 100


def test_palette_distance_is_symmetric():
    palette_a = [{"rgb": (0, 0, 0), "hex": "#000000", "percent": 100.0}]
    palette_b = [
        {"rgb": (255, 255, 255), "hex": "#ffffff", "percent": 50.0},
        {"rgb": (0, 0, 0), "hex": "#000000", "percent": 50.0},
    ]

    assert palette_distance(palette_a, palette_b) == palette_distance(
        palette_b, palette_a
    )


def test_loaf_score_rewards_compact_oval_mask():
    rows, cols = np.ogrid[:80, :120]
    oval = ((rows - 40) / 25) ** 2 + ((cols - 60) / 42) ** 2 <= 1

    metrics = loaf_score(oval)

    assert 0 <= metrics["score"] <= 10
    assert metrics["score"] > 5
    assert metrics["aspect"] > 1


def test_sample_indices_caps_rows_reproducibly():
    first = sample_indices(100, 10)
    second = sample_indices(100, 10)

    assert len(first) == 10
    assert np.array_equal(first, second)


def test_cluster_pixels_respects_sample_limit():
    dark = np.tile(np.array([[20, 20, 20]], dtype=np.uint8), (80, 1))
    light = np.tile(np.array([[220, 220, 220]], dtype=np.uint8), (80, 1))

    _, scatter_pixels, scatter_labels = cluster_pixels(
        np.vstack([dark, light]),
        num_colors=2,
        max_cluster_pixels=25,
    )

    assert len(scatter_pixels) == 25
    assert len(scatter_labels) == 25


def test_palette_table_html_does_not_render_rows_as_code():
    html = palette_table_html(
        [{"rgb": (67, 51, 34), "hex": "#433322", "percent": 32.67}]
    )

    assert html.startswith('<div class="palette-table-wrap">')
    assert "<tbody><tr>" in html
    assert "\n            <tr>" not in html
