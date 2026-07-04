from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap
from PIL import Image, ImageFilter
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture


MAX_CLUSTER_PIXELS = 30_000
MAX_SCATTER_PIXELS = 6_000


def rgb_to_hex(rgb):
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


@st.cache_data(show_spinner=False)
def load_image_from_bytes(file_bytes) -> Image.Image:
    img = Image.open(BytesIO(file_bytes)).convert("RGBA")
    data = np.array(img)

    alpha = data[:, :, 3]
    if alpha.min() == 255:
        white_bg = (
            (data[:, :, 0] > 200)
            & (data[:, :, 1] > 200)
            & (data[:, :, 2] > 200)
        )
        data = data.copy()
        data[white_bg, 3] = 0
        img = Image.fromarray(data, "RGBA")

    return img


def uploaded_image(uploaded_file) -> Image.Image:
    return load_image_from_bytes(uploaded_file.getvalue())


def foreground_pixels(img: Image.Image):
    data = np.asarray(img)
    alpha = data[:, :, 3]
    mask = alpha > 0
    return data[:, :, :3][mask], mask


def sample_indices(row_count, max_rows, seed=42):
    if row_count <= max_rows:
        return np.arange(row_count)

    rng = np.random.default_rng(seed)
    return rng.choice(row_count, size=max_rows, replace=False)


def sample_rows(values, max_rows, seed=42):
    return values[sample_indices(len(values), max_rows, seed)]


def cluster_pixels(pixels, num_colors=3, method="Fast K-Means"):
    if len(pixels) == 0:
        return [], np.empty((0, 3), dtype=np.uint8), np.array([], dtype=int)

    fit_pixels = sample_rows(pixels, MAX_CLUSTER_PIXELS)
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


@st.cache_data(show_spinner=False)
def analyze_image(file_bytes, num_colors, method):
    img = load_image_from_bytes(file_bytes)
    pixels, mask = foreground_pixels(img)
    results, scatter_pixels, scatter_labels = cluster_pixels(pixels, num_colors, method)
    return {
        "img": img,
        "results": results,
        "pixels": pixels,
        "scatter_pixels": scatter_pixels,
        "scatter_labels": scatter_labels,
        "mask": mask,
    }


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


def donut_figure(results, title="Pixel Color Clusters"):
    color_pcts = [item["percent"] for item in results]
    color_values = [tuple(c / 255 for c in item["rgb"]) for item in results]
    color_labels = [f"{item['hex']} ({item['percent']:.1f}%)" for item in results]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        color_pcts,
        labels=color_labels,
        colors=color_values,
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="w"),
        textprops=dict(color="black"),
    )
    ax.add_artist(plt.Circle((0, 0), 0.70, fc="white"))
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return fig


def scatter3d_figure(pixels, labels, results):
    if len(pixels) == 0 or not results:
        return None

    cmap_colors = [tuple(c / 255 for c in item["rgb"]) for item in results]
    cmap = ListedColormap(cmap_colors)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        pixels[:, 0],
        pixels[:, 1],
        pixels[:, 2],
        c=labels,
        cmap=cmap,
        alpha=0.45,
        s=8,
    )
    ax.set_title("Sampled Pixels in RGB Space")
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue", rotation=90)
    fig.colorbar(scatter, label="Cluster")
    fig.tight_layout()
    return fig


def palette_table(results):
    return [
        {
            "Color": item["hex"],
            "R": item["rgb"][0],
            "G": item["rgb"][1],
            "B": item["rgb"][2],
            "Percent": f"{item['percent']:.2f}%",
        }
        for item in results
    ]


def show_palette_swatches(results):
    if not results:
        st.warning("No foreground pixels found. Try an image with visible cat pixels.")
        return

    columns = st.columns(max(1, len(results)))
    for column, item in zip(columns, results):
        with column:
            st.markdown(
                f"""
                <div style="
                    height: 44px;
                    border-radius: 6px;
                    border: 1px solid #ddd;
                    background: {item['hex']};
                "></div>
                <div style="font-size: 0.85rem; margin-top: 0.25rem;">
                    {item['hex']}<br>{item['percent']:.1f}%
                </div>
                """,
                unsafe_allow_html=True,
            )


def feature_note(description, calculation, instructions, interpretation=None):
    lines = [
        f"**What it does:** {description}",
        f"**Calculation:** {calculation}",
        f"**How to use it:** {instructions}",
    ]
    if interpretation:
        lines.append(f"**How to read it:** {interpretation}")

    st.info("\n\n".join(lines))


def render_single_analysis(num_colors, method):
    feature_note(
        "Finds the main fur colors in one uploaded cat image.",
        (
            f"The app keeps foreground pixels, samples up to {MAX_CLUSTER_PIXELS:,} of "
            "them for speed, clusters RGB values with the selected method, then reports "
            "each cluster as a percentage of sampled foreground color."
        ),
        "Upload a cat image, ideally with a transparent or simple background.",
        (
            "Larger percentages are more dominant colors. The RGB plot shows a sampled "
            "view of how separated or blended the color clusters are."
        ),
    )

    uploaded = st.file_uploader("Upload a cat image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("Upload an image to get started.")
        return

    with st.spinner("Analyzing sampled foreground pixels..."):
        analysis = analyze_image(uploaded.getvalue(), num_colors, method)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input image")
        st.image(analysis["img"], use_container_width=True)
        st.caption(
            f"{analysis['img'].width} x {analysis['img'].height} pixels; "
            f"{len(analysis['pixels']):,} foreground pixels"
        )
    with col2:
        st.subheader("Dominant colors")
        show_palette_swatches(analysis["results"])
        if analysis["results"]:
            st.pyplot(donut_figure(analysis["results"]))

    st.subheader("Cluster breakdown")
    st.table(palette_table(analysis["results"]))

    st.subheader("Sampled RGB plot")
    fig = scatter3d_figure(
        analysis["scatter_pixels"],
        analysis["scatter_labels"],
        analysis["results"],
    )
    if fig:
        st.pyplot(fig)


def render_cat_comparison(num_colors, method):
    feature_note(
        "Compares the dominant-color palette of one reference cat against other cats.",
        (
            "For each reference color, the app finds the nearest comparison color in RGB "
            "space, weights that distance by the reference color percentage, and averages "
            "the result in both directions."
        ),
        "Upload one reference cat, then upload one or more cats to compare against it.",
        (
            "Lower color distance means the palettes are closer. Higher similarity means "
            "the comparison cat is more color-similar to the reference cat."
        ),
    )

    reference = st.file_uploader(
        "Reference cat", type=["png", "jpg", "jpeg", "webp"], key="reference-cat"
    )
    comparisons = st.file_uploader(
        "Cats to compare",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="comparison-cats",
    )

    if reference is None or not comparisons:
        st.info("Upload one reference cat and at least one comparison cat.")
        return

    ref_analysis = analyze_image(reference.getvalue(), num_colors, method)
    st.subheader("Reference palette")
    show_palette_swatches(ref_analysis["results"])
    if not ref_analysis["results"]:
        return

    rows = []
    for cat_file in comparisons:
        cat_analysis = analyze_image(cat_file.getvalue(), num_colors, method)
        distance = palette_distance(ref_analysis["results"], cat_analysis["results"])
        similarity = distance_to_similarity(distance)
        rows.append(
            {
                "Cat": cat_file.name,
                "Color distance": f"{distance:.1f}" if distance is not None else "N/A",
                "Similarity": f"{similarity:.1f}%" if similarity is not None else "N/A",
            }
        )

    st.subheader("Comparison")
    st.table(rows)


def render_jacket_matcher(num_colors, method):
    feature_note(
        "Estimates how visible cat fur may be on a jacket.",
        (
            "The app compares the cat palette and jacket palette using the same weighted "
            "RGB distance as cat comparison, then scales that mismatch to a 0-100 risk "
            "score."
        ),
        "Upload a cat photo and a jacket photo.",
        (
            "Higher risk means stronger color contrast, so shed fur is more likely to "
            "stand out. Lower risk means the jacket color is closer to the cat palette."
        ),
    )

    cat = st.file_uploader("Cat photo", type=["png", "jpg", "jpeg", "webp"], key="jacket-cat")
    jacket = st.file_uploader(
        "Jacket photo", type=["png", "jpg", "jpeg", "webp"], key="jacket-photo"
    )

    if cat is None or jacket is None:
        st.info("Upload a cat image and a jacket image.")
        return

    cat_analysis = analyze_image(cat.getvalue(), num_colors, method)
    jacket_analysis = analyze_image(jacket.getvalue(), num_colors, method)
    mismatch = color_mismatch_score(cat_analysis["results"], jacket_analysis["results"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cat palette")
        st.image(cat_analysis["img"], use_container_width=True)
        show_palette_swatches(cat_analysis["results"])
    with col2:
        st.subheader("Jacket palette")
        st.image(jacket_analysis["img"], use_container_width=True)
        show_palette_swatches(jacket_analysis["results"])

    if mismatch is None:
        st.warning("Could not compare colors because one image has no foreground pixels.")
        return

    st.metric("Fur visibility risk", f"{mismatch:.0f}/100")
    if mismatch >= 65:
        st.warning("High contrast: light fur on dark fabric, or the reverse, may stand out.")
    elif mismatch >= 35:
        st.info("Moderate contrast: some fur is likely to be visible.")
    else:
        st.success("Low contrast: fur should be less obvious on this jacket.")


def render_loaf_scorer():
    feature_note(
        "Scores how loaf-like the visible cat shape is.",
        (
            "The app builds a foreground mask, crops it to the cat shape, estimates edges "
            "for perimeter, then combines circularity, width-to-height aspect ratio, and "
            "mask coverage into a 0-10 score."
        ),
        "Upload a loafing cat image with a transparent, white, or otherwise clean background.",
        (
            "A higher score means the silhouette is compact, oval, and filled-in. Busy "
            "backgrounds or visible paws/tails can lower the score."
        ),
    )

    uploaded = st.file_uploader(
        "Upload a loafing cat image", type=["png", "jpg", "jpeg", "webp"], key="loaf-cat"
    )
    if uploaded is None:
        st.info("Upload a cat image with a clean or transparent background.")
        return

    img = uploaded_image(uploaded)
    _, mask = foreground_pixels(img)
    metrics = loaf_score(mask)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input image")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("Loaf score")
        st.metric("Loafiness", f"{metrics['score']}/10")
        st.table(
            [
                {"Metric": "Circularity", "Value": metrics["circularity"]},
                {"Metric": "Aspect ratio", "Value": metrics["aspect"]},
                {"Metric": "Mask coverage", "Value": metrics["coverage"]},
            ]
        )


def main():
    st.set_page_config(page_title="Cat Color Quantifier", page_icon="cat", layout="wide")
    st.title("Cat Color Quantifier")
    st.write(
        "Analyze cat colors, compare palettes, match fur against fabric, and score loafiness."
    )

    with st.sidebar:
        st.header("Settings")
        num_colors = st.slider(
            "Number of color clusters", min_value=2, max_value=10, value=3
        )
        method = st.selectbox("Clustering method", ["Fast K-Means", "Gaussian Mixture"])
        st.caption(
            f"Color models use up to {MAX_CLUSTER_PIXELS:,} foreground pixels for speed."
        )

    tab_analyze, tab_compare, tab_jacket, tab_loaf = st.tabs(
        ["Analyze", "Compare cats", "Jacket matcher", "Loaf scorer"]
    )

    with tab_analyze:
        render_single_analysis(num_colors, method)

    with tab_compare:
        render_cat_comparison(num_colors, method)

    with tab_jacket:
        render_jacket_matcher(num_colors, method)

    with tab_loaf:
        render_loaf_scorer()


if __name__ == "__main__":
    main()
