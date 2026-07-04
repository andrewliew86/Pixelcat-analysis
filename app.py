from io import BytesIO
from html import escape

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


@st.cache_data(show_spinner=False)
def analyze_image(file_bytes, num_colors, method, max_cluster_pixels):
    img = load_image_from_bytes(file_bytes)
    pixels, mask = foreground_pixels(img)
    results, scatter_pixels, scatter_labels = cluster_pixels(
        pixels,
        num_colors,
        method,
        max_cluster_pixels,
    )
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


def palette_table_html(results):
    rows = []
    for item in results:
        color = escape(item["hex"])
        rows.append(
            "<tr>"
            f'<td><span class="color-chip" style="background:{color};"></span>{color}</td>'
            f"<td>{item['rgb'][0]}</td>"
            f"<td>{item['rgb'][1]}</td>"
            f"<td>{item['rgb'][2]}</td>"
            f"<td>{item['percent']:.2f}%</td>"
            "</tr>"
        )

    return (
        '<div class="palette-table-wrap">'
        '<table class="palette-table">'
        "<thead><tr>"
        "<th>Color</th>"
        "<th>R</th>"
        "<th>G</th>"
        "<th>B</th>"
        "<th>Percent</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def render_palette_table(results):
    if not results:
        st.info("No cluster results to show yet.")
        return

    st.markdown(palette_table_html(results), unsafe_allow_html=True)


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


def apply_pastel_theme():
    st.markdown(
        """
        <style>
        :root {
            --cat-ink: #171626;
            --cat-muted: #3c3854;
            --cat-cream: #fff4d6;
            --cat-paper: #fffaf0;
            --cat-gold: #ffd45a;
            --cat-red: #f1433b;
            --cat-red-deep: #c72935;
            --cat-green: #198d69;
            --cat-mint: #dfe8d6;
            --cat-border: #171626;
        }

        .stApp {
            color: var(--cat-ink);
            background:
                linear-gradient(115deg, #fff0bd 0%, #fff0d8 44%, #dfe8d6 100%);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffe7a8 0%, #f7ead4 58%, #dfe8d6 100%);
            border-right: 3px solid var(--cat-border);
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: var(--cat-ink);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--cat-red-deep);
            text-shadow: none;
        }

        .sidebar-note {
            background: rgba(255, 250, 240, 0.76);
            border: 2px solid var(--cat-border);
            border-left: 8px solid var(--cat-green);
            border-radius: 8px;
            color: var(--cat-ink);
            font-weight: 650;
            line-height: 1.45;
            margin-top: 1rem;
            padding: 0.75rem 0.85rem;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: var(--cat-paper);
            border: 3px solid var(--cat-border);
            border-radius: 8px;
            color: var(--cat-ink);
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] svg {
            color: var(--cat-ink);
            fill: var(--cat-ink);
        }

        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 2.4rem;
            max-width: 1180px;
        }

        h1, h2, h3 {
            color: var(--cat-ink);
            letter-spacing: 0;
        }

        h1 {
            color: var(--cat-red);
            font-size: 4.5rem;
            font-weight: 850;
            line-height: 0.95;
            margin-bottom: 0.4rem;
            text-shadow: 0.07em 0.07em 0 var(--cat-gold);
        }

        p, li, label, div {
            letter-spacing: 0;
        }

        [data-testid="stMarkdownContainer"] p {
            color: var(--cat-muted);
        }

        .cat-feature-note {
            background: var(--cat-paper);
            border: 3px solid var(--cat-border);
            border-left: 12px solid var(--cat-green);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin: 1.35rem 0 1.35rem;
            box-shadow: 7px 7px 0 rgba(23, 22, 38, 0.13);
        }

        .cat-feature-note p {
            margin: 0.35rem 0;
            color: var(--cat-ink);
            line-height: 1.48;
        }

        .cat-feature-note strong {
            color: var(--cat-red-deep);
        }

        [data-testid="stTabs"] div[data-baseweb="tab-list"] {
            gap: 0.85rem;
            border-bottom: 5px solid var(--cat-red);
            padding: 0.25rem 0 1.2rem;
            margin: 0.9rem 0 1.2rem;
        }

        [data-testid="stTabs"] div[data-baseweb="tab-list"] button {
            flex: 0 0 auto;
        }

        [data-testid="stTabs"] div[data-baseweb="tab-highlight"],
        [data-testid="stTabs"] div[data-baseweb="tab-border"] {
            display: none;
        }

        [data-testid="stTabs"] button {
            background: var(--cat-paper);
            border: 3px solid var(--cat-border);
            border-radius: 8px;
            box-shadow: 0 7px 0 var(--cat-ink);
            color: var(--cat-ink);
            font-weight: 800;
            min-height: 46px;
            padding: 0.55rem 1.15rem;
            transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
        }

        [data-testid="stTabs"] button:hover {
            background: #fff6d7;
            transform: translateY(2px);
            box-shadow: 0 5px 0 var(--cat-ink);
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            background: var(--cat-red);
            color: #ffffff;
            box-shadow: 0 7px 0 var(--cat-ink);
        }

        [data-testid="stTabs"] button p {
            color: inherit;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0;
            padding: 0;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 250, 240, 0.86);
            border: 3px dashed #d7a94b;
            border-radius: 8px;
        }

        [data-testid="stFileUploaderDropzone"] button,
        .stButton button {
            background: var(--cat-gold);
            border: 3px solid var(--cat-border);
            border-radius: 8px;
            box-shadow: 0 5px 0 var(--cat-ink);
            color: var(--cat-ink);
            font-weight: 800;
        }

        [data-testid="stAlert"] {
            background: rgba(255, 250, 240, 0.88);
            border: 3px solid var(--cat-border);
            border-left: 10px solid var(--cat-green);
            border-radius: 8px;
            color: var(--cat-ink);
        }

        [data-testid="stMetric"] {
            background: var(--cat-paper);
            border: 3px solid var(--cat-border);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 6px 6px 0 rgba(23, 22, 38, 0.12);
        }

        [data-testid="stTable"] {
            border: 3px solid var(--cat-border);
            border-radius: 8px;
            overflow: hidden;
        }

        .palette-table-wrap {
            background: rgba(255, 250, 240, 0.78);
            border: 3px solid var(--cat-border);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 6px 6px 0 rgba(23, 22, 38, 0.10);
        }

        .palette-table {
            border-collapse: collapse;
            color: var(--cat-ink);
            font-size: 1rem;
            width: 100%;
        }

        .palette-table th {
            background: var(--cat-gold);
            border-bottom: 3px solid var(--cat-border);
            font-weight: 850;
            padding: 0.75rem 0.9rem;
            text-align: left;
        }

        .palette-table td {
            border-bottom: 1px solid rgba(23, 22, 38, 0.16);
            padding: 0.7rem 0.9rem;
            vertical-align: middle;
        }

        .palette-table tr:last-child td {
            border-bottom: none;
        }

        .palette-table th:not(:first-child),
        .palette-table td:not(:first-child) {
            text-align: right;
        }

        .color-chip {
            border: 2px solid var(--cat-border);
            border-radius: 6px;
            display: inline-block;
            height: 1rem;
            margin-right: 0.55rem;
            vertical-align: -0.12rem;
            width: 1rem;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="slider"] {
            color: var(--cat-ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def feature_note(description, calculation, instructions, interpretation=None):
    lines = [
        f"<p><strong>What it does:</strong> {description}</p>",
        f"<p><strong>How it works:</strong> {calculation}</p>",
        f"<p><strong>How to use it:</strong> {instructions}</p>",
    ]
    if interpretation:
        lines.append(f"<p><strong>How to read it:</strong> {interpretation}</p>")

    st.markdown(
        f"<div class=\"cat-feature-note\">{''.join(lines)}</div>",
        unsafe_allow_html=True,
    )


def render_single_analysis(num_colors, method, max_cluster_pixels):
    feature_note(
        "Finds the main colors in your cat's fur.",
        (
            "It looks at the visible cat pixels and groups similar colors together. "
            "For speed, it uses a sample of the image rather than every single pixel."
        ),
        (
            "Upload a cat image that has already been cropped or segmented so the "
            "background is transparent. A clean white background can also work, but "
            "busy backgrounds will confuse the results."
        ),
        (
            "Bigger percentages mean that color appears more often in the cat's fur. "
            "The color chart is a quick summary of the fur palette."
        ),
    )

    uploaded = st.file_uploader("Upload a cat image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("Upload an image to get started.")
        return

    with st.spinner("Analyzing sampled foreground pixels..."):
        analysis = analyze_image(
            uploaded.getvalue(),
            num_colors,
            method,
            max_cluster_pixels,
        )

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
    render_palette_table(analysis["results"])

    st.subheader("Sampled RGB plot")
    fig = scatter3d_figure(
        analysis["scatter_pixels"],
        analysis["scatter_labels"],
        analysis["results"],
    )
    if fig:
        st.pyplot(fig)


def render_cat_comparison(num_colors, method, max_cluster_pixels):
    feature_note(
        "Compares one cat's fur palette with other cats.",
        (
            "It compares the main fur colors from each image and checks how close those "
            "palettes are to one another."
        ),
        (
            "Upload one reference cat, then upload one or more cats to compare. Use "
            "segmented or transparent-background cat images for the fairest comparison."
        ),
        (
            "Higher similarity means the cats have more similar fur colors. Lower color "
            "distance means the palettes are closer."
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

    ref_analysis = analyze_image(
        reference.getvalue(),
        num_colors,
        method,
        max_cluster_pixels,
    )
    st.subheader("Reference palette")
    show_palette_swatches(ref_analysis["results"])
    if not ref_analysis["results"]:
        return

    rows = []
    for cat_file in comparisons:
        cat_analysis = analyze_image(
            cat_file.getvalue(),
            num_colors,
            method,
            max_cluster_pixels,
        )
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


def render_jacket_matcher(num_colors, method, max_cluster_pixels):
    feature_note(
        "Estimates how much your cat's fur might show up on a jacket.",
        (
            "It compares the cat's fur colors with the jacket colors. Bigger color "
            "differences mean shed fur is more likely to stand out."
        ),
        (
            "Upload a segmented cat image and a jacket photo. For the jacket, try to use "
            "a photo where the jacket fills most of the image."
        ),
        (
            "A higher risk score means fur will probably be more noticeable. A lower "
            "score means the jacket is closer to your cat's fur colors."
        ),
    )

    cat = st.file_uploader("Cat photo", type=["png", "jpg", "jpeg", "webp"], key="jacket-cat")
    jacket = st.file_uploader(
        "Jacket photo", type=["png", "jpg", "jpeg", "webp"], key="jacket-photo"
    )

    if cat is None or jacket is None:
        st.info("Upload a cat image and a jacket image.")
        return

    cat_analysis = analyze_image(
        cat.getvalue(),
        num_colors,
        method,
        max_cluster_pixels,
    )
    jacket_analysis = analyze_image(
        jacket.getvalue(),
        num_colors,
        method,
        max_cluster_pixels,
    )
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
        "Scores how loaf-like your cat's shape is.",
        (
            "It looks at the cat's outline and rewards a compact, rounded, filled-in "
            "shape. A classic loaf should look wide, smooth, and tucked-in."
        ),
        (
            "Upload a loafing cat image that has been cropped or segmented with a "
            "transparent background. A clean white background can work, but cluttered "
            "backgrounds will make the score less reliable."
        ),
        (
            "A higher score means a stronger loaf. Visible paws, tails, stretched poses, "
            "or background clutter can lower the score."
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
    apply_pastel_theme()
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
        max_cluster_pixels = st.slider(
            "Foreground pixel sample limit",
            min_value=5_000,
            max_value=100_000,
            value=MAX_CLUSTER_PIXELS,
            step=5_000,
        )
        st.markdown(
            f"""
            <div class="sidebar-note">
                Uses up to {max_cluster_pixels:,} visible cat pixels for clustering.
                Higher values may be a little more precise but slower.
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab_analyze, tab_compare, tab_jacket, tab_loaf = st.tabs(
        ["Analyze", "Compare cats", "Jacket matcher", "Loaf scorer"]
    )

    with tab_analyze:
        render_single_analysis(num_colors, method, max_cluster_pixels)

    with tab_compare:
        render_cat_comparison(num_colors, method, max_cluster_pixels)

    with tab_jacket:
        render_jacket_matcher(num_colors, method, max_cluster_pixels)

    with tab_loaf:
        render_loaf_scorer()


if __name__ == "__main__":
    main()
