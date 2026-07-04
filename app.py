from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


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


def cluster_pixels(pixels, num_colors=3):
    if len(pixels) == 0:
        return [], np.empty((0, 3), dtype=np.uint8), np.array([], dtype=int)

    fit_pixels = sample_rows(pixels, MAX_CLUSTER_PIXELS)
    unique_count = len(np.unique(fit_pixels, axis=0))
    n_clusters = max(1, min(num_colors, unique_count, len(fit_pixels)))

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
def analyze_image(file_bytes, num_colors):
    img = load_image_from_bytes(file_bytes)
    pixels, mask = foreground_pixels(img)
    results, scatter_pixels, scatter_labels = cluster_pixels(pixels, num_colors)
    return {
        "img": img,
        "results": results,
        "pixels": pixels,
        "scatter_pixels": scatter_pixels,
        "scatter_labels": scatter_labels,
        "mask": mask,
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


def render_single_analysis(num_colors):
    uploaded = st.file_uploader("Upload a cat image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("Upload an image to get started.")
        return

    with st.spinner("Analyzing sampled foreground pixels..."):
        analysis = analyze_image(uploaded.getvalue(), num_colors)

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


def main():
    st.set_page_config(page_title="Cat Color Quantifier", page_icon="cat", layout="wide")
    st.title("Cat Color Quantifier")
    st.write(
        "Upload a cat image, sample foreground pixels, and discover the dominant fur colors."
    )

    with st.sidebar:
        st.header("Settings")
        num_colors = st.slider(
            "Number of color clusters", min_value=2, max_value=10, value=3
        )
        st.caption(
            f"Color models use up to {MAX_CLUSTER_PIXELS:,} foreground pixels for speed."
        )

    render_single_analysis(num_colors)


if __name__ == "__main__":
    main()
