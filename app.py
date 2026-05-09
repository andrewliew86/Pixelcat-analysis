from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap
from PIL import Image
from sklearn.cluster import KMeans


def load_image(file) -> Image.Image:
    img = Image.open(file).convert("RGBA")

    alpha = np.array(img)[:, :, 3]
    if alpha.min() == 255:
        data = np.array(img)
        mask = (data[:, :, 0] > 200) & (data[:, :, 1] > 200) & (data[:, :, 2] > 200)
        data[mask, 3] = 0
        img = Image.fromarray(data, "RGBA")

    return img


def analyze_cat_colors(img: Image.Image, num_colors: int = 3):
    data = np.array(img)
    rgb = data[:, :, :3]
    alpha = data[:, :, 3]

    mask = alpha > 0
    pixels = rgb[mask]

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / counts.sum()

    results = [(colors[i], percentages[i] * 100) for i in range(num_colors)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results, pixels, labels


def donut_figure(results):
    color_pcts = [percent for _, percent in results]
    color_values = [tuple(c / 255 for c in color) for color, _ in results]
    color_labels = [f"RGB{tuple(int(c) for c in color)}" for color, _ in results]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        color_pcts,
        labels=color_labels,
        colors=color_values,
        autopct="%.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="w"),
        textprops=dict(color="black"),
    )
    ax.add_artist(plt.Circle((0, 0), 0.70, fc="white"))
    ax.set_title("Pixel Color Clusters", fontsize=13)
    fig.tight_layout()
    return fig


def scatter3d_figure(pixels, labels, results):
    num_clusters = len(results)
    centers, counts = [], []
    for i in range(num_clusters):
        mask = labels == i
        centers.append(pixels[mask].mean(axis=0))
        counts.append(int(np.sum(mask)))
    centers = np.array(centers)
    counts = np.array(counts)

    sorted_idx = np.argsort(counts)[::-1]
    sorted_centers = centers[sorted_idx]

    new_labels = np.zeros_like(labels)
    for new_i, old_i in enumerate(sorted_idx):
        new_labels[labels == old_i] = new_i

    cmap = ListedColormap(sorted_centers / 255.0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        pixels[:, 0], pixels[:, 1], pixels[:, 2],
        c=new_labels, cmap=cmap, alpha=0.5,
    )
    ax.set_title("Pixel Clusters in 3D RGB Space")
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue", rotation=90)
    fig.colorbar(scatter, label="Cluster (sorted by dominance)")
    return fig


st.set_page_config(page_title="Cat Color Quantifier", page_icon="🐱", layout="wide")
st.title("🐱🎨 Cat Color Quantifier")
st.write(
    "Upload a cat image (ideally with the background removed) and discover its "
    "dominant colors using K-Means clustering."
)

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "webp"]
    )
    num_colors = st.slider("Number of color clusters", min_value=2, max_value=10, value=3)
    run = st.button("Analyze", type="primary", disabled=uploaded is None)

if uploaded is None:
    st.info("Upload an image from the sidebar to get started.")
elif run:
    with st.spinner("Crunching pixels..."):
        img = load_image(uploaded)
        results, pixels, labels = analyze_cat_colors(img, num_colors=num_colors)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input image")
        st.image(img, use_container_width=True)
        st.caption(f"{img.width} × {img.height} pixels ({img.width * img.height:,} total)")

    with col2:
        st.subheader("Dominant colors")
        st.pyplot(donut_figure(results))

    st.subheader("Cluster breakdown")
    table_rows = []
    for color, percent in results:
        r, g, b = (int(c) for c in color)
        table_rows.append({"R": r, "G": g, "B": b, "Hex": f"#{r:02x}{g:02x}{b:02x}", "Percent": f"{percent:.2f}%"})
    st.table(table_rows)

    st.subheader("Pixels in 3D RGB space")
    st.pyplot(scatter3d_figure(pixels, labels, results))
