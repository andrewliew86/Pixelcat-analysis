from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from .clustering import cluster_pixels


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

