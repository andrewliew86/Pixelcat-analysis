from collections import deque
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
        # Estimate the studio background from border pixels and remove only the
        # connected border region. This handles white/grey backdrops without
        # deleting light fur or fabric inside the subject.
        rgb = data[:, :, :3].astype(float)
        border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
        background_color = np.median(border, axis=0)
        color_distance = np.linalg.norm(rgb - background_color, axis=2)
        candidate = color_distance < 32
        background = np.zeros(candidate.shape, dtype=bool)
        queue = deque()
        edge = np.zeros(candidate.shape, dtype=bool)
        edge[[0, -1], :] = True
        edge[:, [0, -1]] = True
        for row, col in np.argwhere(candidate & edge):
            background[row, col] = True
            queue.append((row, col))
        while queue:
            row, col = queue.popleft()
            for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if (0 <= rr < candidate.shape[0] and 0 <= cc < candidate.shape[1]
                        and candidate[rr, cc] and not background[rr, cc]):
                    background[rr, cc] = True
                    queue.append((rr, cc))
        data = data.copy()
        data[background, 3] = 0
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

