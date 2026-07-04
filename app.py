import streamlit as st

from src.pixelcat.config import CLUSTER_METHODS, MAX_CLUSTER_PIXELS
from src.pixelcat.ui.pages import (
    render_cat_comparison,
    render_jacket_matcher,
    render_loaf_scorer,
    render_single_analysis,
)
from src.pixelcat.ui.theme import apply_pastel_theme


def render_sidebar():
    with st.sidebar:
        st.header("Settings")
        num_colors = st.slider(
            "Number of color clusters", min_value=2, max_value=10, value=3
        )
        method = st.selectbox("Clustering method", CLUSTER_METHODS)
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

    return num_colors, method, max_cluster_pixels


def main():
    st.set_page_config(page_title="Cat Color Quantifier", page_icon="cat", layout="wide")
    apply_pastel_theme()
    st.title("Cat Color Quantifier")
    st.write(
        "Analyze cat colors, compare palettes, match fur against fabric, and score loafiness."
    )

    num_colors, method, max_cluster_pixels = render_sidebar()

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
