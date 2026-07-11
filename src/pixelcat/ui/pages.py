import streamlit as st

from ..image_processing import analyze_image, foreground_pixels, uploaded_image
from ..plots import donut_figure, scatter3d_figure
from ..scores import (
    color_mismatch_score,
    distance_to_similarity,
    loaf_score,
    palette_distance,
)
from .components import feature_note, render_palette_table, show_palette_swatches


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
                {"Metric": "Body compactness", "Value": metrics["compactness"]},
            ]
        )

