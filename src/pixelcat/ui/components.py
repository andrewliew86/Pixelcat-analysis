from html import escape

import streamlit as st


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

