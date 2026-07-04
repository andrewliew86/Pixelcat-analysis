import streamlit as st


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

