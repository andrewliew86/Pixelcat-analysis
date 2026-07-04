import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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

