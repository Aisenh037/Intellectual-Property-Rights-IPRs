"""
Generate a simple workflow diagram PNG for the repo README.

Output: ipc_workflow.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    # Canvas
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")

    # Helper to draw rounded boxes + centered labels
    def box(x, y, w, h, label, facecolor="#E8F1FF"):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.2,rounding_size=0.08",
                linewidth=1.5, edgecolor="#222222", facecolor=facecolor
            )
        )
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center", fontsize=10, weight="bold")

    # Nodes
    box(0.02, 0.40, 0.22, 0.22, "Raw IPC Dataset", facecolor="#DDEBFF")
    box(0.30, 0.40, 0.26, 0.22, "Cleaning &\nNormalization", facecolor="#E7F7E7")
    box(0.60, 0.40, 0.28, 0.22, "Classification:\nWIPO GREEN / non-GREEN", facecolor="#FFF6D9")
    box(0.90, 0.40, 0.22, 0.22, "Export Clean\nDataset", facecolor="#FFE2E0")

    # Arrows
    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", linewidth=2, color="#222222")
        )

    arrow(0.24, 0.51, 0.30, 0.51)  # Raw -> Cleaning
    arrow(0.56, 0.51, 0.60, 0.51)  # Cleaning -> Classification
    arrow(0.88, 0.51, 0.90, 0.51)  # Classification -> Export

    # Save
    plt.savefig("ipc_workflow.png", dpi=180, bbox_inches="tight")
    print("Saved ipc_workflow.png")

if __name__ == "__main__":
    main()
