"""Deep-ocean figure style shared by the study figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG, PANEL, GRID = "#061726", "#0b2236", "#1d3a55"
TEXT, MUTED = "#e6f1ff", "#8fb3d1"
CYAN, TEAL, AMBER, CORAL, VIOLET, WHITE = "#3fd0ff", "#2ee6a6", "#ffb547", "#ff6b6b", "#a78bfa", "#ffffff"
SERIES = [CYAN, AMBER, TEAL, CORAL, VIOLET]


def apply():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "grid.color": GRID, "grid.linewidth": 0.7, "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0,
    })


def title(fig, text, sub=None):
    """Title and subtitle at fixed distances from the top edge, whatever the figure height."""
    h = fig.get_figheight()
    fig.suptitle(text, x=0.012, y=1 - 0.12 / h, ha="left", va="top", fontsize=13, fontweight="bold", color=TEXT)
    if sub:
        fig.text(0.012, 1 - 0.45 / h, sub, ha="left", va="top", fontsize=9, color=MUTED)


def rect(fig):
    return (0, 0, 1, 1 - 0.62 / fig.get_figheight())
