import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import rcParams

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.surface_labels import surface_labels

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

# Config
planet     = "gj367b"
file_path  = f"out/{planet}/bayes_model_comparison.csv"
output_path = f"out/{planet}/{planet}_bar_surfaces.pdf"

#  Load and filter 
df = pd.read_csv(file_path)
df["ΔlnZ"] = pd.to_numeric(df["ΔlnZ"], errors="coerce")
df["bayes_factor"] = pd.to_numeric(df["bayes_factor"], errors="coerce")

# Keep only surface runs (atmosphere == bare_rock) and surfaces we have labels for
df = df[(df["atmosphere"] == "bare_rock") & (df["surface"].isin(surface_labels))]
df = df.sort_values("bayes_factor", ascending=False)

# Map to human-readable labels
df["label"] = df["surface"].map(surface_labels)

# Colours
def get_color(delta_lnZ):
    return "skyblue" if delta_lnZ >= -3 else "coral"

colors = [get_color(d) for d in df["ΔlnZ"]]

# Plot
fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(
    df["label"],
    df["bayes_factor"],
    color=colors,
    edgecolor="black",
    alpha=0.85,
    linewidth=0.8,
)

ax.yaxis.grid(True, linestyle="--", which="both", color="gray", alpha=0.3)
ax.set_axisbelow(True)
ax.axhline(0.05, color="black", linestyle="--", linewidth=1.5)

ax.set_ylabel("Bayes factor", fontsize=16)
ax.set_title(f"{planet.upper()}: Bayesian model comparison (surfaces)", fontsize=18, pad=12)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=12)
ax.tick_params(axis="y", labelsize=14)
ax.margins(x=0.02)

# Legend
blue_patch = mpatches.Patch(facecolor="skyblue", edgecolor="black", linewidth=0.5,
                             label=r"$\Delta \ln Z \geq -3$ (consistent)", alpha=0.85)
red_patch  = mpatches.Patch(facecolor="coral",   edgecolor="black", linewidth=0.5,
                             label=r"$\Delta \ln Z < -3$ (rejected)",    alpha=0.85)
line_entry = Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                    label=r"$\Delta \ln Z = -3$")

ax.legend(handles=[blue_patch, red_patch, line_entry], loc="upper right",
          fontsize=14, frameon=False)

plt.tight_layout()
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, format="pdf", bbox_inches="tight")
print(f"[INFO] Plot saved to {output_path}")
plt.show()