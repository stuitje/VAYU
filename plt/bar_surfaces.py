import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# --- Config ---
planet = 'gj367b'
file_path = f'out/{planet}/bayes_model_comparison_surface.csv'

# Replace technical surface names with human-readable labels
surface_labels = {
    "albite_dust": "Albite (dust)", "andesite": "Andesite", "basalt_glass": "Basalt glass",
    "basalt_large": "Basalt (large grain)", "basalt_small": "Basalt (small grain)",
    "basalt_tuff": "Basalt tuff", "diorite": "Diorite", "gabbro": "Gabbro",
    "granite": "Granite", "harzburgite": "Harzburgite", "hematite": "Hematite",
    "lherzolite": "Lherzolite", "lunar_anorthosite": "Lunar anorthosite",
    "lunar_marebasalt": "Lunar mare basalt", "magnesium_sulphate": "Magnesium sulfate",
    "mars_basalticshergottite": "Martian basaltic shergottite", "mars_breccia": "Martian breccia",
    "norite": "Norite", "phonolite": "Phonolite", "pyrite": "Pyrite", "rhyolite": "Rhyolite",
   "tephrite": "Tephrite", "tholeiitic_basalt": "Tholeiitic basalt",
   "trachy_basalt": "Trachybasalt", "trachyte": "Trachyte"
}

# --- Load data ---
df = pd.read_csv(file_path)
df = df.sort_values(by='bayes_factor', ascending=False)
df = df[df['surface'].isin(surface_labels.keys())]

# Then apply the mapping
df['surface'] = df['surface'].map(surface_labels)


# Apply the mapping
df['surface'] = df['surface'].map(surface_labels).fillna(df['surface'])

# Color mapping
def get_color(delta_lnZ):
    return "skyblue" if delta_lnZ >= -3 else "coral"

colors = [get_color(delta) for delta in df['ΔlnZ']]

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 6))

bars = ax.bar(
    df['surface'],
    df['bayes_factor'],
    color=colors,
    edgecolor='black',
    alpha=0.85,
    linewidth=0.8
)

# Axis formatting
ax.yaxis.grid(True, linestyle='--', which='both', color='gray', alpha=0.3)
ax.set_axisbelow(True)

# Axis labels and title
ax.set_ylabel('Bayes factor', fontsize=16)
ax.set_title(f'{planet.upper()}: Bayesian model comparison (surfaces)', fontsize=18, pad=12)

# Neutral line
ax.axhline(.05, color='black', linestyle='--', linewidth=1.5, label=r'$\Delta \ln Z =-3$ ' )

# Tick labels
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['surface'], rotation=45, ha='right', fontsize=12)
ax.tick_params(axis='y', labelsize=14)

# Bold selected surface names
#bold_surfaces = {"Norite", "Gabbro", "Lherzolite", "Harzburgite"}
#for label in ax.get_xticklabels():
    #if label.get_text() in bold_surfaces:
       # label.set_fontweight('bold')

ax.margins(x=0.02)




# Legend entries
blue_patch = mpatches.Patch(
    facecolor="skyblue", edgecolor="black", linewidth=0.5,
    label=r'$\Delta \ln Z \geq $-3 (consistent)', alpha=0.85
)
red_patch = mpatches.Patch(
    facecolor="coral", edgecolor="black", linewidth=0.5,
    label=r'$\Delta \ln Z < $-3 (rejected)', alpha=0.85
)
line_entry = Line2D(
    [0], [0],
    color='black', linestyle='--', linewidth=1.5,
    label=r'$\Delta \ln Z = $-3 '
)

# Combine and add legend
ax.legend(handles=[blue_patch, red_patch, line_entry], loc='upper right', fontsize=14, frameon=False)


# Save and show
output_path = f'out/{planet}/{planet}_bar_surfaces.pdf'
plt.tight_layout()
plt.savefig(output_path, dpi=300, format="pdf", bbox_inches='tight', )
print(f"[INFO] Plot saved to {output_path}")
plt.show()
