import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

planet_name = "gj367b"  # Change to "gj367b" for the other planet

# Load the Bayes factor data
df = pd.read_csv(f'out/{planet_name}/bayes_model_comparison_atmo.csv')

# Create a mapping from atmosphere name to bayes factor and delta ln Z
bayes_map = {}
delta_lnz_map = {}
for _, row in df.iterrows():
    atm_name = row['atmosphere']
    bayes_factor = row['bayes_factor']
    delta_lnz = row['ΔlnZ']
    bayes_map[atm_name] = bayes_factor
    delta_lnz_map[atm_name] = delta_lnz

# Define gas species based on planet
if planet_name.lower() in ["gj367b", "trappist-1b", "trappist-1c"]:
    # GJ367b doesn't have HCN, CO, NH3, H2S
    gas_species = ["CO$_2$", "CH$_4$", "SO$_2$", "H$_2$O", "N$_2$"]
    trace_gases_internal = ["CO2", "CH4", "SO2", "H2O"]
else:
    # GJ486b has all gases
    gas_species = ["CO$_2$", "CH$_4$", "SO$_2$", "H$_2$O", "N$_2$", "H$_2$S", "NH$_3$", "HCN", "CO"]
    trace_gases_internal = ["CO2", "CH4", "SO2", "H2O", "H2S", "NH3", "HCN", "CO"]

# Map LaTeX labels to internal gas keys
gas_label_map = {
    "CO$_2$": "CO2",
    "CH$_4$": "CH4",
    "SO$_2$": "SO2",
    "H$_2$O": "H2O",
    "N$_2$": "N2",
    "H$_2$S": "H2S",
    "NH$_3$": "NH3",
    "HCN": "HCN",
    "CO": "CO"
}

# Define all compositions based on planet
all_compositions = {}

# Pure gases (same for both planets)
for gas in ["CO2", "CH4", "SO2", "H2O"]:
    all_compositions[f"{gas}_pure"] = {gas: 1.0}

# N2 + 1 ppm (conditional based on planet)
if planet_name.lower() in ["gj367b", "trappist-1b", "trappist-1c"]:
    for gas in ["CO2", "CH4", "SO2", "H2O"]:
        all_compositions[f"N2+1ppm_{gas}"] = {"N2": 0.999, gas: 0.001}
else:
    for gas in ["CO2", "CH4", "SO2", "H2O", "H2S", "NH3", "HCN", "CO"]:
        all_compositions[f"N2+1ppm_{gas}"] = {"N2": 0.999, gas: 0.001}

# N2 + 1000 ppm (conditional based on planet)
if planet_name.lower() in ["gj367b", "trappist-1b", "trappist-1c"]:
    for gas in ["CO2", "CH4", "SO2", "H2O"]:
        all_compositions[f"N2+1000ppm_{gas}"] = {"N2": 0.9, gas: 0.1}
else:
    for gas in ["CO2", "CH4", "SO2", "H2O", "H2S", "NH3", "HCN", "CO"]:
        all_compositions[f"N2+1000ppm_{gas}"] = {"N2": 0.9, gas: 0.1}

# Create atmosphere name mapping to match CSV format
atm_name_map = {}
# Pure gases (same for both planets)
atm_name_map["CO2_pure"] = ["001bar_CO2", "01bar_CO2", "1bar_CO2", "10bar_CO2", "100bar_CO2"]
atm_name_map["CH4_pure"] = ["001bar_CH4", "01bar_CH4", "1bar_CH4", "10bar_CH4", "100bar_CH4"]
atm_name_map["SO2_pure"] = ["001bar_SO2", "01bar_SO2", "1bar_SO2", "10bar_SO2", "100bar_SO2"]
atm_name_map["H2O_pure"] = ["001bar_H2O", "01bar_H2O", "1bar_H2O", "10bar_H2O", "100bar_H2O"]

# N2 + 1000 ppm (conditional based on planet)
if planet_name.lower() in ["gj367b", "trappist-1b", "trappist-1c"]:
    for gas in ["CO2", "CH4", "SO2", "H2O"]:
        atm_name_map[f"N2+1000ppm_{gas}"] = [
            f"001bar_N2_1000ppm_{gas}",
            f"01bar_N2_1000ppm_{gas}",
            f"1bar_N2_1000ppm_{gas}",
            f"10bar_N2_1000ppm_{gas}",
            f"100bar_N2_1000ppm_{gas}"
        ]
else:
    for gas in ["CO2", "CH4", "SO2", "H2O", "H2S", "NH3", "HCN", "CO"]:
        atm_name_map[f"N2+1000ppm_{gas}"] = [
            f"001bar_N2_1000ppm_{gas}",
            f"01bar_N2_1000ppm_{gas}",
            f"1bar_N2_1000ppm_{gas}",
            f"10bar_N2_1000ppm_{gas}",
            f"100bar_N2_1000ppm_{gas}"
        ]

# N2 + 1 ppm (conditional based on planet)
if planet_name.lower() in ["gj367b", "trappist-1b", "trappist-1c"]:
    for gas in ["CO2", "CH4", "SO2", "H2O"]:
        atm_name_map[f"N2+1ppm_{gas}"] = [
            f"001bar_N2_1ppm_{gas}",
            f"01bar_N2_1ppm_{gas}",
            f"1bar_N2_1ppm_{gas}",
            f"10bar_N2_1ppm_{gas}",
            f"100bar_N2_1ppm_{gas}"
        ]
else:
    for gas in ["CO2", "CH4", "SO2", "H2O", "H2S", "NH3", "HCN", "CO"]:
        atm_name_map[f"N2+1ppm_{gas}"] = [
            f"001bar_N2_1ppm_{gas}",
            f"01bar_N2_1ppm_{gas}",
            f"1bar_N2_1ppm_{gas}",
            f"10bar_N2_1ppm_{gas}",
            f"100bar_N2_1ppm_{gas}"
        ]

# Plot setup with spacing between groups
x_labels = list(all_compositions.keys())
y_labels = ["0.01", "0.1", "1", "10", "100"]
box_size = 0.8

# Define group sizes based on planet
pure_gases = ["CO2", "CH4", "SO2", "H2O"]

# Create x positions with spacing between groups
group_spacing = 0.3  # Space between groups
x_positions = []

# Pure gases (group 1)
for i in range(len(pure_gases)):
    x_positions.append(i)

# 1000 ppm gases (group 2) - add spacing
start_pos = len(pure_gases) + group_spacing
for i in range(len(trace_gases_internal)):
    x_positions.append(start_pos + i)

# 1 ppm gases (group 3) - add more spacing
start_pos = len(pure_gases) + group_spacing + len(trace_gases_internal) + group_spacing
for i in range(len(trace_gases_internal)):
    x_positions.append(start_pos + i)

y_positions = list(range(len(y_labels)))

# Define group boundaries (with 1 ppm and 1000 ppm swapped)
swapped_group_positions = [
    (0, "Pure"),
    (len(pure_gases) + group_spacing, "1 ppm"),
    (len(pure_gases) + group_spacing + len(trace_gases_internal) + group_spacing, "1000 ppm")
]

# Extract molecule labels for x-axis
molecule_labels = []
for atm_type in x_labels:
    if "_pure" in atm_type:
        gas = atm_type.replace("_pure", "")
        # Convert to LaTeX format
        for latex_label, gas_key in gas_label_map.items():
            if gas_key == gas:
                molecule_labels.append(latex_label)
                break
    elif "N2+1000ppm_" in atm_type:
        gas = atm_type.replace("N2+1000ppm_", "")
        # Convert to LaTeX format
        for latex_label, gas_key in gas_label_map.items():
            if gas_key == gas:
                molecule_labels.append(latex_label)
                break
    elif "N2+1ppm_" in atm_type:
        gas = atm_type.replace("N2+1ppm_", "")
        # Convert to LaTeX format
        for latex_label, gas_key in gas_label_map.items():
            if gas_key == gas:
                molecule_labels.append(latex_label)
                break

# Create colormap for Bayes factors (log scale)
# Get all Bayes factors and take log
all_bayes = list(bayes_map.values())
log_bayes = np.log10(np.array(all_bayes))

# Center colormap at 0 (Bayes factor = 1)
vmax = max(abs(log_bayes.min()), abs(log_bayes.max()))
vmin = -vmax

# Use RdBu colormap: red = worse, blue = better, white = neutral (BF=1)
cmap = plt.cm.Reds_r
norm = mcolors.TwoSlopeNorm(vcenter=-10, vmin=-20, vmax=0)

fig, ax = plt.subplots(figsize=(20, 8))

# Draw each square
for i, x in enumerate(x_positions):
    for j, y in enumerate(y_positions):
        atm_type = x_labels[i]

        center_x = x
        center_y = len(y_labels) - 1 - j

        # Get Bayes factor and delta ln Z for this atmosphere and pressure
        if atm_type in atm_name_map:
            csv_atm_name = atm_name_map[atm_type][j]  # j corresponds to pressure level
            if csv_atm_name in bayes_map:
                bayes_factor = bayes_map[csv_atm_name]
                delta_lnz = delta_lnz_map[csv_atm_name]
                log_bayes_factor = np.log10(bayes_factor)
                bayes_color = cmap(norm(log_bayes_factor))
                
                # Check if delta ln Z >= -3 for gold contour
                edge_color = 'gold' if delta_lnz >= -3 else 'black'
                line_width = 2 if delta_lnz >= -3 else 1
            else:
                bayes_color = 'gray'  # Default if not found
                edge_color = 'black'
                line_width = 1
        else:
            bayes_color = 'gray'
            edge_color = 'black'
            line_width = 1

        # Draw main square with Bayes factor color and conditional gold edge
        main_square = patches.Rectangle((center_x - box_size/2, center_y - box_size/2),
                                        box_size, box_size, edgecolor=edge_color, facecolor=bayes_color, linewidth=line_width)
        ax.add_patch(main_square)

# Axis formatting
ax.set_xticks(x_positions)
ax.set_xticklabels(molecule_labels, fontsize=20)
ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(y_labels[::-1])
ax.set_xlim(-0.5, max(x_positions) + 0.5)
ax.set_ylim(-0.5, len(y_labels) - 0.5)
ax.set_ylabel("Pressure [bar]", fontsize=20)
#ax.set_title(f"Atmosphere compositions with Bayes factors - {planet_name.upper()}", fontsize=20)
ax.set_aspect('equal')

# Draw group labels underneath
for i, (start_idx, label) in enumerate(swapped_group_positions):
    if i == 0:  # Pure group
        end_idx = len(pure_gases) - 1
    elif i == 1:  # 1000 ppm group
        end_idx = start_idx + len(trace_gases_internal) - 1
    else:  # 1 ppm group
        end_idx = start_idx + len(trace_gases_internal) - 1
    
    x_center = (start_idx + end_idx) / 2
    ax.text(x_center, -1, label, ha='center', va='top', fontsize=20, fontweight='bold', transform=ax.transData)

# Add vertical lines to separate groups
group_line_color = 'lightgray'
group_line_width = 2
group_line_style = '--'

# Line between Pure and 1000 ppm
line1_x = len(pure_gases) + group_spacing/2 - 0.5
ax.axvline(x=line1_x, color=group_line_color, linewidth=group_line_width, linestyle=group_line_style, alpha=0.7)

# Line between 1000 ppm and 1 ppm
line2_x = len(pure_gases) + group_spacing + len(trace_gases_internal) + group_spacing/2 - 0.5
ax.axvline(x=line2_x, color=group_line_color, linewidth=group_line_width, linestyle=group_line_style, alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=16)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.84, aspect=10, pad=0.015)
cbar.set_label('Bayes factor (log scale)', fontsize=18,  labelpad=10)
#cbar.set_ticks([-50, -40, -30, -20, -10 ,0,0.2, 0.4,0.6,0.8,1])
#cbar.set_ticklabels(['-50','-40','-30','-20','-10','0.0','0.2','0.4','0.6', '0.8','1.0'])
cbar.ax.tick_params(labelsize=15)

plt.tight_layout()
# Uncomment the line below to save the figure
plt.savefig(f"compositions_with_bayes_{planet_name}.pdf", format = "pdf", bbox_inches='tight', dpi=300)
plt.show()