import os
from itertools import product
from tomlkit import document, table, inline_table, dumps
import toml

# paths
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = toml.load(os.path.join(ROOT, "..", "agni_config.toml"))["paths"]

# Output directory
OUTPUT_DIR = os.path.join(ROOT, "..", "res/atmospheres")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# clean-up old TOML's
keep_keywords = ["earth_like", "bare_rock", "SiO"]
for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith(".toml") and not any(kw in filename for kw in keep_keywords):
        try:
            os.remove(os.path.join(OUTPUT_DIR, filename))
            print(f"🗑 Removed {filename}")
        except Exception as e:
            print(f"Could not remove {filename}: {e}")

# Surface pressures in bar
pressures = [1e-2, 1e-1, 1, 10, 100]
p_top_default = 1e-5

# Pure gases
pure_gases = ["H2O", "CO2", "CO", "CH4", "SO2"]

# N2 + trace gas mixes
trace_species = ["H2O", "CH4", "CO2", "SO2", "HCN", "H2S", "CO"]
trace_levels = [1e-3, 1e-6]  # 0.1%, 1 ppm

# some N2 + H2O + O3 example mixtures
n2_h2o_o3_cases = [
    {"N2": 0.98, "H2O": 0.019, "O3": 0.001},
    {"N2": 0.90, "H2O": 0.099, "O3": 0.001},
    {"N2": 0.70, "H2O": 0.299, "O3": 0.001}
]

def format_pressure(p):
    """Format pressures like: 0.01 → '001', 0.1 → '01', 1 → '1', 10 → '10', 100 → '100'"""
    if p == 0.01:
        return "001"
    elif p == 0.1:
        return "01"
    else:
        return str(int(p))

def write_toml(filename, title, p_surf, p_top, vmr_dict):
    doc = document()
    doc["title"] = title

    comp = table()
    comp["p_surf"] = p_surf
    comp["p_top"] = p_top

    vmr = inline_table()
    for k, v in vmr_dict.items():
        vmr[k] = round(v, 9)
    comp["vmr_dict"] = vmr

    doc["composition"] = comp

    full_path = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(full_path, "w") as f:
            f.write(dumps(doc))
        print(f"✔ Wrote {filename}")
    except Exception as e:
        print(f"Failed to write {filename}: {e}")

# generating

# Pure gases
for gas, p in product(pure_gases, pressures):
    p_str = format_pressure(p)
    title = f"{p:.2f} bar {gas}"
    filename = f"{p_str}bar_{gas}.toml"
    write_toml(filename, title, p, p_top_default, {gas: 1.0})

# N2 + trace gas mixes (0.1% and 1 ppm)
for species, trace in product(trace_species, trace_levels):
    suffix = f"{int(trace * 1e6)}ppm"
    mixname = f"N2_{suffix}_{species}"
    vmr_dict = {"N2": 1 - trace, species: trace}

    for p in pressures:
        p_str = format_pressure(p)
        title = f"{p:.2f} bar N2 + {suffix} {species}"
        filename = f"{p_str}bar_{mixname}.toml"
        write_toml(filename, title, p, p_top_default, vmr_dict)

# N2 + H2O + O3 cases
for i, vmr_dict in enumerate(n2_h2o_o3_cases):
    for p in pressures:
        p_str = format_pressure(p)
        title = f"{p:.2f} bar N2 + H2O + O3 ({i+1})"
        filename = f"{p_str}bar_N2_H2O_O3_case{i+1}.toml"
        write_toml(filename, title, p, p_top_default, vmr_dict)
