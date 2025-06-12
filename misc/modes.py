import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import sys
import toml
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.nested_sampling import compare_models 
from src.pipeline import main as run_pipeline

config_path = os.path.join(ROOT, "agni_config.toml")
CONFIG = toml.load(config_path)["paths"]
OUTPUT_DIR = CONFIG["output_dir"]

if __name__ == "__main__":

    planet = "gj486b"
    surface = "greybody"
    reuse_existing = True

    atmospheres = {
        "1bar_H2O": 1.0,
        "01bar_H2O": 0.1,
        "001bar_H2O": 0.01
    }

    redistribution_modes = {
        "substellar": 1.0,
        "dayside": 2/3,
        "full": 1/4,
        "half": 11/24,
        "more": 17/48,
        "less": 27/48
    }

    model_entries = []

    for atmo_key, pressure in atmospheres.items():
        for mode, f in redistribution_modes.items():
            full_atmo_name = f"{atmo_key}_{mode}" if mode != "dayside" else atmo_key
            output_path = os.path.join(OUTPUT_DIR, planet, surface, full_atmo_name, "atm.nc")

            if reuse_existing and os.path.isfile(output_path):
                print(f"[SKIP] Using existing AGNI output: {output_path}")
            else:
                print(f"[INFO] Running AGNI pipeline: Atmosphere={atmo_key}, Mode={mode}")
                sys.argv = ["pipeline.py", planet, "-s", surface, "-a", atmo_key, "-T", mode]
                run_pipeline()

            model_entries.append({
                "atmosphere": atmo_key,
                "mode": mode,
                "full_name": full_atmo_name,
                "f_factor": f,
                "pressure_bar": pressure
            })

    atmo_mode_names = [entry["full_name"] for entry in model_entries]

    df = compare_models(
        planet_name=planet,
        surfaces=[surface],
        atmospheres=atmo_mode_names,
        use_greybody_reference=True,
        write_to_csv=False
    )

    df_filtered = df[df["surface"] == surface].copy()

    model_df = pd.DataFrame(model_entries)
    merged_df = model_df.merge(
        df_filtered,
        left_on="full_name",
        right_on="atmosphere",
        how="left",
        suffixes=("", "_matched")
    )

    merged_df["logZ"] = merged_df["logZ"].fillna(-999)
    merged_df["ΔlnZ"] = merged_df["ΔlnZ"].fillna(-999)
    merged_df["bayes_factor"] = merged_df["bayes_factor"].fillna(1e-3)

    summary_df = merged_df[["atmosphere", "mode", "logZ", "ΔlnZ", "bayes_factor"]].sort_values("ΔlnZ", ascending=False)
    print("\nBayesian Model Comparison Summary:")
    print(summary_df.to_string(index=False, float_format="{:.2f}".format))

    fig, ax = plt.subplots(figsize=(8, 6))

    bayes_vals = merged_df["bayes_factor"].values
    norm = mcolors.TwoSlopeNorm(vcenter=1, vmin=bayes_vals.min(), vmax=bayes_vals.max())

    scatter = ax.scatter(
        merged_df["f_factor"],
        merged_df["pressure_bar"],
        c=bayes_vals,
        cmap="RdBu",
        s=100,
        edgecolors='k',
        norm=norm
    )

    # Set x-ticks as fractions
    factors = sorted(set(redistribution_modes.values()))
    fraction_labels = {
        1.0: "1\nSubstellar",
        2/3: r"$\dfrac{2}{3}$" + "\nDayside",
        1/4: r"$\dfrac{1}{4}$" + "\nFull",
        11/24: r"$\dfrac{11}{24}$" + "\nSemi",
        17/48: r"$\dfrac{17}{48}$",
        27/48: r"$\dfrac{27}{48}$" 
    }   
    ax.set_xticks(factors)
    ax.set_xticklabels([fraction_labels[f] for f in factors])

    ax.set_xlabel("$f$ factor (redistribution)")
    ax.set_ylabel("H$_2$O pressure (bar)")
    ax.set_yscale("log")
    ax.set_title("H$_2$O pressure vs heat redistribution: bayes factor")
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Bayes factor w.r.t. greybody")
    cbar.set_ticks([0.25, 0.5, 0.75, 1, 2, 3, 4])
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, planet, f"{planet}_bayes_comparison_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[INFO] Plot saved to {plot_path}")

    plt.show()
