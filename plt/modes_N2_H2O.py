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

plt.rcParams.update({'font.size': 16})

if __name__ == "__main__":

    planet = "gj367b"
    surface = "greybody"
    reuse_existing = True

    atmospheres = {
        "100bar_N2_1000ppm_H2O": 100,
        "10bar_N2_1000ppm_H2O": 10,
        "1bar_N2_1000ppm_H2O": 1.0,
        "01bar_N2_1000ppm_H2O": 0.1,
        "001bar_N2_1000ppm_H2O": 0.01
    }

    redistribution_modes = {
        "dayside": 2/3,
        "full": 1/4,
        "half": 11/24,
        "more": 17/48,
        "less": 9/16
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
        reference_surface='hematite',
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
    norm = mcolors.TwoSlopeNorm(vcenter=1, vmin=0, vmax=9)

    scatter = ax.scatter(
        merged_df["f_factor"],
        merged_df["pressure_bar"],
        c=bayes_vals,
        cmap="RdBu",
        s=400,
        edgecolors='k',
        norm=norm
    )


    # Highlight points with ΔlnZ >= -3
    highlight_mask = merged_df["ΔlnZ"] >= -3
    ax.scatter(
        merged_df.loc[highlight_mask, "f_factor"],
        merged_df.loc[highlight_mask, "pressure_bar"],
        c= bayes_vals[highlight_mask],
        cmap="RdBu",
        s=400,
        edgecolors='gold',
        linewidths=1.5,
        norm=norm
    )

    # Set x-ticks as fractions
    factors = sorted(set(redistribution_modes.values()))
    fraction_labels = {
        2/3: r"$\dfrac{2}{3}$" + "\nDayside",
        1/4: r"$\dfrac{1}{4}$" + "\nFull",
        11/24: r"$\dfrac{11}{24}$" + "\nSemi",
        17/48: r"$\dfrac{17}{48}$",
        9/16: r"$\dfrac{9}{16}$"
    }
    ax.set_xticks(factors)
    ax.set_xticklabels([fraction_labels[f] for f in factors])

    ax.set_xlabel("$f$ factor (redistribution)", fontsize=16)
    ax.set_ylabel("Atmosphere pressure (bar)", fontsize=16)
    ax.set_yscale("log")
    ax.set_title("N$_2$ + 1000 ppm H$_2$O", fontsize=20)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Bayes factor w.r.t. pyrite", fontsize=14)
    cbar.set_ticks([0.25, 0.5, 0.75, 1, 3, 5, 7,9])
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, planet, f"{planet}_bayes_comparison_plot_N2_H2O.pdf")
    plt.savefig(plot_path, format = 'pdf', dpi=300)
    print(f"\n[INFO] Plot saved to {plot_path}")

    plt.show()
