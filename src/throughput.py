import numpy as np
import matplotlib.pyplot as plt
import os
from pandeia.engine.instrument_factory import InstrumentFactory

def get_throughput(wave_um, filter_name):
    conf = {
        "detector": {
            "nexp": 1,
            "ngroup": 10,
            "nint": 1,
            "readout_pattern": "fastr1",
            "subarray": "full"
        },
        "instrument": {
            "aperture": "imager",
            "filter": filter_name,
            "instrument": "miri",
            "mode": "imaging"
        },
    }
    instrument_factory = InstrumentFactory(config=conf)
    return instrument_factory.get_total_eff(wave_um)

def main():
    # Wavelength range in microns
    wave_um = np.linspace(10.0, 20.0, 1000)

    # Get throughput for F1280W and F1500W
    efficiency_1280 = get_throughput(wave_um, "f1280w")
    efficiency_1500 = get_throughput(wave_um, "f1500w")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wave_um, efficiency_1280, label="F1280W", color="dodgerblue")
    ax.plot(wave_um, efficiency_1500, label="F1500W", color="crimson")
    ax.set_title("JWST MIRI throughput: F1280W and F1500W")
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel("Throughput")
    ax.grid(True, linestyle=":")
    ax.legend()

    # Save
    out_dir = "out/throughput"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "miri_f1280w_f1500w_throughput.png"), dpi=300)
    print(f"Saved plot to {os.path.join(out_dir, 'miri_f1280w_f1500w_throughput.png')}")

if __name__ == "__main__":
    main()
