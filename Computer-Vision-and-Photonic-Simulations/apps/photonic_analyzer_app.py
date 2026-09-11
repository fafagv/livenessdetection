"""
photonic_analyzer_app.py — CLI entry point for photonic waveguide simulations.

New file (the original project had no single entry point for the photonics
scripts — calculate_mode_properties.py, "main (4).py", and "main1 (2).py"
each ran standalone with no shared CLI). This app ties together:
    - src/simulations/photonics/mode_properties.py
    - src/simulations/photonics/waveguide_sensitivity.py
    - src/simulations/photonics/wavelength_shift.py

Run:
    python -m apps.photonic_analyzer_app --wavelength 1.55 --analyte-n 1.45 --plot
"""

import argparse
import logging

from src.simulations.photonics.mode_properties import calculate_mode_properties
from src.simulations.photonics.waveguide_sensitivity import calculate_sensitivity
from src.simulations.photonics.wavelength_shift import extract_wavelength_shift, plot_wavelength_shift

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Photonic-crystal waveguide sensor analyzer.")
    parser.add_argument("--wavelength", type=float, default=1.55, help="Design wavelength (µm).")
    parser.add_argument("--lattice-constant", type=float, default=0.42, help="Lattice constant (µm).")
    parser.add_argument("--holes-diameter", type=float, default=0.25, help="Hole diameter (µm).")
    parser.add_argument("--slab-thickness", type=float, default=0.26, help="Slab thickness (µm).")
    parser.add_argument("--analyte-n", type=float, default=1.45, help="Analyte refractive index.")
    parser.add_argument("--wl-min", type=float, default=1.38, help="Wavelength sweep min (µm).")
    parser.add_argument("--wl-max", type=float, default=1.74, help="Wavelength sweep max (µm).")
    parser.add_argument("--plot", action="store_true", help="Show a sensitivity-vs-wavelength plot.")
    return parser


def run(args=None):
    parser = build_parser()
    ns = parser.parse_args(args=args)

    mode_properties = calculate_mode_properties(
        ns.wavelength, ns.lattice_constant, ns.holes_diameter, ns.slab_thickness
    )
    logger.info("Computed %d mode points.", len(mode_properties))

    sensitivities = calculate_sensitivity(mode_properties, ns.analyte_n)
    logger.info("Mean sensitivity: %.4f", sum(sensitivities) / len(sensitivities))

    shift_results = extract_wavelength_shift(
        wavelength=ns.wavelength,
        lattice_constant=ns.lattice_constant,
        holes_diameter=ns.holes_diameter,
        slab_thickness=ns.slab_thickness,
        analyte_refractive_index=ns.analyte_n,
        wavelength_range=(ns.wl_min, ns.wl_max),
    )
    for wl, sensitivity in shift_results:
        print(f"Wavelength: {wl:.4f} \u03bcm, Sensitivity: {sensitivity:.4f}")

    if ns.plot:
        plot_wavelength_shift(shift_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
