"""
mode_properties.py — Effective refractive index (n_eff) calculation for a
photonic-crystal slab waveguide.

Refactored from calculate_mode_properties.py.

Bug fixed: the original file defined `calculate_mode_properties()` TWICE —
an empty stub, immediately shadowed by the real implementation below it.
Python silently used the second definition, but the dead stub made the file
confusing to read and easy to break during future edits, so it's removed
here. The module-level script code (which ran on import) has also been
moved behind `if __name__ == "__main__":` so importing this module no
longer has side effects (printing, running a full sweep).
"""

from typing import List, Tuple

import numpy as np


def calculate_neff(kx: float, k0: float, lattice_constant: float, holes_diameter: float, slab_thickness: float) -> float:
    """
    Effective refractive index for a given in-plane wavevector.

    NOTE: this is the original project's simplified placeholder formula
    (neff = sqrt(1 + (kx/k0)^2)) and does not solve the true photonic-crystal
    slab eigenvalue problem. lattice_constant, holes_diameter, and
    slab_thickness are accepted for API compatibility / future extension but
    are not yet used in the formula — replace this with a real eigenmode
    solver (e.g. MPB, Lumerical, or a custom FDTD/plane-wave-expansion
    routine) for physically accurate results.
    """
    return float(np.sqrt(1 + (kx / k0) ** 2))


def calculate_mode_properties(
    wavelength: float,
    lattice_constant: float,
    holes_diameter: float,
    slab_thickness: float,
    n_points: int = 100,
) -> List[Tuple[float, float, float]]:
    """
    Sweep the in-plane wavevector kx from 0 to k0 and return a list of
    (kx, n_eff, mode_frequency) tuples.
    """
    k0 = 2 * np.pi / wavelength
    kx_range = np.linspace(0, k0, n_points)

    mode_properties = []
    for kx in kx_range:
        neff = calculate_neff(kx, k0, lattice_constant, holes_diameter, slab_thickness)
        mode_frequency = 2 * np.pi * k0 / neff
        mode_properties.append((float(kx), neff, float(mode_frequency)))

    return mode_properties


if __name__ == "__main__":
    # Constants and parameters (from the original script)
    WAVELENGTH = 1.55  # Micrometers
    LATTICE_CONSTANT = 0.42  # Micrometers (420 nm)
    HOLES_DIAMETER = 0.25  # Micrometers (250 nm)
    SLAB_THICKNESS = 0.26  # Micrometers

    results = calculate_mode_properties(WAVELENGTH, LATTICE_CONSTANT, HOLES_DIAMETER, SLAB_THICKNESS)
    for kx, neff, mode_frequency in results:
        print(f"kx: {kx:.4f}, neff: {neff:.4f}, Mode Frequency: {mode_frequency:.4f}")
