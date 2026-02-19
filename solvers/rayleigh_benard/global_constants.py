"""Parameter grids for Rayleigh-Bénard convection simulations.

Based on RudyMorel/the-well-rbc-sf.
"""

from pathlib import Path

# Parameters (PDE+IC) for Rayleigh-Bénard convection
RBC_GRID = {
    'resolution': [(512, 128)],
    'rayleigh': [1e6, 1e7, 1e8, 1e9, 1e10],
    'prandtl': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    'dT': [0.2, 0.4, 0.6, 0.8, 1.0],
    'seed': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    'init': ["default"],
}

# Default output path
OUTPUT_PATH = Path(__file__).parent / "output"

# Filename template
filename_rbc = "rbc_{}x{}_rayleigh_{:.2e}_prandtl_{:.2e}_dT_{:.2e}_seed_{}"
