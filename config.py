"""
Configuration for DESC quasi-axisymmetric (QA) stellarator optimization.

All configurations enforce QA symmetry: helicity = (M=1, N=0).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# QA symmetry constant – do not change
# ---------------------------------------------------------------------------
QA_HELICITY = (1, 0)  # (M, N) for quasi-axisymmetry


# ---------------------------------------------------------------------------
# Equilibrium initial-guess configuration
# ---------------------------------------------------------------------------
@dataclass
class EquilibriumConfig:
    """Parameters that define the initial equilibrium before optimization.

    Attributes
    ----------
    R_lmn : list[float]
        Fourier coefficients for the R component of the boundary surface.
    Z_lmn : list[float]
        Fourier coefficients for the Z component of the boundary surface.
    modes_R : list[list[int]]
        Mode numbers [m, n] corresponding to each R_lmn entry.
    modes_Z : list[list[int]]
        Mode numbers [m, n] corresponding to each Z_lmn entry.
    NFP : int
        Number of toroidal field periods.
    Psi : float
        Total toroidal magnetic flux (Wb). Determines field strength.
    M : int
        Poloidal spectral resolution of the equilibrium.
    N : int
        Toroidal spectral resolution of the equilibrium.
    sym : bool
        Whether to enforce stellarator symmetry.
    """

    R_lmn: list = field(default_factory=lambda: [1.0, 0.125, 0.1])
    Z_lmn: list = field(default_factory=lambda: [-0.125, -0.1])
    modes_R: list = field(default_factory=lambda: [[0, 0], [1, 0], [0, 1]])
    modes_Z: list = field(default_factory=lambda: [[-1, 0], [0, -1]])
    NFP: int = 2
    Psi: float = 0.04
    M: int = 4
    N: int = 4
    sym: bool = True


# ---------------------------------------------------------------------------
# Multigrid optimization configuration
# ---------------------------------------------------------------------------
@dataclass
class MultigridConfig:
    """Settings for the multigrid QA optimization loop.

    The optimizer progressively unfreezes boundary modes with |m|,|n| <= k
    for k in ``k_stages``.

    Attributes
    ----------
    k_stages : list[int]
        Sequence of maximum mode numbers to optimize over.
        E.g. [1, 2, 3] optimizes modes |m|,|n| <= 1 first, then <= 2, etc.
    rho_surfaces : list[float]
        Normalized toroidal flux labels where QS error is evaluated.
    aspect_ratio_target : float
        Target aspect ratio (R0/a) for the equilibrium.
    aspect_ratio_weight : float
        Relative weight of the aspect-ratio penalty in the objective.
    maxiter_per_stage : int
        Maximum optimizer iterations at each multigrid stage.
    optimizer_method : str
        DESC optimizer method string.
    verbose : int
        Verbosity level (0 = silent, 3 = maximum).
    ftol : float
        Function tolerance for convergence.
    xtol : float
        Parameter tolerance for convergence.
    gtol : float
        Gradient tolerance for convergence.
    initial_trust_ratio : float
        Starting trust-region radius (lower = more cautious steps).
    options : dict
        Extra keyword options forwarded to ``eq.optimize(...)``.
    """

    k_stages: list = field(default_factory=lambda: [1, 2, 3])
    rho_surfaces: list = field(default_factory=lambda: [0.6, 0.8, 1.0])
    aspect_ratio_target: float = 8.0
    aspect_ratio_weight: float = 2.0
    maxiter_per_stage: int = 20
    optimizer_method: str = "proximal-lsq-exact"
    verbose: int = 3
    ftol: float = 1e-4
    xtol: float = 1e-6
    gtol: float = 1e-6
    initial_trust_ratio: float = 1.0
    options: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constrained optimization configuration
# ---------------------------------------------------------------------------
@dataclass
class ConstrainedConfig:
    """Settings for constrained QA optimization (single-stage, all modes free).

    Instead of the multigrid mode-freezing strategy, this approach frees
    *all* boundary modes and relies on inequality constraints (aspect ratio,
    elongation, volume, rotational transform) to regularize the problem.

    Attributes
    ----------
    rho_surfaces : list[float]
        Normalized toroidal flux labels where QS error is evaluated.
    aspect_ratio_bounds : tuple[float, float]
        (min, max) allowed aspect ratio.
    elongation_bounds : tuple[float, float]
        (min, max) allowed elongation.
    target_volume : Optional[float]
        Target plasma volume (m^3). ``None`` → use the initial eq volume.
    target_iota : Optional[float]
        Target average rotational transform. ``None`` → unconstrained.
    maxiter : int
        Maximum optimizer iterations.
    optimizer_method : str
        DESC optimizer method string.
    verbose : int
        Verbosity level.
    ftol : float
        Function tolerance.
    xtol : float
        Parameter tolerance.
    gtol : float
        Gradient tolerance.
    initial_trust_ratio : float
        Starting trust-region radius.
    options : dict
        Extra keyword options forwarded to ``eq.optimize(...)``.
    """

    rho_surfaces: list = field(default_factory=lambda: [0.6, 0.8, 1.0])
    aspect_ratio_bounds: tuple = (7.0, 9.0)
    elongation_bounds: tuple = (0.0, 3.0)
    target_volume: Optional[float] = None
    target_iota: Optional[float] = 0.42
    maxiter: int = 100
    optimizer_method: str = "proximal-lsq-exact"
    verbose: int = 3
    ftol: float = 1e-4
    xtol: float = 1e-6
    gtol: float = 1e-6
    initial_trust_ratio: float = 1.0
    options: dict = field(default_factory=dict)
