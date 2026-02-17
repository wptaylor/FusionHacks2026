"""
Objective and constraint builders for QA stellarator optimization.

Each builder returns a (objective, constraints) pair ready for ``eq.optimize()``.
All objectives enforce **quasi-axisymmetric** symmetry with helicity (1, 0).

To add a new objective configuration, subclass ``ObjectiveBuilder`` and
implement ``build()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from desc.grid import LinearGrid
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
)
from desc.optimize import Optimizer

from config import QA_HELICITY

if TYPE_CHECKING:
    from desc.equilibrium import Equilibrium


# ===================================================================
# Abstract base
# ===================================================================
class ObjectiveBuilder(ABC):
    """Interface for constructing (objective, constraints) pairs.

    Subclass this and implement :meth:`build` to create new optimization
    configurations while keeping the runner (:mod:`optimize_qa`) unchanged.
    """

    @abstractmethod
    def build(
        self,
        eq: "Equilibrium",
        **kwargs,
    ) -> tuple[ObjectiveFunction, tuple]:
        """Return ``(objective, constraints)`` for the given equilibrium.

        Parameters
        ----------
        eq : Equilibrium
            The DESC equilibrium to build objectives for.
        **kwargs :
            Arbitrary extra parameters a subclass may need.

        Returns
        -------
        objective : ObjectiveFunction
            The combined objective to minimise.
        constraints : tuple
            Tuple of DESC objective/constraint objects.
        """
        ...


# ===================================================================
# Multigrid QA builder  (reference example from DESC tutorial)
# ===================================================================
class MultigridQABuilder(ObjectiveBuilder):
    """Build objectives for one stage of multigrid QA optimization.

    At each stage only boundary modes with ``|m|, |n| <= k`` are freed;
    all others (plus the major radius R_{0,0}) are held fixed.

    This is the QA-adapted version of the ``run_qh_step`` function from the
    DESC advanced-optimization tutorial notebook.

    Parameters
    ----------
    k : int
        Maximum mode number to free at this stage.
    rho_surfaces : array-like
        Surfaces where the QS residual is evaluated.
    aspect_ratio_target : float
        Target aspect ratio for the regularisation term.
    aspect_ratio_weight : float
        Relative weight of the aspect-ratio term.
    """

    def __init__(
        self,
        k: int,
        rho_surfaces: list[float] | np.ndarray = (0.6, 0.8, 1.0),
        aspect_ratio_target: float = 8.0,
        aspect_ratio_weight: float = 2.0,
    ):
        self.k = k
        self.rho_surfaces = np.atleast_1d(rho_surfaces)
        self.aspect_ratio_target = aspect_ratio_target
        self.aspect_ratio_weight = aspect_ratio_weight

    # ------------------------------------------------------------------ #
    def build(self, eq: "Equilibrium", **kwargs) -> tuple[ObjectiveFunction, tuple]:
        """Construct multigrid-stage objective and constraints."""

        grid = LinearGrid(
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            rho=self.rho_surfaces,
            sym=True,
        )

        # --- Objective: QS two-term + aspect-ratio regularisation --- #
        objective = ObjectiveFunction(
            (
                QuasisymmetryTwoTerm(
                    eq=eq,
                    helicity=QA_HELICITY,
                    grid=grid,
                ),
                AspectRatio(
                    eq=eq,
                    target=self.aspect_ratio_target,
                    weight=self.aspect_ratio_weight,
                ),
            )
        )

        # --- Constraints: freeze high-order modes + profiles --- #
        #  Free only modes with |m|,|n| <= k; fix R_{0,0} always.
        R_modes = np.vstack(
            (
                [0, 0, 0],  # always fix the major radius
                eq.surface.R_basis.modes[
                    np.max(np.abs(eq.surface.R_basis.modes), 1) > self.k, :
                ],
            )
        )
        Z_modes = eq.surface.Z_basis.modes[
            np.max(np.abs(eq.surface.Z_basis.modes), 1) > self.k, :
        ]

        constraints = (
            ForceBalance(eq=eq),
            FixBoundaryR(eq=eq, modes=R_modes),
            FixBoundaryZ(eq=eq, modes=Z_modes),
            FixPressure(eq=eq),
            FixCurrent(eq=eq),
            FixPsi(eq=eq),
        )

        return objective, constraints


# ===================================================================
# Constrained QA builder  (reference: second half of tutorial notebook)
# ===================================================================
class ConstrainedQABuilder(ObjectiveBuilder):
    """Build objectives for constrained QA optimization.

    All boundary modes (except R_{0,0}) are free.  Regularisation comes
    from inequality constraints on geometric quantities (aspect ratio,
    elongation, volume, iota).

    Parameters
    ----------
    rho_surfaces : array-like
        Surfaces where the QS residual is evaluated.
    aspect_ratio_bounds : tuple[float, float]
        (min, max) aspect ratio.
    elongation_bounds : tuple[float, float]
        (min, max) elongation.
    target_volume : float | None
        Target plasma volume.  ``None`` → use ``eq.compute("V")["V"]``.
    target_iota : float | None
        Target average iota.  ``None`` → omit iota constraint.
    """

    def __init__(
        self,
        rho_surfaces=(0.6, 0.8, 1.0),
        aspect_ratio_bounds=(7.0, 9.0),
        elongation_bounds=(0.0, 3.0),
        target_volume=None,
        target_iota=0.42,
    ):
        self.rho_surfaces = np.atleast_1d(rho_surfaces)
        self.aspect_ratio_bounds = aspect_ratio_bounds
        self.elongation_bounds = elongation_bounds
        self.target_volume = target_volume
        self.target_iota = target_iota

    # ------------------------------------------------------------------ #
    def build(self, eq: "Equilibrium", **kwargs) -> tuple[ObjectiveFunction, tuple]:
        """Construct constrained objective and constraints."""

        # Late imports for optional objectives that may not always be needed
        from desc.objectives import Elongation, RotationalTransform, Volume

        grid = LinearGrid(
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            rho=self.rho_surfaces,
            sym=True,
        )

        # --- Objective: pure QS error --- #
        objective = ObjectiveFunction(
            QuasisymmetryTwoTerm(eq=eq, helicity=QA_HELICITY, grid=grid),
        )

        # --- Constraints --- #
        # Only fix the major-radius R_{0,0} mode
        R_modes_fixed = np.array([[0, 0, 0]])

        constraint_list = [
            ForceBalance(eq=eq),
            FixBoundaryR(eq=eq, modes=R_modes_fixed),
            # FixBoundaryZ intentionally omitted → all Z modes free
            FixPressure(eq=eq),
            FixCurrent(eq=eq),
            FixPsi(eq=eq),
            AspectRatio(eq=eq, bounds=self.aspect_ratio_bounds),
            Elongation(eq=eq, bounds=self.elongation_bounds),
        ]

        vol = self.target_volume
        if vol is None:
            vol = eq.compute("V")["V"]
        constraint_list.append(Volume(eq=eq, target=vol))

        if self.target_iota is not None:
            constraint_list.append(
                RotationalTransform(eq=eq, target=self.target_iota)
            )

        return objective, tuple(constraint_list)


# ===================================================================
# Custom builder template  (copy and modify this for new objectives)
# ===================================================================
class CustomQABuilder(ObjectiveBuilder):
    """Template for a user-defined QA objective configuration.

    Copy this class and fill in ``build()`` with your own objectives and
    constraints.  Any DESC objective can be used here — see the full list at
    https://desc-docs.readthedocs.io/en/stable/api.html#objectives

    Common objectives to consider:
        - ``QuasisymmetryBoozer``   — Boozer-coordinate QS metric
        - ``QuasisymmetryTripleProduct`` — triple-product QS metric
        - ``MagneticWell``          — penalise magnetic hill
        - ``MercierStability``      — Mercier stability criterion
        - ``BootstrapRedlConsistency`` — bootstrap-current self-consistency
        - ``Elongation``, ``Volume``, ``AspectRatio`` — geometric bounds
        - ``RotationalTransform``   — target specific iota profile
        - ``MeanCurvature``         — target mean curvature of boundary
        - ``PlasmaVesselDistance``   — minimum plasma-vessel clearance
    """

    def __init__(self, **kwargs):
        # Store any parameters your custom objective needs
        self.params = kwargs

    def build(self, eq: "Equilibrium", **kwargs) -> tuple[ObjectiveFunction, tuple]:
        """Construct your custom objective and constraints.

        Returns
        -------
        objective : ObjectiveFunction
        constraints : tuple
        """
        raise NotImplementedError(
            "CustomQABuilder.build() is a template — "
            "copy this class and implement your own objectives."
        )

        # ----- Example skeleton (uncomment and modify) ----- #
        # grid = LinearGrid(
        #     M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP,
        #     rho=np.array([0.6, 0.8, 1.0]), sym=True,
        # )
        #
        # objective = ObjectiveFunction((
        #     QuasisymmetryTwoTerm(eq=eq, helicity=QA_HELICITY, grid=grid),
        #     # Add more objectives here ...
        # ))
        #
        # constraints = (
        #     ForceBalance(eq=eq),
        #     FixPressure(eq=eq),
        #     FixCurrent(eq=eq),
        #     FixPsi(eq=eq),
        #     # Add more constraints here ...
        # )
        #
        # return objective, constraints
