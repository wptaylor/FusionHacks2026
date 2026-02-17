"""
Core optimization runners for quasi-axisymmetric (QA) stellarator design.

Two entry-point functions are provided:

- ``run_multigrid_qa``   – progressive mode-unfreezing (the "precise QA" recipe)
- ``run_constrained_qa`` – single-stage with inequality constraints

Both return the optimised ``Equilibrium`` and an ``EquilibriaFamily``
containing all intermediate results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from desc.continuation import solve_continuation_automatic
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.optimize import Optimizer

from config import (
    ConstrainedConfig,
    EquilibriumConfig,
    MultigridConfig,
    QA_HELICITY,
)
from objectives import ConstrainedQABuilder, MultigridQABuilder, ObjectiveBuilder

if TYPE_CHECKING:
    pass


# ===================================================================
# Equilibrium creation helpers
# ===================================================================
def create_initial_equilibrium(
    cfg: EquilibriumConfig,
    verbose: int = 0,
) -> Equilibrium:
    """Create and solve the initial fixed-boundary equilibrium.

    Parameters
    ----------
    cfg : EquilibriumConfig
        Surface geometry, resolution, and field-period settings.
    verbose : int
        Verbosity for the continuation solver.

    Returns
    -------
    eq : Equilibrium
        A solved DESC equilibrium ready for optimisation.
    """
    surf = FourierRZToroidalSurface(
        R_lmn=cfg.R_lmn,
        Z_lmn=cfg.Z_lmn,
        modes_R=cfg.modes_R,
        modes_Z=cfg.modes_Z,
        NFP=cfg.NFP,
        sym=cfg.sym,
    )
    eq = Equilibrium(M=cfg.M, N=cfg.N, Psi=cfg.Psi, surface=surf)
    eq_solved = solve_continuation_automatic(eq, verbose=verbose)[-1]
    return eq_solved


def create_initial_equilibrium_from_file(
    path: str,
    file_format: Optional[str] = None,
) -> Equilibrium:
    """Load an already-solved equilibrium from a DESC or VMEC file.

    Parameters
    ----------
    path : str
        Path to the saved equilibrium (``.h5`` for DESC, or VMEC ``wout_*.nc``).
    file_format : str, optional
        ``"desc"`` or ``"vmec"``.  Auto-detected from extension if ``None``.

    Returns
    -------
    eq : Equilibrium
    """
    import desc.io

    if file_format is None:
        if path.endswith(".h5"):
            file_format = "desc"
        else:
            file_format = "vmec"

    if file_format == "desc":
        eq = desc.io.load(path)
        if isinstance(eq, EquilibriaFamily):
            eq = eq[-1]
    else:
        from desc.vmec import VMECIO
        eq = VMECIO.load(path)

    return eq


# ===================================================================
# Multigrid QA optimization
# ===================================================================
def run_multigrid_qa(
    eq: Equilibrium,
    cfg: MultigridConfig | None = None,
    builder_cls: type[ObjectiveBuilder] = MultigridQABuilder,
) -> tuple[Equilibrium, EquilibriaFamily]:
    """Run multigrid quasi-axisymmetric optimisation.

    Progressively unfreezes boundary modes with ``|m|, |n| <= k`` for each
    ``k`` in ``cfg.k_stages``, running the optimizer at each stage.

    This is the QA-adapted version of the "precise QH" example from the
    DESC advanced-optimization tutorial.

    Parameters
    ----------
    eq : Equilibrium
        Initial solved equilibrium.
    cfg : MultigridConfig, optional
        Optimisation hyper-parameters.  Uses defaults if ``None``.
    builder_cls : type[ObjectiveBuilder]
        The objective builder class to use at each stage.  Defaults to
        :class:`MultigridQABuilder`.

    Returns
    -------
    eq_final : Equilibrium
        The optimised equilibrium after all multigrid stages.
    eqfam : EquilibriaFamily
        Contains the initial equilibrium and the result of every stage.
    """
    if cfg is None:
        cfg = MultigridConfig()

    optimizer = Optimizer(cfg.optimizer_method)
    eqfam = EquilibriaFamily(eq)
    eq_current = eq.copy()

    for k in cfg.k_stages:
        print(f"\n{'='*60}")
        print(f"  Multigrid QA stage: k = {k}")
        print(f"{'='*60}\n")

        builder = builder_cls(
            k=k,
            rho_surfaces=cfg.rho_surfaces,
            aspect_ratio_target=cfg.aspect_ratio_target,
            aspect_ratio_weight=cfg.aspect_ratio_weight,
        )
        objective, constraints = builder.build(eq_current)

        # Merge per-stage options with global defaults
        opts = {
            "initial_trust_ratio": cfg.initial_trust_ratio,
        }
        opts.update(cfg.options)

        eq_new, history = eq_current.optimize(
            objective=objective,
            constraints=constraints,
            optimizer=optimizer,
            maxiter=cfg.maxiter_per_stage,
            ftol=cfg.ftol,
            xtol=cfg.xtol,
            gtol=cfg.gtol,
            verbose=cfg.verbose,
            copy=True,
            options=opts,
        )

        eqfam.append(eq_new)
        eq_current = eq_new

    return eq_current, eqfam


# ===================================================================
# Constrained QA optimization
# ===================================================================
def run_constrained_qa(
    eq: Equilibrium,
    cfg: ConstrainedConfig | None = None,
    builder: ObjectiveBuilder | None = None,
) -> tuple[Equilibrium, EquilibriaFamily]:
    """Run single-stage constrained quasi-axisymmetric optimisation.

    All boundary modes (except R_{0,0}) are free.  Regularisation is
    provided by inequality constraints on aspect ratio, elongation, volume,
    and (optionally) rotational transform.

    Parameters
    ----------
    eq : Equilibrium
        Initial solved equilibrium.
    cfg : ConstrainedConfig, optional
        Optimisation hyper-parameters.  Uses defaults if ``None``.
    builder : ObjectiveBuilder, optional
        Override the default :class:`ConstrainedQABuilder`.

    Returns
    -------
    eq_opt : Equilibrium
        The optimised equilibrium.
    eqfam : EquilibriaFamily
        Contains initial and final equilibria.
    """
    if cfg is None:
        cfg = ConstrainedConfig()

    if builder is None:
        builder = ConstrainedQABuilder(
            rho_surfaces=cfg.rho_surfaces,
            aspect_ratio_bounds=cfg.aspect_ratio_bounds,
            elongation_bounds=cfg.elongation_bounds,
            target_volume=cfg.target_volume,
            target_iota=cfg.target_iota,
        )

    objective, constraints = builder.build(eq)
    optimizer = Optimizer(cfg.optimizer_method)

    opts = {
        "initial_trust_ratio": cfg.initial_trust_ratio,
    }
    opts.update(cfg.options)

    eqfam = EquilibriaFamily(eq)

    eq_opt, history = eq.optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        maxiter=cfg.maxiter,
        ftol=cfg.ftol,
        xtol=cfg.xtol,
        gtol=cfg.gtol,
        verbose=cfg.verbose,
        copy=True,
        options=opts,
    )

    eqfam.append(eq_opt)
    return eq_opt, eqfam


# ===================================================================
# Post-optimization utilities
# ===================================================================
def resolve_equilibrium(eq: Equilibrium, verbose: int = 1) -> Equilibrium:
    """Re-solve the equilibrium to tighten force-balance after optimization.

    After boundary optimization the force-balance residual may be larger
    than desired.  Calling ``eq.solve()`` brings it back down without
    altering the boundary.

    Parameters
    ----------
    eq : Equilibrium
        An optimised equilibrium with potentially elevated force error.
    verbose : int
        Verbosity for the solver.

    Returns
    -------
    eq_clean : Equilibrium
        The re-solved equilibrium with reduced force-balance error.
    """
    eq_clean = eq.copy()
    eq_clean.solve(verbose=verbose)
    return eq_clean


def save_results(
    eqfam: EquilibriaFamily,
    path: str = "qa_optimization_output.h5",
) -> None:
    """Save an equilibrium family to a DESC HDF5 file.

    Parameters
    ----------
    eqfam : EquilibriaFamily
        The family of equilibria (initial + intermediate + final).
    path : str
        Output file path.
    """
    import desc.io

    desc.io.save(eqfam, path)
    print(f"Saved {len(eqfam)} equilibria to {path}")
