#!/usr/bin/env python
"""
Example: Precise quasi-axisymmetric (QA) stellarator optimization.

Adapted from the DESC advanced-optimization tutorial notebook
(PlasmaControl/DESC, commit b29e1911).  The original notebook targets
quasi-helical (QH) symmetry; this script is locked to QA (helicity M=1,N=0).

Two optimization strategies are demonstrated:
  1. Multigrid approach  – progressively unfreezes boundary modes
  2. Constrained approach – all modes free, regularised by inequality bounds

Usage
-----
    python example_precise_qa.py              # run both strategies
    python example_precise_qa.py --multigrid  # multigrid only
    python example_precise_qa.py --constrained # constrained only

Requirements
------------
    pip install desc-opt
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# ------------------------------------------------------------------
# Optional: enable GPU acceleration (uncomment if a CUDA GPU is available)
# ------------------------------------------------------------------
# from desc import set_device
# set_device("gpu")

# ------------------------------------------------------------------
# Optional: JAX compilation cache (speeds up re-runs)
# ------------------------------------------------------------------
# import jax
# jax.config.update("jax_compilation_cache_dir", "./jax-caches")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from config import ConstrainedConfig, EquilibriumConfig, MultigridConfig
from optimize_qa import (
    create_initial_equilibrium,
    resolve_equilibrium,
    run_constrained_qa,
    run_multigrid_qa,
    save_results,
)


# ===================================================================
# 1.  Define the initial equilibrium
# ===================================================================
def make_initial_eq():
    """Create a simple initial equilibrium for QA optimisation.

    This is a QA-oriented starting point: NFP=2, aspect ratio ~8,
    circular cross-section with slight axis torsion to break axisymmetry.
    """
    eq_cfg = EquilibriumConfig(
        # Boundary Fourier coefficients
        R_lmn=[1.0, 0.125, 0.1],
        Z_lmn=[-0.125, -0.1],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
        # Field periods — typical for QA
        NFP=2,
        # Toroidal flux (Wb): Psi ~ (B * a^2) / 2;  gives B ~ 1 T
        Psi=0.04,
        # Spectral resolution (increase for production runs)
        M=4,
        N=4,
    )

    print("Solving initial equilibrium …")
    eq0 = create_initial_equilibrium(eq_cfg, verbose=1)
    print("Initial equilibrium solved.\n")
    return eq0


# ===================================================================
# 2.  Multigrid QA optimization
# ===================================================================
def demo_multigrid(eq0):
    """Run the multigrid (progressive mode-unfreezing) QA optimization."""

    mg_cfg = MultigridConfig(
        # Stages: first optimise |m|,|n|<=1, then <=2, then <=3
        k_stages=[1, 2, 3],
        # Evaluate QS error on these flux surfaces
        rho_surfaces=[0.6, 0.8, 1.0],
        # Aspect-ratio regularisation
        aspect_ratio_target=8.0,
        aspect_ratio_weight=2.0,
        # Per-stage iteration budget (increase for production)
        maxiter_per_stage=20,
        # Optimizer that re-solves equilibrium at each step
        optimizer_method="proximal-lsq-exact",
        verbose=3,
        # Convergence tolerances
        ftol=1e-4,
        xtol=1e-6,
        gtol=1e-6,
        initial_trust_ratio=1.0,
    )

    eq_mg, eqfam_mg = run_multigrid_qa(eq0, cfg=mg_cfg)

    # Optionally tighten force-balance after optimization
    eq_mg = resolve_equilibrium(eq_mg, verbose=1)

    save_results(eqfam_mg, path="qa_multigrid_output.h5")
    return eq_mg, eqfam_mg


# ===================================================================
# 3.  Constrained QA optimization
# ===================================================================
def demo_constrained(eq0):
    """Run the constrained (all modes free) QA optimization."""

    con_cfg = ConstrainedConfig(
        rho_surfaces=[0.6, 0.8, 1.0],
        aspect_ratio_bounds=(7.0, 9.0),
        elongation_bounds=(0.0, 3.0),
        target_volume=None,       # use the initial equilibrium volume
        target_iota=0.42,         # target average rotational transform
        maxiter=100,
        optimizer_method="proximal-lsq-exact",
        verbose=3,
        ftol=1e-4,
        xtol=1e-6,
        gtol=1e-6,
        initial_trust_ratio=1.0,
    )

    eq_con, eqfam_con = run_constrained_qa(eq0, cfg=con_cfg)

    eq_con = resolve_equilibrium(eq_con, verbose=1)

    save_results(eqfam_con, path="qa_constrained_output.h5")
    return eq_con, eqfam_con


# ===================================================================
# 4.  Plotting helpers (requires matplotlib)
# ===================================================================
def plot_results(eqfam):
    """Produce basic diagnostic plots for the optimization results.

    Generates Boozer-surface and QS-error plots for the initial and final
    equilibria in the family.
    """
    try:
        import matplotlib.pyplot as plt
        from desc.plotting import plot_boozer_surface, plot_boundaries, plot_qs_error
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    eq_init = eqfam[0]
    eq_final = eqfam[-1]

    # -- Boozer |B| contours --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_boozer_surface(eq_init, fieldlines=4, ax=axes[0])
    axes[0].set_title("Initial")
    plot_boozer_surface(eq_final, fieldlines=4, ax=axes[1])
    axes[1].set_title("Optimized")
    fig.suptitle("|B| on boundary in Boozer coordinates")
    fig.tight_layout()
    fig.savefig("qa_boozer_surface.png", dpi=150)
    print("Saved qa_boozer_surface.png")

    # -- QS error profile --
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    plot_qs_error(eq_init, helicity=eq_init.helicity, ax=ax2, label="Initial")
    plot_qs_error(eq_final, helicity=eq_final.helicity, ax=ax2, label="Optimized")
    ax2.legend()
    ax2.set_title("QS error vs. ρ")
    fig2.tight_layout()
    fig2.savefig("qa_qs_error.png", dpi=150)
    print("Saved qa_qs_error.png")

    # -- Boundary comparison --
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    plot_boundaries(eqfam, ax=ax3)
    ax3.set_title("Boundary evolution")
    fig3.tight_layout()
    fig3.savefig("qa_boundaries.png", dpi=150)
    print("Saved qa_boundaries.png")

    plt.show()


# ===================================================================
# CLI entry point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Precise QA stellarator optimization with DESC."
    )
    parser.add_argument(
        "--multigrid",
        action="store_true",
        help="Run the multigrid optimization only.",
    )
    parser.add_argument(
        "--constrained",
        action="store_true",
        help="Run the constrained optimization only.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting.",
    )
    args = parser.parse_args()

    # Default: run both if neither flag is set
    run_mg = args.multigrid or (not args.multigrid and not args.constrained)
    run_con = args.constrained or (not args.multigrid and not args.constrained)

    # --- Solve starting equilibrium ---
    eq0 = make_initial_eq()

    # --- Multigrid ---
    if run_mg:
        print("\n" + "=" * 60)
        print("  MULTIGRID QA OPTIMIZATION")
        print("=" * 60)
        eq_mg, eqfam_mg = demo_multigrid(eq0)
        if not args.no_plot:
            plot_results(eqfam_mg)

    # --- Constrained ---
    if run_con:
        print("\n" + "=" * 60)
        print("  CONSTRAINED QA OPTIMIZATION")
        print("=" * 60)
        eq_con, eqfam_con = demo_constrained(eq0)
        if not args.no_plot:
            plot_results(eqfam_con)


if __name__ == "__main__":
    main()
