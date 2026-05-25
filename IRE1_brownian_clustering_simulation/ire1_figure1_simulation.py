#!/usr/bin/env python3
"""
IRE1 Brownian Clustering Simulation - Figure 1

Models two conditions:
1. ER-tethered: 2D membrane diffusion with ER corrals -> subdiffusive (alpha < 1)
2. Cytoplasmic: 3D free Brownian motion -> normal diffusion (alpha ~ 1)

The difference in anomalous diffusion exponent alpha between membrane-bound
and cytoplasmic IRE1 drives different clustering outcomes:
- ER: many small persistent clusters (early stress / ER-opto phenotype)
- Cyto: few large clusters within ~1 hour (late stress / Cyto-opto phenotype)

Generates:
    1) figure1_simulation.png  - multi-panel figure
    2) supplementary_video.mp4 - side-by-side animation (or .gif fallback)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
import os
import time as timer

# ============================================================
# SIMULATION ENGINE
# ============================================================

def run_clustering(
    N=150,
    Lx=12.0, Ly=12.0, Lz=8.0,
    D0=0.005,
    dt=1.0,
    total_time=36000,   # 10 hours in seconds
    r0=0.05,            # base merge radius (um)
    condition="ER",     # "ER" or "Cyto"
    corral_size=2.0,    # ER corral side length (um)
    p_hop=0.02,         # base hopping probability across corral boundary
    snapshot_dt=120.0,  # record snapshot every 2 minutes
    seed=42,
):
    """
    Run IRE1 Brownian clustering simulation.

    ER condition: 2D diffusion on a membrane divided into corrals.
        Particles reflect at corral boundaries with probability (1 - p_hop).
        This produces subdiffusive motion (alpha < 1).
    Cyto condition: 3D free Brownian motion (alpha ~ 1).

    Clusters merge irreversibly when their effective radii overlap.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(total_time / dt)
    snap_every = max(1, int(snapshot_dt / dt))
    is_ER = condition == "ER"

    # Initialize positions
    pos = np.zeros((N, 3))
    pos[:, 0] = rng.uniform(0, Lx, N)
    pos[:, 1] = rng.uniform(0, Ly, N)
    if not is_ER:
        pos[:, 2] = rng.uniform(0, Lz, N)

    sizes = np.ones(N, dtype=int)

    # Recording
    snapshots = [dict(time=0.0, positions=pos.copy(), sizes=sizes.copy())]
    count_hist = [N]
    largest_hist = [1]
    time_hist = [0.0]

    t0 = timer.time()
    for step in range(1, n_steps + 1):
        t = step * dt
        nc = len(sizes)

        # --- Diffusion ---
        if is_ER:
            # Membrane: D ~ 1/n (Saffman-Delbruck for large clusters)
            D = D0 / np.maximum(sizes, 1).astype(float)
            std = np.sqrt(2 * D * dt)

            old_pos = pos[:, :2].copy()
            pos[:, 0] += rng.normal(0, std)
            pos[:, 1] += rng.normal(0, std)

            # --- Corral boundary enforcement ---
            for dim in range(2):
                lim = Lx if dim == 0 else Ly
                # Clamp to domain before computing corral index
                new_clamped = np.clip(pos[:, dim], 1e-10, lim - 1e-10)
                old_c = np.floor(old_pos[:, dim] / corral_size).astype(int)
                new_c = np.floor(new_clamped / corral_size).astype(int)
                crossed = old_c != new_c
                n_crossed = crossed.sum()
                if n_crossed > 0:
                    # Size-dependent hop probability: larger clusters
                    # span more corrals, so they cross more easily.
                    r_eff = r0 * np.cbrt(sizes.astype(float))
                    eff_p = np.minimum(1.0, p_hop * (1.0 + 3.0 * r_eff / corral_size))
                    reflects = crossed.copy()
                    reflects[crossed] &= rng.random(n_crossed) >= eff_p[crossed]
                    if reflects.any():
                        going_pos = pos[reflects, dim] > old_pos[reflects, dim]
                        boundary = np.where(
                            going_pos,
                            (old_c[reflects] + 1) * corral_size,
                            old_c[reflects] * corral_size,
                        )
                        pos[reflects, dim] = 2 * boundary - pos[reflects, dim]

            # Domain reflecting boundaries
            for dim, lim in ((0, Lx), (1, Ly)):
                c = pos[:, dim]
                m = c < 0; c[m] = -c[m]
                m = c > lim; c[m] = 2 * lim - c[m]
                pos[:, dim] = c
        else:
            # Cytoplasm: D ~ 1/n^(1/3) (Stokes-Einstein in 3D)
            D = D0 / np.cbrt(np.maximum(sizes, 1).astype(float))
            std = np.sqrt(2 * D * dt)
            for dim in range(3):
                pos[:, dim] += rng.normal(0, std)
            # Reflecting boundaries
            for dim, lim in ((0, Lx), (1, Ly), (2, Lz)):
                c = pos[:, dim]
                m = c < 0; c[m] = -c[m]
                m = c > lim; c[m] = 2 * lim - c[m]
                pos[:, dim] = c

        # --- Merge nearby clusters ---
        if nc > 1:
            if is_ER:
                dmat = cdist(pos[:, :2], pos[:, :2])
            else:
                dmat = cdist(pos, pos)

            r_eff = r0 * np.cbrt(sizes.astype(float))
            merge_thresh = r_eff[:, None] + r_eff[None, :]
            # Upper triangle: pairs where distance < merge threshold
            np.fill_diagonal(dmat, np.inf)
            candidates = np.argwhere((dmat < merge_thresh) & (np.tri(nc, dtype=bool) == False))

            merged = np.zeros(nc, dtype=bool)
            # Sort by distance for deterministic merging (closest first)
            if len(candidates) > 0:
                pair_dists = dmat[candidates[:, 0], candidates[:, 1]]
                order = np.argsort(pair_dists)
                for idx in order:
                    i, j = candidates[idx]
                    if merged[i] or merged[j]:
                        continue
                    total = sizes[i] + sizes[j]
                    pos[i] = (sizes[i] * pos[i] + sizes[j] * pos[j]) / total
                    sizes[i] = total
                    merged[j] = True

            keep = ~merged
            pos = pos[keep]
            sizes = sizes[keep]

        # --- Record ---
        if step % snap_every == 0:
            snapshots.append(dict(time=t, positions=pos.copy(), sizes=sizes.copy()))
            count_hist.append(len(sizes))
            largest_hist.append(sizes.max())
            time_hist.append(t)

        # Progress
        if step % 3600 == 0:
            elapsed = timer.time() - t0
            print(f"  [{condition}] t={t/3600:.1f}h, clusters={len(sizes)}, "
                  f"largest={sizes.max()}, walltime={elapsed:.1f}s")

    print(f"  [{condition}] Done. Final: {len(sizes)} clusters, largest={sizes.max()}")
    return dict(
        snapshots=snapshots,
        count_history=np.array(count_hist),
        largest_history=np.array(largest_hist),
        time_history=np.array(time_hist),
        params=dict(N=N, Lx=Lx, Ly=Ly, Lz=Lz, D0=D0, dt=dt,
                    total_time=total_time, r0=r0, condition=condition,
                    corral_size=corral_size, p_hop=p_hop),
    )


# ============================================================
# MSD ANALYSIS
# ============================================================

def run_msd_tracking(positions, sizes, params, duration=180.0, dt_track=5.0,
                     seed=99):
    """
    Track existing clusters for MSD analysis (no merging).
    Mimics the paper's protocol: image every 5s for ~3 minutes.

    Uses UNWRAPPED coordinates (no domain boundaries) for correct MSD.
    ER corral boundaries are still applied (they represent real confinement).

    Returns list of dicts with keys: size, msd, tau, alpha
    """
    rng = np.random.default_rng(seed)
    is_ER = params["condition"] == "ER"
    D0 = params["D0"]
    Lx, Ly, Lz = params["Lx"], params["Ly"], params["Lz"]
    corral_size = params["corral_size"]
    p_hop = params["p_hop"]
    r0 = params["r0"]

    n_tracks = len(sizes)
    n_frames = int(duration / dt_track) + 1
    # Sub-stepping for accuracy
    dt_sub = 0.5
    sub_steps = max(1, int(dt_track / dt_sub))

    trajectories = np.zeros((n_tracks, n_frames, 3))
    pos = positions.copy()
    trajectories[:, 0] = pos.copy()

    for frame in range(1, n_frames):
        for _ in range(sub_steps):
            if is_ER:
                D = D0 / np.maximum(sizes, 1).astype(float)
                std = np.sqrt(2 * D * dt_sub)
                old_pos = pos[:, :2].copy()
                pos[:, 0] += rng.normal(0, std)
                pos[:, 1] += rng.normal(0, std)
                # Apply CORRAL boundaries (real confinement) but NOT domain boundaries
                for dim in range(2):
                    lim = Lx if dim == 0 else Ly
                    new_cl = np.clip(pos[:, dim], 1e-10, lim - 1e-10)
                    old_c = np.floor(old_pos[:, dim] / corral_size).astype(int)
                    new_c = np.floor(new_cl / corral_size).astype(int)
                    crossed = old_c != new_c
                    nc = crossed.sum()
                    if nc > 0:
                        r_eff = r0 * np.cbrt(sizes.astype(float))
                        eff_p = np.minimum(1.0, p_hop * (1.0 + 3.0 * r_eff / corral_size))
                        reflects = crossed.copy()
                        reflects[crossed] &= rng.random(nc) >= eff_p[crossed]
                        if reflects.any():
                            going_pos = pos[reflects, dim] > old_pos[reflects, dim]
                            boundary = np.where(
                                going_pos,
                                (old_c[reflects] + 1) * corral_size,
                                old_c[reflects] * corral_size,
                            )
                            pos[reflects, dim] = 2 * boundary - pos[reflects, dim]
                # Domain boundaries for ER (reflecting, prevents escape)
                for dim, lim in ((0, Lx), (1, Ly)):
                    c = pos[:, dim]
                    m = c < 0; c[m] = -c[m]
                    m = c > lim; c[m] = 2 * lim - c[m]
                    pos[:, dim] = c
            else:
                # Cytoplasm: FREE diffusion with NO boundaries (unwrapped)
                # This gives correct MSD = 6*D*tau (alpha = 1) without
                # finite-box confinement artifacts.
                D = D0 / np.cbrt(np.maximum(sizes, 1).astype(float))
                std = np.sqrt(2 * D * dt_sub)
                for dim in range(3):
                    pos[:, dim] += rng.normal(0, std)
                # NO boundary reflections for Cyto MSD tracking
        trajectories[:, frame] = pos.copy()

    # Compute MSD and alpha for each track
    ndim = 2 if is_ER else 3
    results = []
    for i in range(n_tracks):
        traj = trajectories[i, :, :ndim]
        n_pts = len(traj)
        # Use up to half the trajectory for MSD lags
        max_lag = max(2, n_pts // 2)
        lags = np.arange(1, max_lag + 1)
        msd = np.zeros(len(lags))
        for k, lag in enumerate(lags):
            displ = traj[lag:] - traj[:-lag]
            msd[k] = np.mean(np.sum(displ**2, axis=1))
        tau = lags * dt_track
        # Fit alpha from log-log (uniform weights to capture confinement)
        log_tau = np.log10(tau)
        log_msd = np.log10(np.maximum(msd, 1e-20))
        coeffs = np.polyfit(log_tau, log_msd, 1)
        alpha = np.clip(coeffs[0], 0.0, 2.0)
        results.append(dict(size=sizes[i], msd=msd, tau=tau, alpha=alpha))
    return results


# ============================================================
# FIGURE GENERATION
# ============================================================

def find_snapshot_at_time(snapshots, target_hours):
    """Find snapshot closest to target time (in hours)."""
    target_s = target_hours * 3600
    best = min(snapshots, key=lambda s: abs(s["time"] - target_s))
    return best


def make_figure(res_er, res_cyto, msd_er_early, msd_er_late,
                msd_cyto_early, msd_cyto_late,
                filename="figure1_simulation.png"):
    """
    Generate a multi-panel figure matching Figure 1 of the paper.

    Layout (3 rows x 3 cols):
        Row 1: [ER early snap] [ER late snap] [Cyto early snap] [Cyto late snap]
        Row 2: [MSD curves ER+Cyto] [alpha boxplot by condition & size]
        Row 3: [Cluster count vs time] [Largest cluster vs time]
    """
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.4,
                  height_ratios=[1, 0.6, 1, 0.9])

    er_snaps = res_er["snapshots"]
    cyto_snaps = res_cyto["snapshots"]

    snap_er_early = find_snapshot_at_time(er_snaps, 1.0)
    snap_er_late = find_snapshot_at_time(er_snaps, 9.5)
    snap_cyto_early = find_snapshot_at_time(cyto_snaps, 1.0)
    snap_cyto_late = find_snapshot_at_time(cyto_snaps, 9.5)

    Lx_er = res_er["params"]["Lx"]
    Ly_er = res_er["params"]["Ly"]
    Lx_cy = res_cyto["params"]["Lx"]
    Ly_cy = res_cyto["params"]["Ly"]

    size_scale = 25.0

    # --- Row 1: Snapshots ---
    snap_data = [
        (snap_er_early, "ER-tethered (1 h)", Lx_er, Ly_er, "Blues"),
        (snap_er_late, "ER-tethered (10 h)", Lx_er, Ly_er, "Blues"),
        (snap_cyto_early, "Cytoplasmic (1 h)", Lx_cy, Ly_cy, "Reds"),
        (snap_cyto_late, "Cytoplasmic (10 h)", Lx_cy, Ly_cy, "Reds"),
    ]
    panel_labels = ["A", "B", "C", "D"]
    for col, (snap, title, lx, ly, cmap_name) in enumerate(snap_data):
        ax = fig.add_subplot(gs[0, col])
        pos = snap["positions"]
        sz = snap["sizes"]
        cmap = plt.get_cmap(cmap_name)
        log_sz = np.log1p(sz)
        max_log = max(np.log1p(res_cyto["largest_history"].max()), 1)
        colors = cmap(0.3 + 0.6 * log_sz / max_log)
        scatter_sz = size_scale * np.cbrt(sz)
        ax.scatter(pos[:, 0], pos[:, 1], s=scatter_sz, c=colors,
                   alpha=0.85, edgecolors="k", linewidths=0.3)
        ax.set_xlim(0, lx)
        ax.set_ylim(0, ly)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"x ($\mu$m)", fontsize=8)
        if col == 0:
            ax.set_ylabel(r"y ($\mu$m)", fontsize=8)
        ax.tick_params(labelsize=7)
        # Panel label
        ax.text(-0.15, 1.05, panel_labels[col], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        # Cluster stats annotation
        ax.text(0.02, 0.98, f"n={len(sz)} clusters\nmax={sz.max()} mol",
                transform=ax.transAxes, fontsize=6.5, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # --- Row 2: Side view (x-z) showing 3D distribution ---
    Lz_cy = res_cyto["params"]["Lz"]
    side_data = [
        (snap_er_early, "ER z-profile (1 h)", Lx_er, Lz_cy, "tab:blue"),
        (snap_er_late, "ER z-profile (10 h)", Lx_er, Lz_cy, "tab:blue"),
        (snap_cyto_early, "Cyto z-profile (1 h)", Lx_cy, Lz_cy, "tab:red"),
        (snap_cyto_late, "Cyto z-profile (10 h)", Lx_cy, Lz_cy, "tab:red"),
    ]
    for col, (snap, title, lx, lz, color) in enumerate(side_data):
        ax = fig.add_subplot(gs[1, col])
        pos = snap["positions"]
        sz = snap["sizes"]
        scatter_sz = size_scale * np.cbrt(sz)
        ax.scatter(pos[:, 0], pos[:, 2], s=scatter_sz, c=color,
                   alpha=0.7, edgecolors="k", linewidths=0.2)
        ax.axhline(0, color="gray", lw=1.5, alpha=0.6, ls="-")
        ax.set_xlim(0, lx)
        ax.set_ylim(-0.3, lz + 0.3)
        ax.set_xlabel(r"x ($\mu$m)", fontsize=7)
        if col == 0:
            ax.set_ylabel(r"z ($\mu$m)", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.text(0.03, 0.95, "ER membrane", fontsize=6, transform=ax.transAxes,
                    va="top", color="gray")

    # --- Row 3 Left: MSD curves ---
    ax_msd = fig.add_subplot(gs[2, :2])
    # Plot MSD curves from late timepoint, use only a subset for clarity
    for msd_list, label, color in [
        (msd_er_late, "ER-tethered", "tab:blue"),
        (msd_cyto_late, "Cytoplasmic", "tab:red"),
    ]:
        # Take unique clusters (every 3rd track from replicates)
        n_unique = len(msd_list) // 3
        for i, m in enumerate(msd_list[:n_unique]):
            lbl = f"{label} (n={n_unique})" if i == 0 else None
            ax_msd.loglog(m["tau"], m["msd"], "-", color=color, alpha=0.35,
                          lw=1.0, label=lbl)
    # Dynamic reference lines based on data range
    all_msd = [m["msd"] for m in msd_er_late + msd_cyto_late]
    if all_msd:
        msd_mid = np.median([ms[0] for ms in all_msd])
    else:
        msd_mid = 0.01
    tau_ref = np.array([5, 50])
    ax_msd.loglog(tau_ref, msd_mid * (tau_ref / 5)**1.0, "--", color="0.4",
                  lw=1.0, label=r"$\alpha=1$ (free)")
    ax_msd.loglog(tau_ref, msd_mid * (tau_ref / 5)**0.7, ":", color="0.4",
                  lw=1.0, label=r"$\alpha=0.7$ (confined)")
    ax_msd.set_xlabel(r"$\tau$ (s)", fontsize=9)
    ax_msd.set_ylabel(r"MSD ($\mu$m$^2$)", fontsize=9)
    ax_msd.set_title("Mean Squared Displacement (4 h)", fontsize=9)
    ax_msd.legend(fontsize=7, frameon=False, loc="upper left")
    ax_msd.tick_params(labelsize=7)
    ax_msd.text(-0.12, 1.05, "E", transform=ax_msd.transAxes,
                fontsize=12, fontweight="bold", va="bottom")

    # --- Row 3 Right: alpha violin/box plot ---
    ax_alpha = fig.add_subplot(gs[2, 2:])

    # Combine early and late MSD data, stratify by condition
    er_alphas = [m["alpha"] for m in msd_er_early + msd_er_late]
    cyto_alphas = [m["alpha"] for m in msd_cyto_early + msd_cyto_late]
    er_sizes_cat = [m["size"] for m in msd_er_early + msd_er_late]
    cyto_sizes_cat = [m["size"] for m in msd_cyto_early + msd_cyto_late]

    # Stratify: use 1.5 um^2 area threshold from paper.
    # Convert molecule count to approximate area: A ~ pi * (r0 * n^(1/3))^2
    # With r0 = 0.05 um, threshold ~3 molecules for 1.5 um^2 equivalent
    size_thresh = 5  # molecules

    groups = {
        r"ER small" + f"\n(<{size_thresh})":
            [a for a, s in zip(er_alphas, er_sizes_cat) if s < size_thresh],
        r"ER large" + f"\n(>{size_thresh})":
            [a for a, s in zip(er_alphas, er_sizes_cat) if s >= size_thresh],
        r"Cyto small" + f"\n(<{size_thresh})":
            [a for a, s in zip(cyto_alphas, cyto_sizes_cat) if s < size_thresh],
        r"Cyto large" + f"\n(>{size_thresh})":
            [a for a, s in zip(cyto_alphas, cyto_sizes_cat) if s >= size_thresh],
    }

    box_data = []
    box_labels = []
    box_colors = []
    color_list = ["#6baed6", "#2171b5", "#fc9272", "#cb181d"]
    for idx, (label, vals) in enumerate(groups.items()):
        if len(vals) > 0:
            box_data.append(vals)
            box_labels.append(label)
            box_colors.append(color_list[idx])

    if len(box_data) > 0:
        bp = ax_alpha.boxplot(box_data, labels=box_labels, patch_artist=True,
                              widths=0.5, showfliers=False,
                              medianprops=dict(color="black", lw=1.5))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        # Overlay swarm-like points
        rng_plot = np.random.default_rng(0)
        for i, vals in enumerate(box_data):
            x = rng_plot.normal(i + 1, 0.06, size=len(vals))
            ax_alpha.scatter(x, vals, alpha=0.4, s=8, color=box_colors[i],
                             edgecolors="0.3", linewidths=0.2, zorder=5)
            # Annotate median above box
            med = np.median(vals)
            q75 = np.percentile(vals, 75)
            iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
            whisker_top = min(q75 + 1.5 * iqr, max(vals))
            ax_alpha.text(i + 1, whisker_top + 0.06, f"{med:.2f}",
                          ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax_alpha.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax_alpha.set_ylabel(r"Diffusion exponent $\alpha$", fontsize=9)
    ax_alpha.set_title(r"Anomalous diffusion exponent $\alpha$", fontsize=9)
    ax_alpha.tick_params(labelsize=7)
    ax_alpha.text(-0.12, 1.05, "F", transform=ax_alpha.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")
    # Force y limits to show median annotations
    ax_alpha.set_ylim(0.2, 1.6)

    # --- Row 4 Left: Cluster count ---
    ax_count = fig.add_subplot(gs[3, :2])
    t_er = res_er["time_history"] / 3600
    t_cy = res_cyto["time_history"] / 3600
    ax_count.plot(t_er, res_er["count_history"], color="tab:blue",
                  label="ER-tethered", lw=1.5)
    ax_count.plot(t_cy, res_cyto["count_history"], color="tab:red",
                  label="Cytoplasmic", lw=1.5)
    ax_count.set_xlabel("Time (hours)", fontsize=9)
    ax_count.set_ylabel("Number of clusters", fontsize=9)
    ax_count.set_title("Cluster count over time", fontsize=9)
    ax_count.legend(fontsize=8, frameon=False)
    ax_count.tick_params(labelsize=7)
    ax_count.text(-0.12, 1.05, "G", transform=ax_count.transAxes,
                  fontsize=12, fontweight="bold", va="bottom")

    # --- Row 4 Right: Largest cluster ---
    ax_max = fig.add_subplot(gs[3, 2:])
    ax_max.plot(t_er, res_er["largest_history"], color="tab:blue",
                label="ER-tethered", lw=1.5)
    ax_max.plot(t_cy, res_cyto["largest_history"], color="tab:red",
                label="Cytoplasmic", lw=1.5)
    ax_max.set_xlabel("Time (hours)", fontsize=9)
    ax_max.set_ylabel("Largest cluster (molecules)", fontsize=9)
    ax_max.set_title("Largest cluster size over time", fontsize=9)
    ax_max.legend(fontsize=8, frameon=False)
    ax_max.tick_params(labelsize=7)
    ax_max.text(-0.12, 1.05, "H", transform=ax_max.transAxes,
                fontsize=12, fontweight="bold", va="bottom")

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    # Also save as PDF
    pdf_name = os.path.splitext(filename)[0] + ".pdf"
    fig.savefig(pdf_name, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {filename}")
    print(f"Saved figure: {pdf_name}")


# ============================================================
# SUPPLEMENTARY FIGURES
# ============================================================

def _save_fig(fig, filename):
    """Save figure as both PNG and PDF."""
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    pdf_name = os.path.splitext(filename)[0] + ".pdf"
    fig.savefig(pdf_name, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename} and {pdf_name}")


def make_supp_csd(res_er, res_cyto, filename="figure_S1_cluster_size_dist.png"):
    """
    Supplementary Figure: Cluster size distributions at multiple timepoints.
    Shows how ER maintains many small clusters while Cyto coalesces.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(r"Supplementary Figure S1: Cluster Size Distributions",
                 fontsize=12, fontweight="bold", y=0.98)

    timepoints_h = [0.5, 2.0, 4.0, 6.0, 8.0, 10.0]

    for col, t_h in enumerate(timepoints_h):
        if col >= 3:
            row_offset = 1
            c = col - 3
        else:
            row_offset = 0
            c = col

        ax = axes[row_offset, c]

        snap_er = find_snapshot_at_time(res_er["snapshots"], t_h)
        snap_cy = find_snapshot_at_time(res_cyto["snapshots"], t_h)

        sz_er = snap_er["sizes"]
        sz_cy = snap_cy["sizes"]

        max_sz = max(sz_er.max(), sz_cy.max(), 10)
        bins = np.arange(0.5, max_sz + 1.5, 1)

        ax.hist(sz_er, bins=bins, alpha=0.6, color="tab:blue", label="ER",
                edgecolor="white", linewidth=0.5, density=False)
        ax.hist(sz_cy, bins=bins, alpha=0.6, color="tab:red", label="Cyto",
                edgecolor="white", linewidth=0.5, density=False)

        ax.set_xlabel("Cluster size (molecules)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(f"t = {t_h:.1f} h", fontsize=9)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(fontsize=7, frameon=False)

        # Annotate stats
        ax.text(0.95, 0.95,
                f"ER: n={len(sz_er)}, max={sz_er.max()}\n"
                f"Cyto: n={len(sz_cy)}, max={sz_cy.max()}",
                transform=ax.transAxes, fontsize=6, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, filename)


def make_supp_alpha_scatter(msd_er_early, msd_er_late,
                            msd_cyto_early, msd_cyto_late,
                            filename="figure_S2_alpha_vs_size.png"):
    """
    Supplementary Figure: Alpha vs cluster size for individual tracks.
    Demonstrates the correlation between cluster size and diffusion exponent.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(r"Supplementary Figure S2: Anomalous Diffusion Exponent $\alpha$ "
                 r"vs Cluster Size",
                 fontsize=11, fontweight="bold", y=1.02)

    # ER condition
    ax = axes[0]
    for msd_list, label, marker, color in [
        (msd_er_early, "1 h", "o", "#6baed6"),
        (msd_er_late, "4 h", "s", "#2171b5"),
    ]:
        sizes = [m["size"] for m in msd_list]
        alphas = [m["alpha"] for m in msd_list]
        ax.scatter(sizes, alphas, s=20, alpha=0.6, color=color, marker=marker,
                   edgecolors="0.3", linewidths=0.3, label=label, zorder=3)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Cluster size (molecules)", fontsize=9)
    ax.set_ylabel(r"$\alpha$", fontsize=10)
    ax.set_title("ER-tethered", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0.2, 1.6)
    ax.tick_params(labelsize=8)
    ax.text(-0.12, 1.05, "A", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

    # Cyto condition
    ax = axes[1]
    for msd_list, label, marker, color in [
        (msd_cyto_early, "1 h", "o", "#fc9272"),
        (msd_cyto_late, "4 h", "s", "#cb181d"),
    ]:
        sizes = [m["size"] for m in msd_list]
        alphas = [m["alpha"] for m in msd_list]
        ax.scatter(sizes, alphas, s=20, alpha=0.6, color=color, marker=marker,
                   edgecolors="0.3", linewidths=0.3, label=label, zorder=3)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Cluster size (molecules)", fontsize=9)
    ax.set_ylabel(r"$\alpha$", fontsize=10)
    ax.set_title("Cytoplasmic", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0.2, 1.6)
    ax.tick_params(labelsize=8)
    ax.text(-0.12, 1.05, "B", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

    fig.tight_layout()
    _save_fig(fig, filename)


def make_supp_msd_fits(msd_er_late, msd_cyto_late,
                       filename="figure_S3_msd_fits.png"):
    """
    Supplementary Figure: Individual MSD curves with power-law fits.
    Shows quality of alpha extraction from log-log linear fits.
    """
    # Show up to 12 representative tracks per condition
    n_show_er = min(12, len(msd_er_late))
    n_show_cy = min(12, len(msd_cyto_late))
    n_cols = 4
    n_rows_er = max(1, int(np.ceil(n_show_er / n_cols)))
    n_rows_cy = max(1, int(np.ceil(n_show_cy / n_cols)))
    n_rows = n_rows_er + n_rows_cy

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    fig.suptitle(r"Supplementary Figure S3: MSD Curves with Power-Law Fits "
                 r"($\langle r^2 \rangle = C \cdot \tau^\alpha$, 4 h timepoint)",
                 fontsize=11, fontweight="bold", y=1.01)

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # ER tracks
    for idx in range(n_show_er):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        m = msd_er_late[idx]
        tau, msd, alpha = m["tau"], m["msd"], m["alpha"]
        ax.loglog(tau, msd, "o-", color="tab:blue", ms=3, lw=1, alpha=0.8)
        # Overlay fit line
        log_tau = np.log10(tau)
        log_msd = np.log10(np.maximum(msd, 1e-20))
        coeffs = np.polyfit(log_tau, log_msd, 1)
        fit_msd = 10 ** np.polyval(coeffs, log_tau)
        ax.loglog(tau, fit_msd, "--", color="0.3", lw=1)
        ax.set_title(f"ER, n={m['size']}, " + r"$\alpha$" + f"={alpha:.2f}",
                     fontsize=7)
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel(r"MSD ($\mu$m$^2$)", fontsize=7)
        if row == n_rows_er - 1:
            ax.set_xlabel(r"$\tau$ (s)", fontsize=7)

    # Hide unused ER axes
    for idx in range(n_show_er, n_rows_er * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    # Cyto tracks
    for idx in range(n_show_cy):
        row = n_rows_er + idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        m = msd_cyto_late[idx]
        tau, msd, alpha = m["tau"], m["msd"], m["alpha"]
        ax.loglog(tau, msd, "o-", color="tab:red", ms=3, lw=1, alpha=0.8)
        log_tau = np.log10(tau)
        log_msd = np.log10(np.maximum(msd, 1e-20))
        coeffs = np.polyfit(log_tau, log_msd, 1)
        fit_msd = 10 ** np.polyval(coeffs, log_tau)
        ax.loglog(tau, fit_msd, "--", color="0.3", lw=1)
        ax.set_title(f"Cyto, n={m['size']}, " + r"$\alpha$" + f"={alpha:.2f}",
                     fontsize=7)
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel(r"MSD ($\mu$m$^2$)", fontsize=7)
        ax.set_xlabel(r"$\tau$ (s)", fontsize=7)

    # Hide unused Cyto axes
    for idx in range(n_show_cy, n_rows_cy * n_cols):
        row = n_rows_er + idx // n_cols
        col = idx % n_cols
        if row < n_rows:
            axes[row, col].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, filename)


def make_supp_diffusion_scaling(res_er, res_cyto,
                                filename="figure_S4_diffusion_scaling.png"):
    """
    Supplementary Figure: Effective diffusion coefficient vs cluster size,
    showing D_ER ~ 1/n (Saffman-Delbruck) and D_Cyto ~ 1/n^{1/3} (Stokes-Einstein).
    Also shows cluster growth kinetics (total mass in top-N clusters over time).
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Supplementary Figure S4: Diffusion Scaling and Cluster Kinetics",
                 fontsize=11, fontweight="bold", y=1.02)

    # --- Panel A: D vs n theoretical curves ---
    ax = axes[0]
    n_vals = np.arange(1, 201)
    D_er = 0.005 / n_vals
    D_cy = 0.15 / np.cbrt(n_vals)
    ax.loglog(n_vals, D_er, "-", color="tab:blue", lw=2, label=r"ER: $D_0/n$")
    ax.loglog(n_vals, D_cy, "-", color="tab:red", lw=2, label=r"Cyto: $D_0/n^{1/3}$")
    ax.set_xlabel("Cluster size n (molecules)", fontsize=9)
    ax.set_ylabel(r"D ($\mu$m$^2$/s)", fontsize=9)
    ax.set_title("Diffusion coefficient scaling", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=7)
    ax.text(-0.15, 1.05, "A", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

    # --- Panel B: Fraction of mass in top-k clusters over time ---
    ax = axes[1]
    for res, label, color in [
        (res_er, "ER", "tab:blue"),
        (res_cyto, "Cyto", "tab:red"),
    ]:
        snaps = res["snapshots"]
        times = [s["time"] / 3600 for s in snaps]
        frac_top1 = []
        frac_top5 = []
        total_mass = res["params"]["N"]
        for s in snaps:
            sz = np.sort(s["sizes"])[::-1]
            frac_top1.append(sz[0] / total_mass)
            frac_top5.append(sz[:min(5, len(sz))].sum() / total_mass)
        ax.plot(times, frac_top1, "-", color=color, lw=1.5,
                label=f"{label} (top 1)")
        ax.plot(times, frac_top5, "--", color=color, lw=1.5, alpha=0.7,
                label=f"{label} (top 5)")
    ax.set_xlabel("Time (hours)", fontsize=9)
    ax.set_ylabel("Fraction of total mass", fontsize=9)
    ax.set_title("Mass concentration in largest clusters", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 1.05)
    ax.text(-0.15, 1.05, "B", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

    # --- Panel C: Mean cluster size over time ---
    ax = axes[2]
    for res, label, color in [
        (res_er, "ER", "tab:blue"),
        (res_cyto, "Cyto", "tab:red"),
    ]:
        snaps = res["snapshots"]
        times = [s["time"] / 3600 for s in snaps]
        mean_sz = [s["sizes"].mean() for s in snaps]
        median_sz = [np.median(s["sizes"]) for s in snaps]
        ax.plot(times, mean_sz, "-", color=color, lw=1.5, label=f"{label} mean")
        ax.plot(times, median_sz, ":", color=color, lw=1.2, alpha=0.7,
                label=f"{label} median")
    ax.set_xlabel("Time (hours)", fontsize=9)
    ax.set_ylabel("Cluster size (molecules)", fontsize=9)
    ax.set_title("Mean & median cluster size", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    ax.text(-0.15, 1.05, "C", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

    fig.tight_layout()
    _save_fig(fig, filename)


def make_supp_spatial(res_er, res_cyto, filename="figure_S5_spatial.png"):
    """
    Supplementary Figure: Spatial analysis of clusters.
    Shows nearest-neighbor distance distributions and spatial organization.
    """
    from scipy.spatial.distance import pdist

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("Supplementary Figure S5: Spatial Organization of Clusters",
                 fontsize=11, fontweight="bold", y=0.99)

    timepoints_h = [1.0, 4.0, 10.0]

    for col, t_h in enumerate(timepoints_h):
        # Top row: nearest-neighbor distance distributions
        ax = axes[0, col]
        snap_er = find_snapshot_at_time(res_er["snapshots"], t_h)
        snap_cy = find_snapshot_at_time(res_cyto["snapshots"], t_h)

        # ER: 2D NND
        pos_er = snap_er["positions"][:, :2]
        if len(pos_er) > 1:
            d_er = cdist(pos_er, pos_er)
            np.fill_diagonal(d_er, np.inf)
            nnd_er = d_er.min(axis=1)
            ax.hist(nnd_er, bins=30, alpha=0.6, color="tab:blue", label="ER",
                    density=True, edgecolor="white", linewidth=0.5)

        # Cyto: 3D NND
        pos_cy = snap_cy["positions"]
        if len(pos_cy) > 1:
            d_cy = cdist(pos_cy, pos_cy)
            np.fill_diagonal(d_cy, np.inf)
            nnd_cy = d_cy.min(axis=1)
            ax.hist(nnd_cy, bins=30, alpha=0.6, color="tab:red", label="Cyto",
                    density=True, edgecolor="white", linewidth=0.5)

        ax.set_xlabel(r"Nearest-neighbor distance ($\mu$m)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"NND at t = {t_h:.0f} h", fontsize=9)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(fontsize=7, frameon=False)
            ax.text(-0.18, 1.05, "A", transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="bottom")

        # Bottom row: cumulative cluster size distribution (CDF)
        ax = axes[1, col]
        sz_er = np.sort(snap_er["sizes"])[::-1]
        sz_cy = np.sort(snap_cy["sizes"])[::-1]

        # Rank-size plot (log-log)
        ax.loglog(np.arange(1, len(sz_er) + 1), sz_er, "o-", color="tab:blue",
                  ms=3, lw=1, alpha=0.8, label="ER")
        ax.loglog(np.arange(1, len(sz_cy) + 1), sz_cy, "s-", color="tab:red",
                  ms=3, lw=1, alpha=0.8, label="Cyto")
        ax.set_xlabel("Rank", fontsize=8)
        ax.set_ylabel("Cluster size (molecules)", fontsize=8)
        ax.set_title(f"Rank-size plot at t = {t_h:.0f} h", fontsize=9)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(fontsize=7, frameon=False)
            ax.text(-0.18, 1.05, "B", transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="bottom")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, filename)


# ============================================================
# VIDEO GENERATION
# ============================================================

def make_video(res_er, res_cyto, filename="supplementary_video", fps=24):
    """
    Side-by-side animation showing 3D nature of both simulations.
    Top row: x-y top-down view
    Bottom row: x-z side view (shows ER at z=0 vs Cyto filling 3D volume)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    snaps_er = res_er["snapshots"]
    snaps_cyto = res_cyto["snapshots"]
    n_frames = min(len(snaps_er), len(snaps_cyto))

    Lx = res_er["params"]["Lx"]
    Ly = res_er["params"]["Ly"]
    Lz = res_cyto["params"]["Lz"]
    size_scale = 30.0

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25,
                  height_ratios=[1.2, 1])
    fig.subplots_adjust(bottom=0.08, top=0.90)

    # Top row: 3D perspective views
    ax_er3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_cy3d = fig.add_subplot(gs[0, 1], projection="3d")

    # Bottom row: x-z side views (shows height distribution)
    ax_er_xz = fig.add_subplot(gs[1, 0])
    ax_cy_xz = fig.add_subplot(gs[1, 1])

    for ax3d in (ax_er3d, ax_cy3d):
        ax3d.set_xlim(0, Lx)
        ax3d.set_ylim(0, Ly)
        ax3d.set_zlim(0, Lz)
        ax3d.set_xlabel(r"x ($\mu$m)", fontsize=8, labelpad=2)
        ax3d.set_ylabel(r"y ($\mu$m)", fontsize=8, labelpad=2)
        ax3d.set_zlabel(r"z ($\mu$m)", fontsize=8, labelpad=2)
        ax3d.tick_params(labelsize=6)
        ax3d.view_init(elev=25, azim=45)
        # Draw ER membrane plane at z=0
        X, Y = np.meshgrid([0, Lx], [0, Ly])
        Z = np.zeros_like(X)
        ax3d.plot_surface(X, Y, Z, alpha=0.15, color="gray")

    for ax_side, lx in [(ax_er_xz, Lx), (ax_cy_xz, Lx)]:
        ax_side.set_xlim(0, lx)
        ax_side.set_ylim(-0.5, Lz)
        ax_side.set_xlabel(r"x ($\mu$m)", fontsize=9)
        ax_side.set_ylabel(r"z ($\mu$m)", fontsize=9)
        ax_side.axhline(0, color="gray", lw=1, alpha=0.5, label="ER membrane")
        ax_side.tick_params(labelsize=7)

    suptitle = fig.suptitle(
        r"IRE1$\alpha$ Brownian Clustering: ER-tethered (2D membrane) vs "
        r"Cytoplasmic (3D volume)",
        fontsize=11, fontweight="bold",
    )
    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=11)

    def update(frame_idx):
        snap_e = snaps_er[frame_idx]
        snap_c = snaps_cyto[frame_idx]
        t_h = snap_e["time"] / 3600

        # ---- ER ----
        pos_e = snap_e["positions"]
        sz_e = snap_e["sizes"]
        s_e = size_scale * np.cbrt(sz_e)

        ax_er3d.cla()
        ax_er3d.set_xlim(0, Lx); ax_er3d.set_ylim(0, Ly); ax_er3d.set_zlim(0, Lz)
        ax_er3d.set_xlabel("x", fontsize=7, labelpad=1)
        ax_er3d.set_ylabel("y", fontsize=7, labelpad=1)
        ax_er3d.set_zlabel("z", fontsize=7, labelpad=1)
        ax_er3d.tick_params(labelsize=5)
        ax_er3d.view_init(elev=25, azim=45 + 0.15 * frame_idx)
        X, Y = np.meshgrid([0, Lx], [0, Ly])
        ax_er3d.plot_surface(X, Y, np.zeros_like(X), alpha=0.12, color="gray")
        ax_er3d.scatter(pos_e[:, 0], pos_e[:, 1], pos_e[:, 2],
                        s=s_e, c="tab:blue", alpha=0.8, edgecolors="k",
                        linewidths=0.2, depthshade=True)
        ax_er3d.set_title(f"ER-tethered (z=0)\n{len(sz_e)} clusters, max={sz_e.max()}",
                          fontsize=8, pad=2)

        ax_er_xz.cla()
        ax_er_xz.set_xlim(0, Lx); ax_er_xz.set_ylim(-0.5, Lz)
        ax_er_xz.axhline(0, color="gray", lw=1, alpha=0.5)
        ax_er_xz.scatter(pos_e[:, 0], pos_e[:, 2], s=s_e, c="tab:blue",
                         alpha=0.8, edgecolors="k", linewidths=0.2)
        ax_er_xz.set_xlabel(r"x ($\mu$m)", fontsize=8)
        ax_er_xz.set_ylabel(r"z ($\mu$m)", fontsize=8)
        ax_er_xz.set_title("Side view (x-z)", fontsize=8)
        ax_er_xz.tick_params(labelsize=6)

        # ---- Cyto ----
        pos_c = snap_c["positions"]
        sz_c = snap_c["sizes"]
        s_c = size_scale * np.cbrt(sz_c)

        ax_cy3d.cla()
        ax_cy3d.set_xlim(0, Lx); ax_cy3d.set_ylim(0, Ly); ax_cy3d.set_zlim(0, Lz)
        ax_cy3d.set_xlabel("x", fontsize=7, labelpad=1)
        ax_cy3d.set_ylabel("y", fontsize=7, labelpad=1)
        ax_cy3d.set_zlabel("z", fontsize=7, labelpad=1)
        ax_cy3d.tick_params(labelsize=5)
        ax_cy3d.view_init(elev=25, azim=45 + 0.15 * frame_idx)
        X, Y = np.meshgrid([0, Lx], [0, Ly])
        ax_cy3d.plot_surface(X, Y, np.zeros_like(X), alpha=0.12, color="gray")
        ax_cy3d.scatter(pos_c[:, 0], pos_c[:, 1], pos_c[:, 2],
                        s=s_c, c="tab:red", alpha=0.8, edgecolors="k",
                        linewidths=0.2, depthshade=True)
        ax_cy3d.set_title(f"Cytoplasmic (3D)\n{len(sz_c)} clusters, max={sz_c.max()}",
                          fontsize=8, pad=2)

        ax_cy_xz.cla()
        ax_cy_xz.set_xlim(0, Lx); ax_cy_xz.set_ylim(-0.5, Lz)
        ax_cy_xz.axhline(0, color="gray", lw=1, alpha=0.5)
        ax_cy_xz.scatter(pos_c[:, 0], pos_c[:, 2], s=s_c, c="tab:red",
                         alpha=0.8, edgecolors="k", linewidths=0.2)
        ax_cy_xz.set_xlabel(r"x ($\mu$m)", fontsize=8)
        ax_cy_xz.set_ylabel(r"z ($\mu$m)", fontsize=8)
        ax_cy_xz.set_title("Side view (x-z)", fontsize=8)
        ax_cy_xz.tick_params(labelsize=6)

        time_text.set_text(f"t = {t_h:.2f} hours")
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )

    base = os.path.splitext(filename)[0]
    mp4_name = base + ".mp4"
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(mp4_name, writer=writer)
        print(f"Saved video: {mp4_name}")
    except Exception as e:
        print(f"FFmpeg failed ({e}), trying GIF...")
        gif_name = base + ".gif"
        try:
            writer = animation.PillowWriter(fps=min(fps, 15))
            anim.save(gif_name, writer=writer)
            print(f"Saved GIF: {gif_name}")
        except Exception as e2:
            print(f"Could not save animation: {e2}")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    OUT_DIR = "/Users/maxzwilson/Documents/iPython/2025 Stochastic IRE1"

    # --- Shared parameters ---
    common = dict(
        N=200,
        Lx=12.0, Ly=12.0, Lz=8.0,
        dt=1.0,
        total_time=36000,   # 10 hours
        r0=0.05,            # 50 nm base merge radius
        snapshot_dt=120.0,  # every 2 min
    )

    # --- ER-tethered simulation ---
    # ER membrane is modeled as corrals (~1.2 um) with very low hopping probability.
    # Within-corral merging is fast, but inter-corral transport is rare,
    # maintaining many small persistent clusters (matching ER-opto-IRE1 phenotype).
    print("Running ER-tethered simulation...")
    res_er = run_clustering(
        **common,
        D0=0.005,           # um^2/s membrane diffusion
        condition="ER",
        corral_size=1.2,    # ER corral size (um) - ~100 corrals in domain
        p_hop=0.0003,       # very rare inter-corral hopping -> subdiffusion
        seed=42,
    )

    # --- Cytoplasmic simulation ---
    # Free 3D Brownian motion -> normal diffusion (alpha ~ 1)
    # Higher D leads to faster encounters and rapid coalescence within hours
    print("\nRunning Cytoplasmic simulation...")
    res_cyto = run_clustering(
        **common,
        D0=0.15,            # um^2/s cytoplasmic diffusion (30x faster)
        condition="Cyto",
        corral_size=1.2,    # unused for Cyto
        p_hop=1.0,          # unused for Cyto
        seed=42,
    )

    # --- MSD Analysis ---
    # Mimic the paper's protocol: track clusters for ~3 min with 5-sec intervals.
    # Run multiple replicates per timepoint for robust alpha estimates.
    # Early = 1 hour, Late = 4 hours (ensures both conditions have clusters).
    print("\nRunning MSD analysis (multiple replicates)...")

    def multi_replicate_msd(snap, params, n_reps=5, **kwargs):
        """Run n_reps independent tracking simulations, return all results."""
        all_results = []
        for rep in range(n_reps):
            res = run_msd_tracking(
                snap["positions"], snap["sizes"], params,
                seed=kwargs.get("base_seed", 100) + rep * 17,
                **{k: v for k, v in kwargs.items() if k != "base_seed"},
            )
            all_results.extend(res)
        return all_results

    snap_er_1h = find_snapshot_at_time(res_er["snapshots"], 1.0)
    snap_cyto_1h = find_snapshot_at_time(res_cyto["snapshots"], 1.0)
    snap_er_4h = find_snapshot_at_time(res_er["snapshots"], 4.0)
    snap_cyto_4h = find_snapshot_at_time(res_cyto["snapshots"], 4.0)

    msd_er_early = multi_replicate_msd(
        snap_er_1h, res_er["params"], n_reps=3,
        duration=180, dt_track=5, base_seed=100,
    )
    msd_cyto_early = multi_replicate_msd(
        snap_cyto_1h, res_cyto["params"], n_reps=3,
        duration=180, dt_track=5, base_seed=200,
    )
    msd_er_late = multi_replicate_msd(
        snap_er_4h, res_er["params"], n_reps=3,
        duration=180, dt_track=5, base_seed=300,
    )
    msd_cyto_late = multi_replicate_msd(
        snap_cyto_4h, res_cyto["params"], n_reps=3,
        duration=180, dt_track=5, base_seed=400,
    )

    print(f"  ER early: {len(msd_er_early)} tracks, "
          f"alpha range: [{min(m['alpha'] for m in msd_er_early):.2f}, "
          f"{max(m['alpha'] for m in msd_er_early):.2f}]")
    print(f"  ER late:  {len(msd_er_late)} tracks, "
          f"alpha range: [{min(m['alpha'] for m in msd_er_late):.2f}, "
          f"{max(m['alpha'] for m in msd_er_late):.2f}]")
    print(f"  Cyto early: {len(msd_cyto_early)} tracks, "
          f"alpha range: [{min(m['alpha'] for m in msd_cyto_early):.2f}, "
          f"{max(m['alpha'] for m in msd_cyto_early):.2f}]")
    print(f"  Cyto late:  {len(msd_cyto_late)} tracks, "
          f"alpha range: [{min(m['alpha'] for m in msd_cyto_late):.2f}, "
          f"{max(m['alpha'] for m in msd_cyto_late):.2f}]")

    # --- Generate Main Figure (PNG + PDF) ---
    print("\nGenerating main figure...")
    make_figure(
        res_er, res_cyto,
        msd_er_early, msd_er_late,
        msd_cyto_early, msd_cyto_late,
        filename=os.path.join(OUT_DIR, "figure1_simulation.png"),
    )

    # --- Generate Supplementary Figures (PNG + PDF) ---
    print("\nGenerating supplementary figures...")

    make_supp_csd(
        res_er, res_cyto,
        filename=os.path.join(OUT_DIR, "figure_S1_cluster_size_dist.png"),
    )

    make_supp_alpha_scatter(
        msd_er_early, msd_er_late,
        msd_cyto_early, msd_cyto_late,
        filename=os.path.join(OUT_DIR, "figure_S2_alpha_vs_size.png"),
    )

    make_supp_msd_fits(
        msd_er_late, msd_cyto_late,
        filename=os.path.join(OUT_DIR, "figure_S3_msd_fits.png"),
    )

    make_supp_diffusion_scaling(
        res_er, res_cyto,
        filename=os.path.join(OUT_DIR, "figure_S4_diffusion_scaling.png"),
    )

    make_supp_spatial(
        res_er, res_cyto,
        filename=os.path.join(OUT_DIR, "figure_S5_spatial.png"),
    )

    # --- Generate Video ---
    print("\nGenerating video...")
    make_video(
        res_er, res_cyto,
        filename=os.path.join(OUT_DIR, "supplementary_video"),
        fps=24,
    )

    print("\nAll done!")
