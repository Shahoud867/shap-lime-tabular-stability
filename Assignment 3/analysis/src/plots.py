"""Publication-grade matplotlib visualization module for Assignment 3.

Implements a unified IEEE-compliant style:
  - Font family: DejaVu Sans, >= 8.0 pt at 3.5" (single-col) and 7.16" (double-col)
  - White background, no top/right spines, no 3D charts
  - Colorblind-safe Okabe-Ito palette
  - Vector PDF (TrueType fonttype 42) + 400-DPI PNG exports

All plotting functions read empirical results directly from experiment outputs
(zero hardcoded data values).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Colorblind-safe Okabe-Ito palette & IEEE column geometry
# ---------------------------------------------------------------------------

IEEE_SINGLE_COL_IN = 3.50
IEEE_DOUBLE_COL_IN = 7.16
EXPORT_DPI = 400

OKABE_ITO: dict[str, str] = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "teal": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "slate": "#334155",
    "grey": "#64748B",
}

MODEL_VARIANT_STYLE: dict[str, tuple[str, str, str]] = {
    "logit": (OKABE_ITO["blue"], "-", "o"),
    "cslogit": (OKABE_ITO["sky"], "--", "s"),
    "boost": (OKABE_ITO["teal"], "-", "^"),
    "csboost": (OKABE_ITO["vermilion"], "--", "D"),
    "forest": (OKABE_ITO["orange"], "-.", "v"),
    "lr": (OKABE_ITO["blue"], "-", "o"),
    "rf": (OKABE_ITO["orange"], "-", "s"),
    "xgb": (OKABE_ITO["teal"], "-", "^"),
}


def apply_publication_style() -> None:
    """Apply consistent IEEE publication rcParams across all result plots."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 9.5,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#CBD5E1",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CBD5E1",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    extra_dirs: Sequence[Path] | None = None,
    dpi: int = EXPORT_DPI,
) -> tuple[Path, Path]:
    """Save a figure as vector PDF and 400-DPI PNG, and copy to extra_dirs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.04, facecolor="#FFFFFF")
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="#FFFFFF")

    if extra_dirs:
        for dest in extra_dirs:
            dest = Path(dest)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf_path, dest / pdf_path.name)
            shutil.copy2(png_path, dest / png_path.name)

    return pdf_path, png_path


# ---------------------------------------------------------------------------
# Fig 3: Part 1 Replication Curves (CoV & SRA vs. default rate pi)
# ---------------------------------------------------------------------------

def plot_replication_curves(
    rep_df: pd.DataFrame,
    out_dir: Path,
    extra_dirs: Sequence[Path] | None = None,
    stem: str = "fig3_replication_curves",
) -> tuple[Path, Path]:
    """Plot instance-level CoV and SRA (depth 10) vs. training default rate pi.

    Expected columns in rep_df (either pre-aggregated or per-instance):
      - dataset, model (or variant), explainer, pi (in % or [0,1]), cov, sra
      - optional: cov_q25, cov_q75, sra_q25, sra_q75 (computed if per-instance)
    """
    apply_publication_style()
    df = rep_df.copy()
    model_col = "model" if "model" in df.columns else "variant"
    pi_col = "pi" if "pi" in df.columns else "default_rate"
    if df[pi_col].max() <= 1.0:
        df["pi_pct"] = df[pi_col] * 100.0
    else:
        df["pi_pct"] = df[pi_col].astype(float)

    datasets = sorted(df["dataset"].unique())
    explainers = sorted(df["explainer"].unique()) if "explainer" in df.columns else ["all"]
    primary_exp = explainers[0]
    if "explainer" in df.columns and len(explainers) > 1:
        # Show SHAP vs LIME side-by-side or pool across primary datasets in 2x2 grid
        pass

    n_ds = max(1, len(datasets))
    fig, axes = plt.subplots(
        2, n_ds,
        figsize=(IEEE_DOUBLE_COL_IN, 4.25),
        sharex=True,
        squeeze=False,
    )

    for col_idx, ds in enumerate(datasets):
        sub_ds = df[df["dataset"] == ds]
        if "explainer" in sub_ds.columns:
            # Plot SHAP PermutationExplainer as primary and note or facet if both present;
            # if multiple explainers exist, aggregate or show SHAP solid / LIME or mean per model.
            # Prefer SHAP PermutationExplainer if present, else first explainer.
            pref = [e for e in sub_ds["explainer"].unique() if "shap" in str(e).lower() or "perm" in str(e).lower()]
            exp_raw = str(pref[0]) if pref else str(sub_ds["explainer"].iloc[0])
            exp_label = "PermutationSHAP" if "perm" in exp_raw.lower() else exp_raw.upper()
            if pref:
                sub_ds = sub_ds[sub_ds["explainer"] == pref[0]]
        else:
            exp_label = "Explainer"

        for row_idx, (metric, ylabel) in enumerate([
            ("cov", "Median CoV\n(SD / |Mean|)"),
            ("sra", "Median SRA\n(Rank Var., d=10)"),
        ]):
            ax = axes[row_idx, col_idx]
            if metric not in sub_ds.columns:
                ax.text(0.5, 0.5, f"{metric} unavailable", ha="center", va="center", transform=ax.transAxes)
                continue

            for m_name, grp in sub_ds.groupby(model_col):
                m_key = str(m_name).lower()
                color, ls, marker = MODEL_VARIANT_STYLE.get(m_key, (OKABE_ITO["slate"], "-", "o"))
                agg = grp.groupby("pi_pct", as_index=False).agg(
                    med=(metric, "median"),
                    q25=(metric, lambda s: np.nanpercentile(s, 25)),
                    q75=(metric, lambda s: np.nanpercentile(s, 75)),
                ).sort_values("pi_pct")

                ax.plot(
                    agg["pi_pct"], agg["med"],
                    color=color, linestyle=ls, marker=marker,
                    markersize=4.0, linewidth=1.4, label=str(m_name),
                )
                if len(grp) > len(agg):
                    ax.fill_between(agg["pi_pct"], agg["q25"], agg["q75"], color=color, alpha=0.10)

            if row_idx == 0:
                ax.set_title(f"{ds.upper()} ({exp_label})")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 1:
                ax.set_xlabel(r"Training Default Rate $\pi$ (%)")
                ax.set_ylim(bottom=0.0)
                ax.set_xticks(sorted(sub_ds["pi_pct"].unique()))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", ncol=min(5, len(labels)),
            bbox_to_anchor=(0.5, 1.03), frameon=True, handlelength=2.2,
        )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    paths = save_figure(fig, out_dir, stem, extra_dirs=extra_dirs)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Fig 4: Local Explanation Change vs. Explainer Noise Floor (H2)
# ---------------------------------------------------------------------------

def plot_perturbation_vs_floor(
    ext_df: pd.DataFrame,
    out_dir: Path,
    extra_dirs: Sequence[Path] | None = None,
    stem: str = "fig4_perturbation_vs_floor",
) -> tuple[Path, Path]:
    """Plot perturbed local explanation change D_k vs. noise floor q_i across cells (H2).

    Expected columns in ext_df (instance-level or draw-level):
      - dataset, model, explainer
      - delta_e (or d_k / dk_mean): mean perturbed D_k under prediction-preserving draws
      - noise_floor (or q_i / floor_q95): 95th percentile noise floor D_k
    """
    apply_publication_style()
    df = ext_df.copy()
    de_col = next((c for c in ("d_k", "dk", "delta_e", "dk_mean", "D_k") if c in df.columns), None)
    nf_col = next((c for c in ("noise_floor", "q_i", "floor_q95", "floor_mean") if c in df.columns), None)
    if de_col is None or nf_col is None:
        raise ValueError(f"Missing explanation-change or noise-floor columns in {list(df.columns)}")

    df["cell"] = df["model"].astype(str).str.upper() + " + " + df["explainer"].astype(str)
    datasets = sorted(df["dataset"].unique())
    cells = sorted(df["cell"].unique())

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(IEEE_DOUBLE_COL_IN, 3.35),
        sharey=True,
        squeeze=False,
    )

    x_pos = np.arange(len(cells))
    width = 0.35

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        sub = df[df["dataset"] == ds]

        pert_mean, pert_err = [], []
        floor_mean, floor_err = [], []

        for cell in cells:
            c_sub = sub[sub["cell"] == cell]
            if c_sub.empty:
                pert_mean.append(np.nan); pert_err.append(0.0)
                floor_mean.append(np.nan); floor_err.append(0.0)
                continue
            pv = c_sub[de_col].dropna().to_numpy(dtype=float)
            fv = c_sub[nf_col].dropna().to_numpy(dtype=float)
            pm = float(np.mean(pv)) if len(pv) else np.nan
            fm = float(np.mean(fv)) if len(fv) else np.nan
            pe = 1.96 * float(np.std(pv, ddof=1)) / np.sqrt(len(pv)) if len(pv) > 1 else 0.0
            fe = 1.96 * float(np.std(fv, ddof=1)) / np.sqrt(len(fv)) if len(fv) > 1 else 0.0
            pert_mean.append(pm)
            pert_err.append(pe)
            floor_mean.append(fm)
            floor_err.append(fe)

        ax.bar(
            x_pos - width / 2, floor_mean, width=width,
            color=OKABE_ITO["sky"], edgecolor=OKABE_ITO["slate"], linewidth=0.7,
            yerr=floor_err, capsize=2.5, error_kw={"lw": 0.9},
            label=r"Noise Floor $q_i$ (95th pct)",
        )
        ax.bar(
            x_pos + width / 2, pert_mean, width=width,
            color=OKABE_ITO["vermilion"], edgecolor=OKABE_ITO["slate"], linewidth=0.7,
            yerr=pert_err, capsize=2.5, error_kw={"lw": 0.9},
            label=r"Perturbed Mean $D_4$ ($|\Delta p| \leq \tau$)",
        )

        ax.set_title(ds.upper())
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cells, rotation=32, ha="right")
        ax.set_ylim(0.0, 0.95)
        if idx == 0:
            ax.set_ylabel(r"Top-$4$ Jaccard Distance $D_4$ (Mean $\pm$ 95% CI)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 1.04), frameon=True,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    paths = save_figure(fig, out_dir, stem, extra_dirs=extra_dirs)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Fig 5: ROC Curves — F(x) vs. Baselines & Post-Hoc |Delta p| (H3 & H5)
# ---------------------------------------------------------------------------

def plot_roc_comparison(
    inst_df: pd.DataFrame,
    out_dir: Path,
    extra_dirs: Sequence[Path] | None = None,
    stem: str = "fig5_roc_comparison",
) -> tuple[Path, Path]:
    """Plot ROC curves predicting instance instability Y_i across datasets (H3, H5).

    Expected columns in inst_df:
      - dataset, y_instab (binary 0/1)
      - score_f (or f_score): ex-ante fragility F(x)
      - score_margin (or b_margin): negative decision margin -|p(x)-t|
      - optional: score_treevar (or b_var): RF tree prediction variance
      - optional: mean_abs_dp (or abs_dp): post-hoc |Delta p| reference
    """
    apply_publication_style()
    df = inst_df.copy()
    if "explainer" in df.columns:
        non_lin = df[df["explainer"].astype(str) != "LinearSHAP"]
        if not non_lin.empty:
            df = non_lin
    y_col = next((c for c in ("y_instab", "Y_i", "y_i", "unstable_label") if c in df.columns), None)
    if y_col is None:
        raise ValueError(f"Missing binary instability label column in {list(df.columns)}")

    predictors = [
        ("score_f", "F(x) [Proposed]", OKABE_ITO["teal"], "-", 1.8),
        ("score_margin", "Margin Baseline", OKABE_ITO["blue"], "--", 1.35),
        ("score_treevar", "Tree Variance", OKABE_ITO["orange"], "-.", 1.35),
        ("mean_abs_dp", r"Post-hoc $|\Delta p|$ (Ref.)", OKABE_ITO["vermilion"], ":", 1.45),
    ]
    # Flexible column aliases
    aliases = {
        "score_f": ["score_f", "f_score", "F_x", "fragility"],
        "score_margin": ["score_margin", "b_margin", "margin_score"],
        "score_treevar": ["score_treevar", "b_var", "tree_var"],
        "mean_abs_dp": ["mean_abs_dp", "abs_dp", "mean_delta_p"],
    }

    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(IEEE_DOUBLE_COL_IN, 2.95),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        sub = df[df["dataset"] == ds].dropna(subset=[y_col])
        y_true = sub[y_col].astype(int).to_numpy()

        ax.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], lw=0.9, ls="--", alpha=0.6, label="Chance (0.50)")

        for key, label, color, ls, lw in predictors:
            col = next((c for c in aliases[key] if c in sub.columns), None)
            if col is None:
                continue
            valid = sub[[y_col, col]].dropna()
            if valid[y_col].nunique() < 2:
                continue
            yt = valid[y_col].astype(int).to_numpy()
            ys = valid[col].astype(float).to_numpy()
            fpr, tpr, _ = roc_curve(yt, ys)
            auc_val = roc_auc_score(yt, ys)
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, label=f"{label}: {auc_val:.3f}")

        ax.set_title(f"{ds.upper()} (TreeSHAP + LIME)" if "explainer" in sub.columns else ds.upper())
        ax.set_xlabel("False Positive Rate (1 - Spec.)")
        if idx == 0:
            ax.set_ylabel("True Positive Rate (Sens.)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(
            loc="lower right",
            fontsize=7.3 if len(datasets) >= 3 else 8.0,
            handlelength=1.5,
            labelspacing=0.28,
            borderpad=0.35,
            framealpha=0.88,
        )

    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem, extra_dirs=extra_dirs)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Fig 6: Double Dissociation & Discretisation Intervention (H4b & H4c)
# ---------------------------------------------------------------------------

def plot_double_dissociation(
    dissoc_df: pd.DataFrame,
    h4c_df: pd.DataFrame | None,
    out_dir: Path,
    extra_dirs: Sequence[Path] | None = None,
    stem: str = "fig6_double_dissociation",
) -> tuple[Path, Path]:
    """Plot double dissociation coefficients (H4b) and LIME discretisation ablation (H4c).

    Expected columns in dissoc_df:
      - explainer ('TreeSHAP', 'LIME'), predictor ('C_tree', 'C_bin')
      - beta (log-odds or standardized coef), ci_low, ci_high
    Expected columns in h4c_df (optional):
      - dataset, discretisation ('On', 'Off'), instability_rate (or u_bar), ci_low, ci_high
    """
    apply_publication_style()
    has_h4c = h4c_df is not None and not h4c_df.empty
    ncols = 2 if has_h4c else 1
    fig_w = IEEE_DOUBLE_COL_IN if has_h4c else IEEE_SINGLE_COL_IN
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 2.85), squeeze=False)

    # Panel A: H4b Double Dissociation forest/coefficient plot
    ax0 = axes[0, 0]
    df_b = dissoc_df.copy()
    explainers = list(df_b["explainer"].unique())
    predictors = list(df_b["predictor"].unique())

    y_positions = np.arange(len(explainers))
    offsets = np.linspace(-0.16, 0.16, max(1, len(predictors)))
    pred_colors = {
        "C_tree": OKABE_ITO["teal"],
        "c_tree": OKABE_ITO["teal"],
        "C_bin": OKABE_ITO["vermilion"],
        "c_bin": OKABE_ITO["vermilion"],
    }

    pred_labels = {
        "C_tree": r"$C_{\mathrm{tree}}$",
        "c_tree": r"$C_{\mathrm{tree}}$",
        "C_bin": r"$C_{\mathrm{bin}}$",
        "c_bin": r"$C_{\mathrm{bin}}$",
    }

    for p_idx, pred in enumerate(predictors):
        sub = df_b[df_b["predictor"] == pred]
        color = pred_colors.get(pred, OKABE_ITO["blue"])
        marker = "o" if "tree" in pred.lower() else "s"
        for e_idx, exp_name in enumerate(explainers):
            row = sub[sub["explainer"] == exp_name]
            if row.empty:
                continue
            r0 = row.iloc[0]
            est = float(r0["beta"])
            lo = est - float(r0["ci_low"])
            hi = float(r0["ci_high"]) - est
            lbl = pred_labels.get(pred, pred) if e_idx == 0 else None
            ax0.errorbar(
                est, y_positions[e_idx] + offsets[p_idx],
                xerr=[[max(0.0, lo)], [max(0.0, hi)]],
                fmt=marker, color=color, ecolor=color,
                capsize=3.5, markersize=5.5, elinewidth=1.4, label=lbl,
            )

    ax0.axvline(0.0, color=OKABE_ITO["slate"], linestyle="--", linewidth=1.0)
    ax0.set_yticks(y_positions)
    ax0.set_yticklabels(explainers)
    ax0.set_ylim(-0.45, len(explainers) - 0.55)
    ax0.set_xlabel(r"Log-Odds Slope $\beta$ on Reason Flip (95% Cluster CI)")
    ax0.set_title("(a) H4b: Crossing Double Dissociation")
    ax0.legend(loc="best")

    # Panel B: H4c Discretisation On vs Off
    if has_h4c:
        ax1 = axes[0, 1]
        df_c = h4c_df.copy()
        val_col = next((c for c in ("instability_rate", "u_bar", "mean_u", "rc_rate") if c in df_c.columns), None)
        disc_col = next((c for c in ("discretisation", "discretize", "condition") if c in df_c.columns), None)
        datasets = sorted(df_c["dataset"].unique())
        x = np.arange(len(datasets))
        w = 0.34
        for idx_d, state in enumerate(sorted(df_c[disc_col].unique(), reverse=True)):
            sub_s = df_c[df_c[disc_col] == state].set_index("dataset").reindex(datasets)
            vals = sub_s[val_col].to_numpy(dtype=float)
            color = OKABE_ITO["vermilion"] if str(state).lower() in ("on", "true", "1") else OKABE_ITO["teal"]
            yerr = None
            if "ci_low" in sub_s.columns and "ci_high" in sub_s.columns:
                yerr = [
                    np.maximum(0.0, vals - sub_s["ci_low"].to_numpy(dtype=float)),
                    np.maximum(0.0, sub_s["ci_high"].to_numpy(dtype=float) - vals),
                ]
            ax1.bar(
                x + (idx_d - 0.5) * w, vals, width=w,
                color=color, edgecolor=OKABE_ITO["slate"], linewidth=0.7,
                yerr=yerr, capsize=2.5, label=f"Discretisation: {state}",
            )
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.upper() for d in datasets])
        ax1.set_ylabel(r"Mean LIME Instability Rate $\bar{U}_i$")
        ax1.set_title("(b) H4c: Discretisation Intervention")
        ax1.legend(loc="best")

    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem, extra_dirs=extra_dirs)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Fig 7: Robustness — Nested Models M0->M3 & Scale s Sensitivity (R1 & R3)
# ---------------------------------------------------------------------------

def plot_robustness_ablation(
    nested_df: pd.DataFrame,
    scale_df: pd.DataFrame | None,
    out_dir: Path,
    extra_dirs: Sequence[Path] | None = None,
    stem: str = "fig7_robustness_ablation",
) -> tuple[Path, Path]:
    """Plot nested model / F ablation delta-R^2 & delta-AUROC (R1) + scale sensitivity (R3).

    Expected columns in nested_df:
      - variant (e.g., 'M0: Base', 'M1: +|Δp|', 'M2: +C', 'M3: +F(x)', 'Abl: No Cover', ...)
      - delta_r2 (or adj_r2) and optional ci_low, ci_high
      - optional: delta_auroc (or auroc)
    Expected columns in scale_df:
      - s (perturbation scale 0.01, 0.05, 0.10), model_explainer (or cell), instability_rate (or u_bar)
    """
    apply_publication_style()
    has_scale = scale_df is not None and not scale_df.empty
    ncols = 2 if has_scale else 1
    fig_w = IEEE_DOUBLE_COL_IN if has_scale else IEEE_SINGLE_COL_IN
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 3.1), squeeze=False)

    # Panel A: R1 Nested & Ablation
    ax0 = axes[0, 0]
    df_n = nested_df.copy()
    metric_col = next((c for c in ("delta_r2", "adj_r2", "r2_adj", "auroc") if c in df_n.columns), None)
    y_pos = np.arange(len(df_n))
    vals = df_n[metric_col].to_numpy(dtype=float)
    colors = [
        OKABE_ITO["teal"] if ("m3" in str(v).lower() or "+f" in str(v).lower() or "full f" in str(v).lower())
        else (OKABE_ITO["orange"] if "abl" in str(v).lower() or "drop" in str(v).lower() else OKABE_ITO["blue"])
        for v in df_n["variant"]
    ]
    pretty_labels = [
        str(v)
        .replace("M1: +|Delta p|", r"M1: $+|\Delta p|$")
        .replace("Abl: Cover-Weighted F_tree", r"Abl: Cover-Wt $F_{\mathrm{tree}}$")
        .replace("Abl: Drop F_bin (Tree only)", r"Abl: Drop $F_{\mathrm{bin}}$ (Tree)")
        .replace("Abl: Drop F_tree (Bin only)", r"Abl: Drop $F_{\mathrm{tree}}$ (Bin)")
        for v in df_n["variant"]
    ]
    xerr = None
    if "ci_low" in df_n.columns and "ci_high" in df_n.columns:
        xerr = [
            np.maximum(0.0, vals - df_n["ci_low"].to_numpy(dtype=float)),
            np.maximum(0.0, df_n["ci_high"].to_numpy(dtype=float) - vals),
        ]
    ax0.barh(
        y_pos, vals, xerr=xerr, color=colors, edgecolor=OKABE_ITO["slate"],
        linewidth=0.7, capsize=2.5, height=0.62,
    )
    if "category" in df_n.columns and df_n["category"].nunique() > 1:
        cats = df_n["category"].to_numpy()
        for k in range(1, len(cats)):
            if cats[k] != cats[k - 1]:
                ax0.axhline(k - 0.5, color=OKABE_ITO["grey"], linestyle="--", linewidth=0.9)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(pretty_labels)
    ax0.invert_yaxis()
    lbl_map = {
        "delta_r2": r"Base $R^2$ / Incremental $\Delta R^2$ (95% CI)",
        "adj_r2": r"Adjusted $R^2$",
        "r2_adj": r"Adjusted $R^2$",
        "auroc": "AUROC for Instability $Y_i$",
    }
    ax0.set_xlabel(lbl_map.get(metric_col, metric_col))
    ax0.set_title("(a) R1: Nested Models & $F(x)$ Ablation")

    # Panel B: R3 Scale s sensitivity
    if has_scale:
        ax1 = axes[0, 1]
        df_s = scale_df.copy()
        cell_col = "cell" if "cell" in df_s.columns else ("model_explainer" if "model_explainer" in df_s.columns else "explainer")
        val_col = next((c for c in ("instability_rate", "u_bar", "rc_rate", "delta_e") if c in df_s.columns), None)
        markers = ["o", "s", "^", "D", "v", "P"]
        palette_list = [
            OKABE_ITO["blue"], OKABE_ITO["teal"], OKABE_ITO["vermilion"],
            OKABE_ITO["orange"], OKABE_ITO["purple"], OKABE_ITO["sky"],
        ]
        for idx_c, (cname, grp) in enumerate(df_s.groupby(cell_col)):
            g_sorted = grp.sort_values("s")
            ax1.plot(
                g_sorted["s"], g_sorted[val_col],
                color=palette_list[idx_c % len(palette_list)],
                marker=markers[idx_c % len(markers)],
                linewidth=1.4, markersize=4.5, label=str(cname),
            )
        ax1.set_xlabel(r"Perturbation Scale $s$ ($\times \mathrm{SD}_j$)")
        ax1.set_ylabel(r"Mean Instability Rate $\bar{U}_i$")
        ax1.set_xticks(sorted(df_s["s"].unique()))
        ax1.set_title(r"(b) R3: Perturbation Scale Sensitivity")
        ax1.legend(loc="center right", bbox_to_anchor=(0.99, 0.47), fontsize=7.4, ncol=1, labelspacing=0.25)

    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem, extra_dirs=extra_dirs)
    plt.close(fig)
    return paths
