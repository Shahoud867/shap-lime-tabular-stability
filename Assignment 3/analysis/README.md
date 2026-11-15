# Crossing Boundaries, Flipping Reasons

Reproducible experimental code, statistical analysis, and figure generation for **"Crossing Boundaries, Flipping Reasons: Discretisation- and Tree-Threshold-Driven Reason-Code Instability Under Prediction-Preserving Perturbations"** (Methods in Business Research, Assignment 3).

**Authors**
- **Shahoud Shahid** (23i-2515)
- **Saif Shahzad** (23i-2634)

---

## 1. Environment Setup

All scripts were developed and executed in a dedicated Python 3.11 virtual environment (`a3_xai`) on Windows 11 with `PYTHONNOUSERSITE=1` to isolate project dependencies from any user-site packages.

```powershell
# PowerShell setup
python -m venv C:\Users\shaho\.venvs\a3_xai
$env:PYTHONNOUSERSITE = "1"
& "C:\Users\shaho\.venvs\a3_xai\Scripts\python.exe" -m pip install -r "Assignment 3/analysis/requirements.txt"
```

Verify the test suite before running the pipeline (from `Assignment 3/analysis/`, or with `PYTHONPATH="Assignment 3/analysis"`):

```powershell
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = "Assignment 3/analysis"
& "C:\Users\shaho\.venvs\a3_xai\Scripts\python.exe" -m pytest "Assignment 3/analysis/tests" -v
```

---

## 2. Datasets, Checksums & Licences

Raw datasets are cached locally under `Assignment 3/analysis/data/raw/` with SHA-256 digests and provenance recorded in `data/raw/checksums_and_licenses.json`:

| Dataset | Source / DOI | Licence | SHA-256 (`data/raw/*.csv`) |
|---|---|---|---|
| **HMEQ** (`hmeq.csv`, 5,960 × 12) | SAS Viya / HMEQ credit default mirror | Public educational benchmark mirror | `aecb99e8e6b3ccf5f3c0f8ee189bbcd6b7b457fccc5f8a61d8c9f1a0b27074cd` |
| **South German Credit (SGCS)** (`south_german_credit.zip`, 1,000 × 20) | UCI ML Repository (`doi:10.24432/C5X89F`) | CC BY 4.0 | `0b40d40eb7321693d559e247a556f88a6cc8df8489c3cb2ae084db7592584551` |
| **Taiwan Default Credit** (`taiwan_default.csv`, 30,000 × 23) | OpenML 42477 / UCI (`doi:10.24432/C55S3H`) | CC BY 4.0 | `febb7438e64468a5fb175ace36bc4f718a1cc11c22555beced7bdedfc6d32a21` |
| **Bank Marketing** (`bank_marketing.csv`, 45,211 × 16) | OpenML 1461 / UCI | CC BY 4.0 | `af53b8521e2e571402dcf7d91309b16de3b3179fe5bfe9909d9737992b118750` |
| **Adult Census Income** (`adult_income.csv`, 48,842 × 14) | OpenML 1590 / UCI | CC BY 4.0 | `d273d956c28eba72241f69dcbb78e37e93862248503a1873fb44b9fc2a6ae00b` |
| **Heart Disease** (`heart_disease.csv`, 1,190 × 11) | OpenML 43682 | CC0: Public Domain | `55c17bcedcd0962cdd1c6e12ffb3791f91788b1b96f8717f0551ade11b2f942a` |

---

## 3. Reproducible Execution Commands

Run all stages from the project root (`Project/`) with `PYTHONNOUSERSITE=1`:

```powershell
$env:PYTHONNOUSERSITE = "1"
$PY = "C:\Users\shaho\.venvs\a3_xai\Scripts\python.exe"

# Stage 01: Data cleaning, screening diagnostics (missingness, >1.5 IQR, skewness), and exact power analysis
& $PY -u "Assignment 3/analysis/scripts/01_clean.py" --config "Assignment 3/analysis/config.yaml"

# Stage 02: Part 1 base-paper replication (Ballegeer, Bogaert & Benoit, 2025) + E1 variance decomposition
& $PY -u -W ignore "Assignment 3/analysis/scripts/02_replicate.py" --config "Assignment 3/analysis/config.yaml"

# Stage 03: Part 2 main extension experiment, ablations (H4c, R6, H6), case study, and H2-H6 stats
& $PY -u -W ignore "Assignment 3/analysis/scripts/03_extend.py" --config "Assignment 3/analysis/config.yaml"

# Stage 04: Robustness & sensitivity suite (R1-R9) and LODO fragility gate evaluation
& $PY -u -W ignore "Assignment 3/analysis/scripts/04_robustness.py" --config "Assignment 3/analysis/config.yaml"

# Stage 05: Publication figures (400 DPI PDF + PNG) and summary tables
& $PY -u "Assignment 3/analysis/scripts/05_plots.py" --config "Assignment 3/analysis/config.yaml"

# Stage 06: Phase 2 A* Extension (6 datasets, F_jump & F_cal LODO, SOTA baselines: S-LIME, GPEC, ST-SHAP, Epistemic UQ, AXOM, non-credit H4b/H4c)
& $PY -u -W ignore "Assignment 3/analysis/scripts/06_a_star_extension.py" --config "Assignment 3/analysis/config.yaml"
```

> **Part 1 Runtime Governance (§12.2):** Under full-grid pilot benchmarking (`149.9` min projected), `scripts/02_replicate.py` automatically applies the pre-planned Cut #3 governance when `J >= 10` and `len(rates) > 5`: five ordered imbalance rates $\pi \in \{1, 5, 10, 20, 30\}\%$ across `J=10` refits, `bg_size=25`, `lime_num_samples=500`, and `permutation_shap_max_evals=150`, while keeping Part 2 (`Stage 03` & `Stage 04`) at exact unreduced budgets (`bg=100`, `lime_num_samples=1000` vs `5000`, `max_evals=500`, `R=25`, `K=10`).

### Live Demo (< 2 Seconds) & End-to-End Demo Pipeline (~12.4 Minutes)

```powershell
# Live adverse-action fragility audit on the HMEQ + XGB case-study applicant (~1.2s)
& $PY -u "Assignment 3/analysis/scripts/run_demo.py" --config "Assignment 3/analysis/config.yaml" --dataset hmeq --applicant auto

# Complete Stages 01-05 end-to-end smoke/demo pipeline (isolated outputs in results/demo/)
& $PY -u -W ignore "Assignment 3/analysis/scripts/run_demo.py" --run-pipeline-config "Assignment 3/analysis/config_demo.yaml"
```

---

## 4. Repository Structure

```text
Assignment 3/analysis/
├── config.yaml                 # Full experimental configuration (seeds, thresholds, hyperparams)
├── config_demo.yaml            # Reduced end-to-end demo pipeline config (~12.4 min across Stages 01-05)
├── requirements.txt            # Pinned Python packages
├── data/
│   ├── raw/                    # Cached raw CSVs, data_codebook.json, checksums_and_licenses.json
│   └── processed/              # Cleaned/verified parquet & metadata outputs from Stage 01
├── src/
│   ├── data.py                 # Loaders, Preprocessor (median/mode imputation, StandardScaler, OneHotEncoder(drop="first"), (0.5, 99.5) clip bounds)
│   ├── models.py               # LR, CS-LR, XGB, CS-XGB, RF estimators, IDCS CV tuning & metrics
│   ├── idcs.py                 # Instance-dependent credit cost model (Bahnsen/Ballegeer)
│   ├── perturb.py              # Gaussian perturbable-feature sampling & prediction-preserving filter
│   ├── explain.py              # LinearSHAP, TreeSHAP, PermutationExplainer, LIME wrappers
│   ├── crossings.py            # Exact tree-split (C_tree) and LIME quantile-bin (C_bin) counters
│   ├── fragility.py            # Closed-form ex-ante F_tree(x), F_bin(x), F_jump(x), and F_cal quantile/rz calibrators
│   ├── baselines.py            # SOTA Phase 2 baselines: S-LIME, GPEC, ST-SHAP, Epistemic UQ, AXOM
│   ├── metrics.py              # CoV, SRA, top-k Jaccard D_k, Spearman D_rho, normalized L1 D_1
│   ├── stats.py                # Power, JT trend, E1 ANOVA, cluster bootstrap mediation/AUROC/AURC, DeLong, HC3 OLS
│   └── plots.py                # Plotting utilities for Figures 1-10
├── scripts/
│   ├── 01_clean.py             # Stage 01 runner
│   ├── 02_replicate.py         # Stage 02 runner
│   ├── 03_extend.py            # Stage 03 runner
│   ├── 04_robustness.py        # Stage 04 runner
│   ├── 05_plots.py             # Stage 05 figure & table generator
│   ├── 06_a_star_extension.py  # Stage 06 Phase 2 A* multi-domain, F_jump, F_cal LODO & SOTA baseline runner
│   └── run_demo.py             # Live applicant audit & end-to-end demo runner
├── notebooks/
│   └── 23i-2515_A.ipynb        # Reproducible notebook mirroring paper sections
├── tests/                      # Pytest suite (core modules + synthetic DGP statistical validation)
└── results/                    # Cached experiment parquets, statistical tables (CSV/JSON), case study
    ├── astar/                  # Phase 2 Stage 06 A* extension tables (Exp A1-A4 & LODO)
    ├── demo/                   # Isolated outputs from config_demo.yaml
    └── figures/                # Generated 400 DPI PDF/PNG plots (Figures 3-10)
```
