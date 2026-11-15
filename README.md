# Stability and Fragility of Local Explanations in Tabular Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests: 51/51 Passing](https://img.shields.io/badge/pytest-51%20passed-success.svg)](#testing--verification)
[![PRISMA 2020](https://img.shields.io/badge/PRISMA-2020%20Compliant-brightgreen.svg)](#research-stages)
[![IEEEtran: Conference](https://img.shields.io/badge/IEEEtran-Track%20C%20Empirical-orange.svg)](#paper--manuscript)
[![Artifact: Reproducible](https://img.shields.io/badge/Artifact-100%25%20Reproducible-success.svg)](#reproducing-the-full-pipeline)

An end-to-end empirical and theoretical research framework investigating the **fragility, drift mechanisms, and ex-ante predictability of local explanations (SHAP and LIME)** in high-stakes tabular machine learning models under prediction-preserving perturbations.

This repository hosts the complete codebase, statistical analysis pipeline, benchmark datasets, reproducible experimental outputs, and IEEE LaTeX manuscript developed across three cumulative research phases for **Methods in Business Research (MG 3010)**, designed to meet the rigorous empirical standards of top-tier AI and data mining venues (*ACM SIGKDD*, *AAAI*, *ACM FAccT*, and *EJOR*).

---

## Table of Contents

- [Overview & Problem Statement](#overview--problem-statement)
- [Key Scientific Discoveries & Novelty](#key-scientific-discoveries--novelty)
- [Repository Structure](#repository-structure)
- [Benchmark Datasets](#benchmark-datasets)
- [System Architecture & Methodological Flow](#system-architecture--methodological-flow)
- [Installation & Environment Setup](#installation--environment-setup)
- [Quick Start: Live Single-Applicant Audit](#quick-start-live-single-applicant-audit)
- [Reproducing the Full Pipeline](#reproducing-the-full-pipeline)
- [Testing & Verification](#testing--verification)
- [Paper & Manuscript](#paper--manuscript)
- [Authors & Citation](#authors--citation)
- [License & Academic Integrity](#license--academic-integrity)

---

## Overview & Problem Statement

In regulated high-stakes classification (e.g., consumer credit scoring under **U.S. ECOA Regulation B** and the **EU AI Act**, clinical triage, and fraud detection), post-hoc feature attribution algorithms—primarily **KernelSHAP / TreeSHAP** and **LIME**—are widely used to generate adverse-action notices and reason codes.

Practitioners and compliance officers routinely make a critical implicit assumption:  
> *If a model's predicted probability is stable under negligible measurement noise ($|\Delta p| \le \tau$), its post-hoc explanation must also be reliable and stable.*

**This project investigates and disproves that assumption.** We demonstrate that predictions and explanations routinely **decouple**: an applicant's decision and probability can remain completely preserved ($|\Delta p| \le 0.05$), yet their top-ranked adverse-action reason codes flip across **$23.5\%$ to $92.3\%$** of draws.

```
Conventional Flawed Assumption:
Perturbation x' ───> Prediction Shift |Δp| ───[ Weak / Decoupled ]───> Explanation Flip D_k

Discovered Causal Mechanism:
Perturbation x' ───> Boundary Crossings C (Tree Splits C_tree, LIME Bins C_bin) ───> Explanation Flip D_k
```

---

## Key Scientific Discoveries & Novelty

1. **Causal Double Dissociation:**  
   Across **$48,547$ credit tree draws** and **$54,194$ non-credit tree draws**, we prove that SHAP and LIME fail via distinct, non-overlapping mechanisms:
   - **LIME instability** is mediated strictly by **quantile-bin boundary crossings ($C_{\mathrm{bin}}$)** ($\mathrm{OR} = 1.361$ to $1.865$, $p < 10^{-8}$) and is invariant to tree split crossings ($p = 0.168$ to $0.641$).
   - **TreeSHAP instability** is driven strictly by **tree-split boundary crossings ($C_{\mathrm{tree}}$)** ($\mathrm{OR} = 1.019$ to $1.091$, $p \le 1.22 \times 10^{-7}$) and is completely unaffected by feature discretisation.
   - Removing LIME discretisation eliminates this artifact, reducing instability by **$83.4\%$** (HMEQ), **$82.1\%$** (SGCS), and **$31.7\text{ pp}$** (Adult).

2. **Explainer Noise-Floor Separation ($E1$):**  
   Under balanced data ($\pi = 30\%$), **$95.8\%–96.2\%$ of LIME across-refit variance is pure Monte Carlo sampling noise**, which completely masks genuine attribution drift under a wide baseline noise floor ($\bar{q}_{95} \in [0.358, 0.722]$). Exact TreeSHAP has zero sampling noise, exposing stark perturbation fragility ($23.5\%$–$92.3\%$ top-4 flips).

3. **Closed-Form Fragility Theory (Propositions P1 & P2):**  
   Rather than running expensive Monte Carlo perturbation loops or training Gaussian Process surrogates, we derive exact closed-form expected-crossing formulas directly from learned model structures under Gaussian noise $\mathcal{N}(0, (s\cdot\mathrm{SD}_j)^2)$:
   - **$F_{\mathrm{tree}}(x; s)$ & $F_{\mathrm{bin}}(x; s)$:** Exact closed-form expectations of split and bin crossings.
   - **$F_{\mathrm{jump}}^{\mathrm{ens}}(x; s)$ (Proposition P2):** Value-jump attribution sensitivity bound, weighting each tree split by its conditional subtree outcome disparity $\Delta_{t,v} = |V(v_L) - V(v_R)|$ and normalized by ensemble prediction spread $\hat{\sigma}_{\mathrm{pred}}$.

4. **Empirical Dominance Against SOTA Baselines ($N=4{,}800$ pooled):**  
   $F_{\mathrm{jump}}^{\mathrm{ens}}$ achieves **$0.7020$** AUROC and **$0.2377$** AURC at **$2.42\text{ ms/instance}$**, substantially outperforming recent SOTA baselines:
   - **GPEC** (Hill et al., *AISTATS 2024*): $0.5189$ AUROC ($36.80\text{ ms}$)
   - **Epistemic UQ** (Mikriukov et al., *2026*): $0.4607$ AUROC ($4.65\text{ ms}$)
   - **Decision Margin Baseline ($B_{\mathrm{margin}}$):** $0.5592$ AUROC
   - Reaches **$95.7\%$** of the post-hoc $|\Delta p|$ oracle bound ($0.7338$) at a **$95\times$ latency reduction** ($2.42\text{ ms}$ vs. $230.83\text{ ms}$).

5. **Scale-Free Leave-One-Dataset-Out (LODO) Gating:**  
   Raw fragility thresholds suffer from cross-dataset scale collapse (e.g., flagging $0\%$ of instances on SGCS). Training-quantile calibration ($F_{\mathrm{cal}}^{\mathrm{quant}}$) eliminates degenerate folds (**$0/6$**), bounds review rates to **$48.6\%–62.9\%$**, and cuts accepted instability risk by **$+20.32\%$ to $+23.53\%$** across unseen test domains.

---

## Repository Structure

```text
├── Assignment 1/                       # Research Proposal & Initial Design (Sep 2026)
│   ├── Literature_Review_Table.docx    # Preliminary literature matrix
│   └── proposal.pdf                    # Research proposal (tau-preservation & noise floors)
│
├── Assignment 2/                       # PRISMA 2020 Systematic Literature Review (Oct–Nov 2026)
│   ├── Deliverables/
│   │   ├── library/                    # RIS and BibTeX databases (79 included studies, S01–S79)
│   │   ├── report/                     # PRISMA LaTeX report & compiled deliverables
│   │   ├── screening/                  # Dual-screening workbook (1,381 screened records)
│   │   └── search_scripts/             # Automated retrieval, deduplication & synthesis code
│   └── outputs/                        # Reconciled submission archives
│
├── Assignment 3/                       # Empirical Research Paper & Implementation (Nov 2026)
│   ├── admin/                          # Research governance, audits & pre-registration plans
│   │   ├── a_star_extension_spec.md    # Proposition P2 spec & SOTA baseline protocols
│   │   ├── AI_COMPLIANCE_AND_REWRITE_GUIDE.md # Academic integrity & Overleaf rewrite guide
│   │   ├── analysis_plan.md            # Pre-registered hypotheses (H1a–H6, E1, R1–R9)
│   │   ├── base_paper_audit.md         # Ballegeer et al. (2025, EJOR Q1) audit & verification
│   │   ├── literature_gap_matrix.md    # 16-study comparative literature gap matrix
│   │   ├── novelty_log.md              # Explicit novelty boundary documentation
│   │   ├── qa_verification_report.md   # 19-item independent QA audit (51/51 tests pass)
│   │   └── venue_strategy.md           # A* conference roadmap (KDD / AAAI / FAccT / EJOR)
│   │
│   ├── analysis/                       # Reproducible Experimental Pipeline
│   │   ├── config.yaml                 # Full experimental configuration (seeds, thresholds, grids)
│   │   ├── config_demo.yaml            # Fast end-to-end demo configuration (~10.9 min)
│   │   ├── requirements.txt            # Strictly pinned dependency manifest
│   │   ├── data/
│   │   │   ├── codebook/               # Feature dictionaries and column descriptions
│   │   │   ├── processed/              # Deterministic, leakage-safe preprocessed datasets
│   │   │   └── raw/                    # Checksums, licenses, and download manifests
│   │   ├── notebooks/
│   │   │   └── 23i-2515_A.ipynb        # Submission notebook (8 code cells, fully executed)
│   │   ├── results/
│   │   │   ├── astar/                  # Phase 2 SOTA baselines & 6-domain experimental outputs
│   │   │   ├── demo/                   # Isolated results produced by config_demo.yaml
│   │   │   ├── figures/                # Publication-grade vector PDFs & 400 DPI PNGs
│   │   │   └── *.csv / *.parquet       # Stage 01–04 experimental data tables
│   │   ├── scripts/
│   │   │   ├── 01_clean.py             # Stage 01: Screening, profiling, and power analysis
│   │   │   ├── 02_replicate.py         # Stage 02: Ballegeer et al. (2025) replication + E1 ANOVA
│   │   │   ├── 03_extend.py            # Stage 03: Fixed-model perturbations, H2–H6, case study
│   │   │   ├── 04_robustness.py        # Stage 04: Robustness ablation suite (R1–R9)
│   │   │   ├── 05_plots.py             # Stage 05: Publication figure generator
│   │   │   ├── 06_a_star_extension.py  # Stage 06: Phase 2 SOTA baselines & 6-domain suite
│   │   │   └── run_demo.py             # CLI for single-applicant audits and demo pipelines
│   │   ├── src/
│   │   │   ├── baselines.py            # S-LIME, GPEC, ST-SHAP, Epistemic UQ, and AXOM
│   │   │   ├── crossings.py            # Tree split and LIME bin boundary crossing counters
│   │   │   ├── data.py                 # Loaders and leakage-free Preprocessor pipeline
│   │   │   ├── explain.py              # LinearSHAP, TreeSHAP, PermutationSHAP, and LIME
│   │   │   ├── fragility.py            # Closed-form F_tree, F_bin, F_jump, and F_cal scorers
│   │   │   ├── idcs.py                 # Instance-Dependent Cost-Sensitive loss and metrics
│   │   │   ├── metrics.py              # Jaccard D_k, CoV, and SRA stability metrics
│   │   │   ├── models.py               # Estimator wrappers (LR, RF, XGB, CS-Logit, CS-XGB)
│   │   │   ├── perturb.py              # Gaussian perturbation engine with tau-filtering
│   │   │   ├── plots.py                # Okabe-Ito colorblind-safe publication styling
│   │   │   └── stats.py                # DeLong tests, cluster bootstrap, AURC, and nested OLS
│   │   └── tests/                      # Comprehensive pytest test suite (51/51 passing)
│   │
│   ├── diagrams/                       # Editable Draw.io XML sources and exported assets
│   │   ├── fig1_conceptual_model.*     # Conceptual framework diagram
│   │   ├── fig2_methodology_flowchart.*# Research methodology flowchart
│   │   └── fig8_case_study.*           # Single-applicant adverse-action reason code audit
│   │
│   └── paper/                          # IEEEtran Conference Paper Package
│       ├── figures/                    # Vector PDF & 400 DPI PNG figures used in paper
│       ├── main.pdf                    # Compiled 13-page IEEE research paper (0 warnings)
│       ├── main.tex                    # Source LaTeX document (Sections I–XI in 10 pages)
│       └── references.bib              # 40 verified citations (100% DOI/URL-backed)
│
└── .gitignore                          # Clean repository filters
```

---

## Benchmark Datasets

The experimental suite benchmarks **6 diverse tabular datasets** spanning finance, marketing, socio-economics, and healthcare. All splits are deterministically preprocessed with zero test-set leakage:

| Dataset | Domain | Samples ($N$) | Features ($D$) | Default / Target Rate | Provenance / DOI |
|---|---|---:|---:|---:|---|
| **HMEQ** | Home Equity Credit | 5,960 | 12 | 19.95% | SAS / Baesens et al. (2016) |
| **SGCS** | South German Credit | 1,000 | 20 | 30.00% | UCI ML Repository (`doi:10.24432/C5X89F`) |
| **Taiwan Default** | Credit Card Default | 30,000 | 23 | 22.12% | OpenML 42477 (`doi:10.24432/C55S3H`) |
| **Bank Marketing** | Telemarketing Deposit | 45,211 | 16 | 11.70% | OpenML 1461 / Moro et al. |
| **Adult Income** | Census Income Prediction | 48,842 | 14 | 23.93% | OpenML 1590 / Kohavi (1996) |
| **Heart Disease** | Clinical Cardiology Risk | 1,190 | 11 | 52.86% | OpenML 43682 |

Exact SHA-256 digests and open-source licensing terms are tracked in `Assignment 3/analysis/data/raw/checksums_and_licenses.json`.

---

## System Architecture & Methodological Flow

The pipeline executes in sequential, compartmentalized stages:

```
[Raw Benchmarks]
       │
       ▼
[01_clean.py] ──> Data screening, IQR outlier detection, skewness & G*Power analysis (n_exact=193)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[02_replicate.py]                 [03_extend.py]
Part 1: Ballegeer et al. (2025)   Part 2: Fixed-model perturbations (tau=0.05, s=0.05)
IDCS vs. Cost-Agnostic Retraining Noise-floor subtraction (K=10 seeds)
E1 Nested Variance Decomposition  Threshold crossings (C_tree, C_bin) & Double Dissociation
       │                                 │
       └────────────────┬────────────────┘
                        │
                        ▼
                 [04_robustness.py]
                 R1: Nested HC3 OLS ablation
                 R2–R9: Scale, tau, LIME sample size, and LODO transfer
                        │
                        ▼
                 [06_a_star_extension.py]
                 Phase 2: 6-Domain SOTA Benchmark (S-LIME, GPEC, ST-SHAP, Epistemic UQ, AXOM)
                 Proposition P2 value-jump F_jump & Quantile-calibrated LODO gating
                        │
                        ▼
                 [05_plots.py] ──> Publication Figures (Vector PDF + 400 DPI PNG)
```

---

## Installation & Environment Setup

### Prerequisites
- **Python 3.10+** (Python 3.11.4 recommended)
- **PowerShell** (Windows) or **Bash** (Linux / macOS)
- **MiKTeX / TeX Live** (optional, for compiling the IEEE LaTeX paper)

### 1. Clone the Repository
```bash
git clone https://github.com/Shahoud867/shap-lime-tabular-stability.git
cd shap-lime-tabular-stability
```

### 2. Set Up Virtual Environment & Dependencies
To guarantee reproducibility and avoid system package contamination, always set `PYTHONNOUSERSITE=1`:

**On Windows (PowerShell):**
```powershell
python -m venv .venv
$env:PYTHONNOUSERSITE = "1"
.\.venv\Scripts\Activate.ps1
pip install -r "Assignment 3/analysis/requirements.txt"
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
export PYTHONNOUSERSITE=1
source .venv/bin/activate
pip install -r "Assignment 3/analysis/requirements.txt"
```

---

## Quick Start: Live Single-Applicant Audit

Run a live adverse-action reason-code stability audit for an individual test applicant (**HMEQ + XGBoost**, Applicant `#369`) in **~1.1 seconds**:

```powershell
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = "Assignment 3/analysis"
python "Assignment 3/analysis/scripts/run_demo.py" --config "Assignment 3/analysis/config.yaml" --dataset hmeq --applicant auto
```

### Sample CLI Output:
```text
================================================================================
EX-ANTE EXPLANATION FRAGILITY AUDIT
================================================================================
Selected Test Applicant:  #369 (Adverse Decision / REJECT)
Model Predicted Risk:     p(x) = 0.9998 (Threshold t = 0.1994)
Ex-Ante Fragility Score:  F_tree(x; s=0.05) = 1.1288
Decision Gate Decision:   FLAGGED FOR MANUAL REVIEW (F_tree >= 0.0254 LODO Cutoff)

Perturbation Invariance Check (R = 25 draws, s = 0.05):
- Preserved Predictions:  20 / 25 (80.0%) within |Δp| <= 0.05
- Predicted Class Flip:   0 / 25 (0.0% flipped - decision strictly preserved)
- Mean Tree Crossings:    1.1720 splits crossed per draw
- Top-4 Reason Flips:     100.0% of preserved draws flipped reasons (VALUE <-> CLAGE)

Conclusion: Reason codes overturned despite 100% decision preservation.
Audit completed in 1.10s.
```

---

## Reproducing the Full Pipeline

### Fast End-to-End Demo Pipeline (~10.9 Minutes)
To verify the entire 5-stage experimental pipeline end-to-end without running heavy multi-hour grids, execute the demo runner (writes isolated outputs to `Assignment 3/analysis/results/demo/`):

```powershell
$env:PYTHONNOUSERSITE = "1"
python "Assignment 3/analysis/scripts/run_demo.py" --run-pipeline-config "Assignment 3/analysis/config_demo.yaml"
```

### Full Multi-Domain SOTA Benchmark Pipeline
To reproduce all tables and figures from scratch:

```powershell
$env:PYTHONNOUSERSITE = "1"
$PY = "python"
$CFG = "Assignment 3/analysis/config.yaml"

# 1. Download/verify benchmark datasets and generate clean splits
& $PY "Assignment 3/analysis/data/download_data.py"

# 2. Stage 01: Screening, profiling, and power analysis
& $PY "Assignment 3/analysis/scripts/01_clean.py" --config $CFG

# 3. Stage 02: Base replication (Ballegeer et al. 2025) + E1 variance decomposition
& $PY -W ignore "Assignment 3/analysis/scripts/02_replicate.py" --config $CFG

# 4. Stage 03: Fixed-model perturbations, H2-H6, and double dissociation
& $PY -W ignore "Assignment 3/analysis/scripts/03_extend.py" --config $CFG

# 5. Stage 04: Robustness ablation suite (R1-R9) and LODO transfer
& $PY -W ignore "Assignment 3/analysis/scripts/04_robustness.py" --config $CFG

# 6. Stage 06: Phase 2 SOTA baselines (S-LIME, GPEC, ST-SHAP, Epistemic UQ, AXOM) across 6 domains
& $PY -W ignore "Assignment 3/analysis/scripts/06_a_star_extension.py" --config $CFG

# 7. Stage 05: Render all publication figures (Vector PDF + 400 DPI PNG)
& $PY "Assignment 3/analysis/scripts/05_plots.py" --config $CFG
```

---

## Testing & Verification

The repository maintains an independent unit test suite covering data loaders, metric definitions, crossing detectors, fragility bounds, statistical tests, and baseline implementations.

Execute the test suite:
```powershell
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = "Assignment 3/analysis"
python -m pytest "Assignment 3/analysis/tests" -v --tb=short
```

**Status:** `51 passed in 17.60s` (exit code `0`).

Key test modules:
- `test_data.py`: Preprocessing, imputation, and zero-leakage split verification.
- `test_crossings_fragility.py`: Tree split extraction, bin crossing counts, and Proposition P1 Monte Carlo validation.
- `test_stats.py`: Synthetic DGP recovery for DeLong AUROC, cluster bootstrap, and paired AURC difference tests.
- `test_phase2_astar.py`: Exact RF/XGB node value jump extraction, Proposition P2 theoretical bounds, quantile calibration scale invariance, and S-LIME / GPEC / ST-SHAP algorithmic fidelity.

---

## Paper & Manuscript

The complete research paper is compiled with `pdflatex` + `bibtex` using the official IEEE conference template:

- **Location:** [`Assignment 3/paper/main.pdf`](Assignment%203/paper/main.pdf)
- **LaTeX Source:** [`Assignment 3/paper/main.tex`](Assignment%203/paper/main.tex)
- **Bibliography:** [`Assignment 3/paper/references.bib`](Assignment%203/paper/references.bib) (40 entries, 100% DOI/URL verified)
- **Format:** `\documentclass[conference]{IEEEtran}`, 13 pages (Sections I–XI in Pages 1–10; References on Page 11; Appendices A–B on Pages 12–13; 0 errors, 0 warnings).

### Recompiling the Paper:
```bash
cd "Assignment 3/paper"
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

---

## Research Progression Across Assignments

This project represents a continuous, phased research trajectory across three course milestones:

| Phase | Deliverable | Scope & Role |
|---|---|---|
| **Assignment 1** | *Research Proposal* | Designed the foundational research questions (RQ1–RQ3), prediction-preserving $\tau$-filtering, and explainer noise-floor framework. |
| **Assignment 2** | *Systematic Literature Review* | Executed a PRISMA 2020 review screening 1,381 records across 5 databases (79 included studies) identifying empirical gaps in fixed-model perturbation audits and crossing mechanisms. |
| **Assignment 3** | *Empirical Study & Extension* | Replicated Ballegeer et al. (2025, *EJOR* Q1), discovered the causal double dissociation, formulated Propositions P1 & P2, and evaluated against 5 SOTA baselines across 6 tabular domains. |

---

## Authors & Citation

- **Shahoud Shahid** (23i-2515) — *FAST National University of Computer and Emerging Sciences, Islamabad*
- **Saif Shahzad** (23i-2634) — *FAST National University of Computer and Emerging Sciences, Islamabad*

### Citation Format:
```bibtex
@inproceedings{shahid2026crossing,
  author    = {Shahid, Shahoud and Shahzad, Saif},
  title     = {Crossing the Threshold: Mechanism and Ex-Ante Prediction of Local Explanation Instability in Credit Scoring},
  booktitle = {Proceedings of Methods in Business Research (Track C: Empirical Study)},
  year      = {2026},
  pages     = {1--13}
}
```

---

## License & Academic Integrity

- **Code & Experiments:** Licensed under the [MIT License](LICENSE).
- **Academic Integrity Notice:** The code, empirical outputs, and mathematical derivations in this repository are verified reference artifacts. In accordance with university academic policies, final submitted paper prose must be authored and typed independently in Overleaf, maintaining Turnitin Similarity and AI writing indicators strictly below 15%.
