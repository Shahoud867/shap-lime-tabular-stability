# Stability of SHAP and LIME Explanations for Tabular Models

[![License](https://img.shields.io/badge/License-Academic-blue.svg)](LICENSE)
[![PRISMA 2020](https://img.shields.io/badge/PRISMA-2020%20Compliant-brightgreen.svg)](#systematic-literature-review)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

> **Methods in Business Research (MG 3010)**  
> **Topic:** Systematic literature review and empirical evaluation of SHAP and LIME explanation stability, drift, and fragility mechanisms in high-stakes tabular machine learning models.

---

## Executive Summary

In high-stakes tabular classification problems (e.g., credit risk assessment, fraud detection, medical triage), practitioners frequently assume that if a model produces consistent predictions under minor input perturbations, its post-hoc explanations—such as **SHAP** (*SHapley Additive exPlanations*) and **LIME** (*Local Interpretable Model-agnostic Explanations*)—are similarly reliable.

This research project conducts a systematic literature review (following **Kitchenham & Charters** guidelines and **PRISMA 2020** reporting standards) and empirical analysis to:
1. Systematize stability metrics and perturbation protocols.
2. Investigate the phenomenon of **explanation drift / decoupling** (where predictions remain invariant while explanations fluctuate).
3. Identify underlying mechanisms of instability (split thresholds, bin edges, collinearity, sampling variance) and evaluate pre-perturbation fragility scores.

---

## Research Questions & PICOC Framework

### PICOC Formulation
| Element | Description |
|---|---|
| **Population** | Machine-learning models trained on tabular (structured) data, particularly linear models and tree ensembles (Random Forest, XGBoost, LightGBM) in high-stakes settings. |
| **Intervention** | Post-hoc feature attribution methods from the SHAP and LIME families (KernelSHAP, TreeSHAP, LIME, and variants). |
| **Comparison** | Alternative explainers, model families, perturbation/resampling regimes, stabilized explainer variants, or repeated-call baseline noise floors. |
| **Outcome** | Explanation stability, rank correlation, top-$k$ feature overlap, Lipschitz continuity, variance, decoupling/drift, and predictive fragility scores. |
| **Context** | Empirical, methodological, and theoretical studies (2016–2026). |

### Research Questions
* **RQ1 (Metrics & Regimes):** How do studies measure the stability of SHAP and LIME explanations for tabular models, and under which perturbation or resampling regimes?
* **RQ2 (Prediction-Explanation Decoupling):** What evidence shows whether stable or accurate predictions indicate stable SHAP and LIME explanations, and when does that link break?
* **RQ3 (Mechanisms & Fragility Predictors):** Which mechanisms (split thresholds, bin edges, decision boundaries, collinearity, sampling) explain SHAP and LIME instability on tabular models, and can instability be predicted without running perturbations?

---

## Repository Structure

```text
├── Assignment 1/                       # Project Foundation & Proposal
│   ├── Literature_Review_Table.docx    # Preliminary literature matrix
│   └── proposal.pdf                    # Research proposal & methodology design
│
├── Assignment 2/                       # Systematic Literature Review (SLR)
│   ├── Assignment 02 .pdf              # Consolidated Assignment 2 submission
│   ├── Deliverables/
│   │   ├── README.md                   # Detailed deliverables guide & rebuild instructions
│   │   ├── library/                    # BibTeX and RIS study databases
│   │   │   ├── all_screened_records.ris  # 1,381 screened records with decisions
│   │   │   ├── included_studies.bib      # 79 included studies (S01–S79)
│   │   │   └── included_studies.ris
│   │   ├── links/                      # Full-text tracking & DOIs
│   │   │   ├── fulltext_links.md
│   │   │   ├── included_fulltext_links.csv
│   │   │   └── reports_not_retrieved.csv
│   │   ├── report/                     # PRISMA 2020 LaTeX report & generated tables
│   │   │   ├── main.tex / main.pdf     # Primary SLR report document
│   │   │   ├── preamble.tex
│   │   │   ├── evidence_lists.tex
│   │   │   └── tab_*.tex               # Data extraction, search logs, validation tables
│   │   ├── screening/                  # Dual screening and synthesis workbooks
│   │   │   └── MBR_A2_Screening_Workbook.xlsx
│   │   └── search_scripts/             # Automated retrieval, deduplication & synthesis pipeline
│   │       ├── build_master.py         # Harmonizes raw search queries into master dataset
│   │       ├── coding.py               # Qualitative and quantitative study coding
│   │       ├── dedup.py                # Multi-database record deduplication
│   │       ├── digest.py / enrich.py   # Corpus metadata enrichment
│   │       └── data/                   # Search logs, queries, and decision records
│   └── work/                           # Working directory & extracted text artifacts
│       ├── final/                      # Intermediate pipeline JSON mappings
│       ├── ft_pdfs/                    # Full-text research papers (PDFs)
│       └── ft_txt/                     # Extracted plain-text corpus for content analysis
│
└── .gitignore                          # Clean build & environment filters
```

---

## Rebuilding the SLR Pipeline

The systematic review data processing, table generation, and LaTeX outputs can be fully reproduced using the scripts in `Assignment 2/Deliverables/search_scripts/`:

```bash
cd "Assignment 2/Deliverables/search_scripts"

# 1. Run study coding and master dataset generation
python coding.py ../../work
python build_master.py ../../work

# 2. Build LaTeX tables and screening workbook
python make_search_log.py ../../work
python build_tex.py ../../work ../report
python build_workbook.py ../../work ../screening/MBR_A2_Screening_Workbook.xlsx

# 3. Export reference libraries & cross-check
python build_library.py ../../work ..
python crosscheck.py ..
```

To recompile the final PDF report:
```bash
cd "../report"
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

---

## Authors & Citation

* **Shahoud Shahid** (23i-2515) — FAST National University of Computer and Emerging Sciences, Islamabad

