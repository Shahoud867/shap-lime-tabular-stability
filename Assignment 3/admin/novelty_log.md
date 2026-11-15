# Novelty Verification & Publication-Status Log

**Check Date:** 2026-11-15  
**Conducted By:** `lit-strategist`

---

## 1. Publication-Status Audit of Preprints (S25, S52, S62, S75, S77)

Checked via **Crossref Title API** and **arXiv API (`export.arxiv.org/api/query`)** on 2026-11-15:

| Corpus ID | Author(s) & Year | Title | arXiv ID | `journal_ref` / Crossref Match | Current Status (2026-11-15) |
|---|---|---|---|---|---|
| **S25** | Pala & García-Cuesta (2024) | Combination of Weak Learners eXplanations to Improve Random Forest eXplicability Robustness | `arXiv:2402.19025v1` | None | **Preprint** (label explicitly in gap table & BibTeX) |
| **S52** | Mikriukov, Montavon & Höhne (2026) | Uncertainty Gating for Cost-Aware Explainable Artificial Intelligence | `arXiv:2603.29915v1` | None | **Preprint** (label explicitly; code `comrados/uq-xai`) |
| **S62** | Tolay (2026) | Beyond Detection Accuracy: Measuring Explanation Cost, Stability, and Utility for Resource-Aware IoT Intrusion Detection | `arXiv:2608.10349v1` | None (`comment`: "Submitted to Inter...") | **Preprint** (label explicitly) |
| **S75** | Thackshanaramana B (2026) | Hypothesis Class Determines Explanation: Why Accurate Models Disagree on Feature Attribution | `arXiv:2603.15821v1` | None (`comment`: "Submitted to TMLR") | **Preprint** (label explicitly) |
| **S77** | Breeden (2026) | How Understanding Forecast Uncertainty Resolves the Explainability Problem in Machine Learning Models | `arXiv:2602.00179v3` | None | **Preprint** (label explicitly) |

---

## 2. Targeted Novelty Re-Check Searches (2026-11-15)

| Search Query | Platform | Key Findings | Impact on Our Novelty Claim |
|---|---|---|---|
| `"split threshold" "SHAP stability" OR "expected crossing" explanation fragility` | Web Search | Confirms threshold-crossing fragility is recognized conceptually (Pala & García-Cuesta 2024; Breeden 2026; smooth/soft trees), but no published study formulates a closed-form expected-crossing count $F_{\text{tree}}(x)$ / $F_{\text{bin}}(x)$ under Gaussian input perturbations. | Confirms we must state *what is not new* (crossings cause instability: S25, S77) while claiming the closed-form $F(x)$ formulation and empirical validation. |
| `"prediction-preserving" "SHAP" "LIME" tabular credit scoring stability` | Web Search | Finds Hasnain et al. (2026, *Applied AI Letters*, software defect CPDP) and Tolay (2026, IoT TreeSHAP only). No peer-reviewed credit-scoring paper combines prediction-preserving $\tau$-filtering for both SHAP and LIME across LR/RF/XGB with noise-floor controls. | Confirms G1 and G2 remain open. |
| `"reason code" stability perturbation credit scoring adverse action` | Web Search | Regulatory literature (ECOA / Reg B Comment 9(b)(2)-1; CFPB Circular 2022-03, note: withdrawn May 2025) highlights adverse-action reason-code risk, without an instance-level ex-ante crossing-fragility gate. | Motivates case study ($k=4$) via Reg B Comment 9(b)(2)-1 directly rather than relying solely on the withdrawn Circular. |
| `6gakwu/credit-scoring-xai` | GitHub API (`README.md`) | M.Sc. project (Jeff Ogakwu, Toronto Metropolitan Univ., pushed 2026-09-22) testing SHAP & LIME stability under small perturbations and similar real applicants on Home Credit Default Risk. | **Non-peer-reviewed student repository; do not cite as literature.** Differs materially: lacks prediction-preserving $\tau$-filter, lacks explainer noise-floor subtraction, lacks threshold/bin crossing mediation & double dissociation, lacks closed-form $F(x)$, and does not replicate Ballegeer et al. (2025) IDCS/imbalance. |

---

## 3. Exact Novelty Boundary (What We Claim vs. What We Acknowledge as Prior Art)

- **Explicitly NOT claimed as novel:**
  1. That split-threshold or decision-boundary crossings drive instability in tree ensembles or discretised LIME (Pala & García-Cuesta, 2024; Khan et al., 2024; Breeden, 2026).
  2. That tree-derived diagnostics or uncertainty proxies can predict explanation instability (Mikriukov et al., 2026; Dubey et al., 2026).
  3. Prediction-preserving $\tau$-filtering alone (Agarwal et al., 2022; Hasnain et al., 2026; Tolay, 2026).
- **Claimed as novel ("to our knowledge", conditional on our 2026-11-15 search):**
  1. Unifying Ballegeer et al.'s (2025) retraining-based IDCS/imbalance framework with a fixed-model, noise-floor-controlled, prediction-preserving perturbation audit for **both** SHAP (LinearSHAP, exact TreeSHAP, PermutationExplainer) and LIME across LR, RF, and XGB on credit-scoring datasets (HMEQ, SGCS, Taiwan Default).
  2. Quantifying split-threshold and LIME bin-edge crossings ($C_{\text{tree}}$, $C_{\text{bin}}$) as the instance-level **mediating mechanism** ($a \cdot b$ with 5,000-resample cluster bootstrap) and testing a **double dissociation** (plus a discretisation-off intervention) separating structural routing effects from explainer stochasticity.
  3. Formulating and validating a perturbation-free, calibration-free **expected-crossing fragility score** $F_{\text{tree}}(x)$ and $F_{\text{bin}}(x)$ (exact expectation of crossing count under Gaussian input noise) against decision-margin and tree-variance uncertainty baselines, embedded in an adverse-action reason-code fragility gate.
