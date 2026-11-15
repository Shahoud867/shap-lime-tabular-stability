# Venue Strategy & Path from Assignment 3 to an A* Submission

**Prepared By:** `lit-strategist`  
**Date:** 2026-11-15

---

## 1. Realistic Target Venues (Tiered Roadmap)

### Tier 1 — A* / Flagship Targets (Post-Course Extension Required)
| Venue | Type | Why It Fits | Main Hurdle & What Must Be Added |
|---|---|---|---|
| **NeurIPS / ICML / ICLR** (Main Track, CORE A*) | ML Conference | Formal Proposition P1, closed-form expected-crossing theory $F(x)$, double dissociation intervention, and benchmark evaluation of explanation fragility. | Requires substantial theoretical depth (formal error bounds linking crossing probability to $\|\phi(x') - \phi(x)\|$), ≥10 diverse tabular benchmarks beyond credit, and broader baselines (GPEC, conformal uncertainty, S-LIME, KernelSHAP). |
| **AAAI / ACM SIGKDD** (Main Track, CORE A*) | AI / Data Mining | Applied ML & high-stakes algorithmic auditing (KDD Applied Data Science or AAAI AI Alignment/XAI tracks); adverse-action case study fits ADS well. | Needs multi-domain tabular benchmarks (credit, healthcare, fraud), full $J=25$ replication + neural/csforest variants, and an open-source pip-installable diagnostic tool. |
| **ACM FAccT** (CORE A / Flagship Fairness & Accountability) | Socio-technical / Accountability | Directly addresses ECOA/Reg B & EU AI Act adverse-action reason-code reliability under realistic borrower noise. | Must integrate subgroup/fairness disparities in $F(x)$ (e.g., whether protected/marginalized groups sit disproportionately near thresholds), which is currently out of scope in A3. |
| **European Journal of Operational Research (EJOR)** (Q1 / Top OR) or **Information Fusion** / **IEEE TPAMI** | Flagship Journal | EJOR is the exact venue of the base paper (Ballegeer et al., 2025). Direct replication-plus-mechanistic-extension of an EJOR article is a natural follow-up submission. | EJOR expects full 4-dataset IDCS coverage (including `csforest`, `csnet`, $J=25$ refits), profit/economic cost quantification of unstable reason codes, and comprehensive sensitivity checks. |

### Tier 2 — Immediate Post-Grading Targets (Realistic with Light Extension)
- **ECML-PKDD Workshops (e.g., XKDD)** or **AAAI/NeurIPS XAI Workshops:** Ideal immediate outlet after grading using our pre-registered empirical design and double-dissociation mechanism.
- **World Conference on Explainable Artificial Intelligence (xAI, Springer CCIS)** or **EXPLAINS:** Specialized XAI conferences where a focused credit-scoring stability + closed-form fragility audit fits squarely within scope.
- **Applied Q1 OR/IS Journals** (*Expert Systems with Applications*, *Decision Support Systems*, *Journal of the Operational Research Society*): Strong fallback if targeting a journal directly rather than a workshop.

---

## 2. What Differentiates an A* Submission from the 8–10 Page Assignment 3 Paper

| Dimension | Assignment 3 Course Submission (8–10 pp. IEEE) | Competitive A* Conference / EJOR Journal Submission |
|---|---|---|
| **Dataset Breadth** | 3 public credit datasets (HMEQ, SGCS, Taiwan Default). | 8–12 public tabular benchmarks spanning credit (incl. GMSC, German, Home Credit, LendingClub), fraud, and medical risk. |
| **Base Replication Fidelity** | $J=10$ (or $J=25$) refits on `logit`, `cslogit`, `boost`, `csboost`, `forest`; `csforest`/`net`/`csnet` omitted. | Full $J=25$ across all 8 model variants (`csforest`, `net`, `csnet` included) + HEC supplementary comparison. |
| **Theoretical Depth** | Proposition P1 (piecewise invariance) + closed-form $F(x) = \mathbb{E}[C]$ by linearity of expectation. | Formal upper/lower probabilistic Lipschitz bounds relating $F(x)$ to expected attribution shift $\mathbb{E}[\|\phi(x')-\phi(x)\|]$, plus optimal smooth-tree/binning remedies. |
| **Competing Baselines** | Decision margin $B_{\text{margin}}$, RF tree variance $B_{\text{var}}$, post-hoc $|\Delta p|$. | Full comparison against Mikriukov et al. (2026) epistemic uncertainty across all model classes, Breeden (2026) conformal intervals, Hill et al. (2024) GPEC, and local Lipschitz sampling. |
| **Explainer Coverage** | LinearSHAP, exact TreeSHAP, PermutationExplainer (bridge), LIME (disc on/off). | Add KernelSHAP, stabilized LIME (S-LIME / AKDE-LIME), and counterfactual-recourse stability under $F(x)$ gating. |
| **Artifact & Reproducibility** | Course analysis zip + `run_demo.py` + pre-registered `analysis_plan.md`. | Standalone Python package (`pip install`), public GitHub + Zenodo DOI, OSF pre-registration timestamp, and interactive model-card audit report. |
| **Compliance Constraint** | Must NOT be posted on arXiv/ResearchGate prior to course grading (Turnitin self-match risk). | Post preprint only after final university grades are released; confirm venue dual-submission / prior-publication policy and disclose AI tool usage per venue rules. |
