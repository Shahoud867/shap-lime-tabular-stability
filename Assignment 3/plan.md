# Assignment 3 — Research and Execution Plan

**Project:** Prediction-preserving perturbations and local explanation stability in credit scoring
**Team:** Shahoud Shahid (23i-2515), Saif Shahzad (23i-2634)
**Brief:** `Assignment 3/Assignment_3_Research_Paper.pdf` (11 pages, read in full)
**Deadline:** Saturday 21 November 2026, 11:59 PM — no extension. **Internal submit target: Sat 21 Nov, 6:00 PM.**
**Plan version:** v1.0, 11 November 2026 (Wednesday). Ten working days remain.

---

## 0. How to use this document

**Status labels used throughout**

| Label | Meaning |
|---|---|
| **[FACT]** | Verified from the brief, our files, or a source we actually read (source named). |
| **[DECISION]** | Recommended choice. The team must confirm or change it in the decision log (§4.7). |
| **[ASSUMPTION]** | Working assumption, not yet verified. Each one has a check attached. |
| **[OPEN]** | Unresolved question. Listed in §15 with an owner and a deadline. |

### 0.1 Compliance warning — read before anything else

The brief (p. 9) says: *"AI tools may be used only for grammar checks, LaTeX debugging and code debugging. They must not write sections of the paper, invent references or generate results. Declare any use in an Acknowledgment line."* Plagiarism or AI-generated content above 15% gives **zero marks for the whole group**.

Consequences for how this plan may be used:

1. **This plan was produced with an AI assistant (Claude).** AI-assisted research planning is not on the brief's list of permitted uses. Ask the instructor whether it is acceptable (§15, Q3), and disclose it in the Acknowledgment either way. Treat every [DECISION] here as advice. The team makes and owns each decision.
2. **Do not copy any wording from this plan, the Assignment 2 report, or AI_SESSION_NOTES.md into the paper.** The Assignment 2 report text was AI-drafted, so reusing it is the fastest route to failing the AI check. Write every section from scratch in your own words.
3. **Write the analysis code yourselves.** AI may help debug it. This plan specifies *what* each module must do and how to test it, and deliberately contains no implementation code.
4. **Both members must type in Overleaf.** The brief says the edit history is used to confirm contribution, and the viva marks individuals who cannot explain their part.

---

## 1. Executive summary

| Item | Recommendation | Status |
|---|---|---|
| Paper track | **Track C — Empirical study** (8–10 pp., 25–40 refs, `\documentclass[conference]{IEEEtran}`) | [DECISION] |
| Base paper | **Ballegeer, Bogaert & Benoit (2025), "Evaluating the stability of model explanations in instance-dependent cost-sensitive credit scoring," *European Journal of Operational Research* 326(3):630–640, doi:10.1016/j.ejor.2025.05.039** | [DECISION]; quartile screenshot required |
| Base model replicated | SHAP/LIME **local** explanation instability across retraining = f(model class, cost-sensitivity, class imbalance), measured with CoV and SRA | [DECISION] |
| Extended model | Adds an instance-level pathway: perturbation magnitude → **threshold / bin-edge crossings** (mediator) → explanation change, controlling for the explainer's **noise floor**; plus an ex-ante, perturbation-free **expected-crossing fragility score F(x)** | [DECISION] |
| Datasets | Base-paper public data: **HMEQ**, **South German Credit (SGCS)**. Main ("own") sample: **Taiwan Default of Credit Card Clients**. GMSC optional | [DECISION] + [OPEN] Q1 |
| Models | Logistic regression (LR), random forest (RF), XGBoost (XGB); cost-sensitive variants cslogit/csboost for the replication | [DECISION] |
| Explainers | LinearSHAP / exact TreeSHAP, LIME (tabular, default discretisation), SHAP PermutationExplainer (base paper's choice, for the replication bridge) | [DECISION] |
| Core hypotheses | H1 (replication), H2 (noise floor), H3 (prediction change is a weak indicator), H4 (crossing mechanism), H5 (fragility score beats cheap baselines); H6 exploratory | [DECISION] |
| Case study | Adverse-action reason codes for a rejected home-equity applicant (HMEQ) | [DECISION] |
| Biggest risk | AI-use / similarity rule → zero marks. Second: the cost-sensitive (IDCS) replication becomes a time sink | §14 |
| Publication path | Assignment paper first. Extended version after grading (more datasets, full-fidelity replication, theory) aimed at an XAI-focused workshop or conference. **Acceptance is not assured** | §17 |

---

## 2. What Assignment 3 requires (extracted from the brief)

Every requirement below is mapped to plan sections and owners in the traceability matrix (§16).

### 2.1 Rules for every group (p. 1)
- One publishable-quality paper in IEEE LaTeX on Overleaf, plus analysis files and a live demo. Total 100 marks.
- **Base paper:** a Q1/Q2 *journal* paper published in 2025 or 2026. Conference papers, arXiv, SSRN and other preprints are not accepted. Verify the quartile on SCImago (SJR) or WoS JCR and **keep a screenshot**.
- **Other references:** 2021–2026. Older works only if foundational.
- **Data:** test the model on your own data plus at least one more relevant sample or dataset. Where the base paper's data is public, use it as well.
- **Tools:** script-based tools are preferred. The paper must be in LaTeX on Overleaf with the IEEE template.
- **Integrity:** plagiarism or AI content above 15% = zero marks. No extension.

### 2.2 Common requirements (pp. 2–3)
- **(a) End of the Introduction, in this order:** research gap (2–3 sentences); novelty (one short paragraph, including how it differs from the base paper); 3–5 numbered contributions, each starting with a verb and, where possible, containing a number; a paper-organisation paragraph.
- **(b) Gap-based literature table:** 10–15 studies from 2021–2026, including the base paper. Columns: Ref, Year, Theory/Method, Context/Industry, Sample/Data, Key findings, Limitations/Research gap. Last row = "Proposed 2026, Your study … Gaps addressed: G1, G3". Follow it with a paragraph naming gaps G1, G2, …
- **(c) Figures:** conceptual framework diagram, methodology/research-design flowchart, case-study diagram (where applicable), and results figures comparing base vs. proposed plus robustness. Self-drawn (draw.io/Visio/TikZ/plotting code). **No AI-generated images**. 400 DPI or vector. Text readable at about 8 pt. Consistent style, white background, labelled axes with units, no 3D. Keep editable sources. Every figure numbered, captioned and referenced in the text before it appears.

### 2.3 Track C — Required work (p. 7)
1. **Replicate the base model:** adapt its measurement instrument (cite every scale), use your own data, estimate the same model, report results next to the published ones and explain differences.
2. **Identify gaps:** gap table (gap → evidence → how addressed).
3. **Propose an extended model** that addresses at least one gap. Novelty must be real: a new construct, mediator or moderator, a new theoretical combination, or a new method.
4. **Evaluate base vs. extended** on the main sample and at least one additional sample, using the same criteria plus any justified new ones.
5. **Robustness:** remove or change each added component one at a time (nested model comparison), reported as table + plot. Add at least one alternative specification or subsample check.
6. **Statistical rigour:** justify sample size (e.g., a priori power analysis in G*Power); reliability (α, CR), validity (AVE, HTMT) and common-method-bias checks; 5,000-resample bootstrap CIs for indirect effects; state the significance level.
7. **Ethics:** consent, anonymity and voluntary participation for primary data. Keep the consent form.
8. **Practical implications** and a detailed case study in a real business setting.

### 2.4 Track C — Paper outline (pp. 7–8)
1 Title, Abstract (150–250 words), Index Terms · 2 Introduction · 3 Literature Review and Hypotheses Development (ends with the comparison table) · 4 Theoretical Background (theory + construct-definitions table) · 5 Conceptual Model and Research Design (model diagram, flowchart, numbered H1…, equations, sampling, instrument with source of each scale, data collection) · 6 Base vs. Extended Model Comparison (side-by-side table) · 7 Data and Analysis Setup (sample profile, data screening, software and version, estimation settings, evaluation criteria with formulas and thresholds) · 8 Results and Discussion (measurement model, structural model, 400-DPI plots base vs. extended on every sample, explain why) · 9 Robustness Analysis (table + plot per variant) · 10 Practical Implications and Case Study (scenario diagram + step-by-step walkthrough) · 11 Limitations and Threats to Validity · 12 Conclusion and Future Work · 13 References (IEEE).

**Minimum figures/tables:** conceptual model diagram, research-design flowchart, case-study diagram, numbered hypotheses, model equations, sample profile table, base-vs-extended comparison table, measurement and structural model tables, hypotheses results table, 3+ result plots, robustness table + plot.

### 2.5 Analysis files and format rules (p. 8)
- Justify the tool choice. Analysis configuration table: sample size, response rate, estimation method, missing-data treatment, number of bootstrap resamples, software and version, significance level.
- File structure mirrors the paper: data cleaning → base model → extended model → robustness → plots.
- Fixed random seeds; dependency file; README with run steps and both members' names and roll numbers; anonymised data plus codebook; every script section commented; plots saved at ≥400 DPI as the same files used in the paper.
- **IEEE formatting:** do not change margins, fonts or columns. Figure caption below, table caption above. booktabs tables, no vertical lines, best result in bold. Numbered equations, every symbol defined, notation table if more than 10 symbols. Hypotheses numbered, stated once, summarised in a supported/not-supported table. `\label` + `Fig.~\ref`/`Table~\ref`/`Eq.~\eqref` before first appearance. BibTeX only, IEEE numbered style, every entry checked; no Wikipedia, blogs or Medium. Formal style: past tense for what you did, present tense for what results show. Define acronyms at first use.

### 2.6 Deliverables, marks, checklist (pp. 4, 9–11)
- **Deliverables:** (1) paper PDF `i22-xxxx_REPORT_SECTION.pdf` plus Overleaf source .zip; (2) analysis files `i22-xxxx_SECTION.ipynb` (or .zip) with dependency file, anonymised data + codebook, questionnaire/interview guide (n/a for us), editable diagram sources; (3) plagiarism + AI report (<15%); (4) live demo + viva, all members.
- Names and roll numbers on page 1 of the paper and in the README. Share an Overleaf **view link** with the instructor. CRediT-style contribution paragraph at the end of the paper.
- **Marks:** paper 70 = intro 8 + literature/gap table 10 + novelty & methodology 14 + results & base comparison 14 + robustness 7 + practical implications & case study 6 + discussion/limitations/conclusion 5 + figures & IEEE formatting 6. Analysis files 15 = runs end-to-end 6 + comments/structure 5 + reproducibility 4. Demo 15 = live run 6 + individual viva 6 + time management/clarity 3.
- **19-item submission checklist** (pp. 10–11). Reproduced in §16.2 with owners.

---

## 3. Assessment of where we stand

### 3.1 What exists and what it is worth for Assignment 3

| Asset | Location | Reuse for A3 | Caveat |
|---|---|---|---|
| A1 proposal (RQs, perturbation design, τ-filter, noise-floor idea, metrics) | `Assignment 1/proposal.pdf` | **High** — the extension design grows directly from it | Several A1 choices are revised in §3.3 |
| A2 corpus: 79 included studies with extraction and quality records | `Assignment 2/Deliverables/screening/MBR_A2_Screening_Workbook.xlsx`, `included_studies.bib` | **High** as a reading list and for candidate rows in the gap table | Extraction was AI-drafted and only partly human-verified (`Assignment 2/AI_AUDIT_STATUS.md`). **Re-read every paper you cite** |
| A2 report text | `Assignment 2/Deliverables/report/main.tex` | **None** (do not reuse prose) | AI-drafted, so it would fail the AI check; it may also self-match on Turnitin |
| Gap analysis and novelty check | A2 synthesis §7; this session's reads of S52, S62, S75, S77 (6 Nov) and today's search | **High** for positioning | Novelty must be stated narrowly (§5.1) |
| Base-paper protocol details | Read today from arXiv:2509.01409 (accepted manuscript, HTML v1) | **High** | Stability numbers exist only in Figures 4–7 and the supplementary material; several settings are unstated (§6.2.4) |
| Code | None for the experiment | — | Everything from §7 still has to be written (by the team) |

**[FACT]** Assignment 2 was committed as "Complete Assignment 2 submission" on 7 Nov 2026 (commit e333866). As of 6 Nov, `AI_AUDIT_STATUS.md` records the human dual-screening columns as empty and the student verification as outstanding. **[OPEN]** Whether these were finished before submission. It matters only if A3 describes A2 as human-screened. Do not make that claim unless it is true.

### 3.2 What the literature check established (as of 11 Nov 2026)

- **[FACT]** The base paper induces instability only by **retraining** on resampled data (25 iterations × 8 default rates). It measures it per test instance with CoV and SRA. It **does not** perturb inputs, **does not** separate explainer sampling noise from model variation (its SHAP is the stochastic `PermutationExplainer`), and offers **no mechanism** for instability. Its stated future work includes "more stable XAI" and controlled synthetic data. (Source: arXiv:2509.01409, §§3.5–3.8 and 6, read via HTML.)
- **[FACT]** The idea that crossing splits drives instability is **not new**. S25 (Pala & García-Cuesta, preprint 2024) shows a single split crossing makes the RF SHAP Lipschitz ratio diverge. S77 (Breeden, preprint 2026) argues boundary crossings drive instability in piecewise models. S23 (Khan et al., AISTATS 2024) notes that discretised tabular LIME is not "astute". S43 (preprint 2026) has a split-count diagnostic for collinearity.
- **[FACT]** Cheap instability proxies exist, but they rest on different signals. S52 (preprint 2026) uses epistemic uncertainty from tree variance and validates it with correlation and precision/recall, not AUROC. S75 (preprint 2026) uses an ensemble-agreement score that needs several models. S77 uses conformal uncertainty, which needs a calibration set. S62 (preprint 2026) applies prediction-preserving perturbations, but only with TreeSHAP, without LIME and without a mechanism.
- **[FACT]** Today's two web searches found no work that combines prediction-preserving perturbation with noise floors and split/bin-edge crossing probability for SHAP *and* LIME in credit scoring. A GitHub project (`6gakwu/credit-scoring-xai`) evaluates SHAP/LIME stability under small perturbations on Home Credit data. It is not peer-reviewed. Check it before submission and do not cite it as literature.
- **Conclusion:** the open gap is the **combination** of these elements, an **ex-ante expected-crossing score**, and the **link to the base paper's factors**. It is not the observation that crossings matter. Every novelty claim must say "to our knowledge" and name the search evidence.

### 3.3 Critical review of earlier decisions (what changes from A1/A2)

| Earlier choice | Problem | Revision |
|---|---|---|
| A1 primary dataset: ULB credit-card fraud (PCA-anonymised, 0.17% positives) | Anonymised features make a reason-code case study meaningless. It is not the base paper's data. The extreme imbalance makes instance sampling awkward | Use the base paper's public credit data (HMEQ, SGCS) plus Taiwan Default (named features). ULB is out of scope |
| A1 RQ2: correlate prediction change \|Δp\| with explanation change **within** the τ-filtered set | **Range restriction.** Filtering to \|Δp\| ≤ 0.05 truncates \|Δp\|, which biases any correlation toward zero and makes "weak link" look stronger than it is | Test H3 on the **full** perturbation set (including non-preserving draws) and within the filtered set, and report both. Primary metric: AUROC of \|Δp\| for detecting reason-code changes |
| A1: "no composite reliability score" | Still correct for weighted mixtures of metrics. F(x) is different: an expected count derived from model structure, with **no fitted weights** | Keep the A1 principle. State explicitly that F(x) is not a weighted composite |
| A1 perturbation types P1–P3, faithfulness axis, cross-method agreement, PFI | Too much for 10 days. Not needed for the hypotheses | Keep P1 (all perturbable continuous features) as primary and P3 (low-importance-only) as an optional robustness check. Drop P2, faithfulness, PFI and agreement |
| Noise floor tested by Wilcoxon for all explainers | Exact LinearSHAP/TreeSHAP are deterministic, so the floor is exactly 0 and the test is degenerate | For exact explainers, report the share of perturbations with any change (Clopper–Pearson CI). Use the Wilcoxon test only for stochastic explainers |
| Implicit plan to claim "tree-derived diagnostics are new" | False (S43; also S52's tree variance) | Claim only the expected-crossing formulation and its validation against baselines |
| Using our own perturbation regime only | Track C requires replicating the base model first | Two-part design: replication (Part 1) then extension (Part 2) |

### 3.4 Why Track C, and not A or B

| Track | Fit | Decisive issues |
|---|---|---|
| A — SLR | Reuses A2 most | Needs **two independent human screeners with kappa** (not done in A2), a **pilot of 5–8 expert interviews or ≥30 survey respondents** (primary data with consent), bibliometrics, and 10–14 pp. Reusing A2 text breaks the AI rule. Produces no empirical contribution for the publication goal |
| B — Survey | Partial | 60–100 references, an original taxonomy that must beat existing reviews, a comparative experiment *and* an extension, 12–16 pp. Too much writing for 10 days |
| **C — Empirical** | **Best** | Matches the gap and the publication goal. Shortest paper (8–10 pp., 25–40 refs). **Main friction:** its SEM/questionnaire criteria (α, CR, AVE, HTMT, CMB, consent) do not apply to a computational study. Handled by mapping to analogues (§6.6) and asking the instructor (§15, Q2) |

---

## 4. Strategic decisions

### 4.1 Base paper — [DECISION]
**Ballegeer et al. (2025), EJOR 326(3):630–640.** Reasons:
- **[FACT]** It is a 2025 journal article (Crossref-verified in A2). EJOR is a long-established operations-research journal. **[ASSUMPTION]** It is ranked Q1 in SJR. **Check:** screenshot SCImago today and save it to `admin/base_paper_quartile.png`.
- Same model families (LR, RF, XGB), both explainers (SHAP, LIME), a business setting (credit scoring), four public datasets, and a measurement instrument we can replicate (CoV, SRA).
- Its own limitations (retraining-only instability, explainer noise not separated, no mechanism) are exactly our gaps G1–G4.
- Reading copy: arXiv:2509.01409 (accepted manuscript). Cite the EJOR version. **Get the published PDF and the supplementary material through the HEC Digital Library.** The supplementary material holds the per-dataset results and variable lists we need.

**Backups, if the quartile check fails or the instructor rejects it** (quartiles not yet verified):
1. Lin & Wang (2025), *Risks* 13(12):238 — SHAP stability across 100 seed-varied XGBoost models on Taiwan credit data.
2. Qi, Chen & Si (2026), *Mathematics* 14:3283 — explanation-stability optimisation on public HR-attrition data.

Switching base paper after 13 Nov costs about two days. Decide by **Thu 12 Nov**.

### 4.2 Datasets — [DECISION]

| Role | Dataset | Size (rows × features), positives | Why | Availability |
|---|---|---|---|---|
| Base-paper sample 1 | **HMEQ** (home equity loans) | 5,960 × 12, 19.95% | In the base paper. Has a loan-amount column (`LOAN`) for costs. Readable features for the case study | Public mirrors exist. **[OPEN]** licence: ship a download script, not the file, unless redistribution is allowed |
| Base-paper sample 2 | **South German Credit** (UCI, corrected 2019 version) | 1,000 × 20, 30% | In the base paper. Has credit amount. Fast | UCI, DOI 10.24432/C5X89F. **[ASSUMPTION]** CC BY 4.0 (check the UCI page) |
| Main / "own" sample | **Taiwan Default of Credit Card Clients** (UCI) | 30,000 × 23, 22.1% | Not in the base paper (an independent sample). Named features. Credit-limit column available as an exposure proxy. Used by S34 (Lin & Wang 2025) | UCI. **[ASSUMPTION]** CC BY 4.0 |
| Optional | GMSC (Kaggle) | 112,915 × 10, 6.74% | In the base paper | Kaggle competition terms; no loan-amount column (base paper's cost source unstated) |
| Excluded | VUB credit scoring | 18,667 × 20 | In the base paper | **[OPEN]** No public URL in the base paper. Skip unless the Petrides et al. (2022) data statement gives one |

**[OPEN] Q1:** whether public, secondary data that we prepare and experiment on counts as "your own data". Our *analysed* data, the explanation-stability measurements, is generated by our experiments. The underlying feature data is public. If the instructor insists on primary data, see the contingency in §14 (R7).

### 4.3 Models and explainers — [DECISION]
- **Part 1 (replication):** logit, cslogit, boost (XGB), csboost; forest (RF) traditional only. csforest and the neural net and csnet are dropped: they are out of scope, and the A1 rationale excludes neural models. Record this as a deviation.
- **Part 2 (extension):** LR, RF, XGB trained at the natural default rate. Add cslogit/csboost only for exploratory H6.
- **Explainers:** LinearSHAP (LR, exact), TreeSHAP (RF/XGB, exact, interventional with a fixed 100-row background), LIME tabular with default quartile discretisation. LIME without discretisation runs only as an ablation (H4c). SHAP `PermutationExplainer` runs only in Part 1 and in the noise-floor bridge (E1), matching the base paper.

### 4.4 Scope boundaries — [DECISION]
In scope: binary credit-default classification on tabular data; three model families; two explainer families; Gaussian measurement-noise perturbations on continuous features.
Out of scope: neural models, gradient explainers, images and text, distribution shift, faithfulness metrics, adversarial attacks, fairness auditing (mention only as a limitation).

### 4.5 Team roles — [DECISION] (change if needed)
- **Saif — Part 1 lead:** replication pipeline, IDCS models, H1, E1, the nested-model robustness analysis; writes Literature Review + gap table, Base-vs-Extended comparison, replication results.
- **Shahoud — Part 2 lead:** perturbation engine, explainers, crossings, F(x), H2–H5; writes Theoretical Background, Conceptual Model & Design, equations and notation, extension results.
- **Joint:** Introduction, case study, discussion, limitations, diagrams, README, demo. Each member must be able to explain the **whole** pipeline at the viva (a teach-back session is scheduled on 19 Nov).

### 4.6 Analysis tools — [DECISION]
Python 3.11; pandas, numpy, scikit-learn, xgboost, shap, lime, scipy, statsmodels, matplotlib. G*Power 3.1 for the a-priori power analysis (screenshot). draw.io for diagrams (`.drawio` sources kept). Pin exact versions in `requirements.txt` on day 1. **Why script-based:** the brief prefers it, and it allows fixed seeds and an end-to-end rerun for the demo.

### 4.7 Decision log (fill in, date, initial)

| ID | Decision | Accepted? (Y/N/changed) | Date | Initials |
|---|---|---|---|---|
| D1 | Track C | | | |
| D2 | Base paper = Ballegeer et al. 2025 (EJOR) | | | |
| D3 | Datasets HMEQ + SGCS + Taiwan (GMSC optional) | | | |
| D4 | Models/explainers as in §4.3 | | | |
| D5 | Hypotheses H1–H5 confirmatory, H6 exploratory | | | |
| D6 | Roles as in §4.5 | | | |
| D7 | Group leader for file naming | | | |

---

## 5. Research design

### 5.1 Gap, novelty, contributions (content specification — write the final text yourselves)

**Gaps** (the paper's gap paragraph should cite the evidence for each):
- **G1** The base paper and the credit-scoring XAI literature induce instability by retraining or resampling (base paper; S34; S58). Local explanation stability under **prediction-preserving input changes at a fixed model** is untested on credit data for SHAP *and* LIME with LR/RF/XGB. S62 is closest but covers TreeSHAP only on IoT data.
- **G2** Reported instability **mixes model variation with explainer stochasticity.** The base paper's SHAP (PermutationExplainer) and LIME are both sampling-based, and no noise floor is reported.
- **G3** No study **explains** the base paper's instability. Crossing mechanisms are discussed (S25, S77, S23) but not quantified at the instance level against noise floors in a business setting.
- **G4** Cheap ex-ante warnings of fragile local explanations rely on uncertainty (S52), multi-model agreement (S75) or calibration data (S77). None estimates, from model structure alone, **how likely a realistic input change is to cross a split threshold or LIME bin edge**.

**Novelty (one paragraph in the paper):** a replication-plus-extension of the base paper on its own public credit data. The extension adds (i) a noise-floor-controlled, prediction-preserving perturbation test of local SHAP and LIME explanations, (ii) threshold and bin-edge crossings as a mediating mechanism, including a double-dissociation test, and (iii) a closed-form expected-crossing score F(x) that needs neither perturbation runs nor calibration data, validated against margin and uncertainty baselines. The paper must state what is *not* new: that crossings can cause instability (S25, S77).

**Contributions (3–5, verb-first, with numbers filled in after the results):**
1. *Replicate* the base paper's retraining-based stability analysis for N model variants on 2 of its public datasets, reporting where results agree and differ.
2. *Separate* explainer stochasticity from model-induced instability for N configurations (E1, H2).
3. *Show* how often local reason codes change under prediction-preserving perturbations across 3 datasets, and that \|Δp\| detects this with AUROC = … (H3).
4. *Identify* threshold/bin-edge crossings as the mediating mechanism (indirect effect = …, 95% CI …), including a double dissociation between TreeSHAP and LIME (H4).
5. *Propose and validate* F(x): AUROC = … vs. … for the best baseline (H5), with an open, reproducible audit notebook used in a credit adverse-action case study.

### 5.2 Research questions and hypotheses

**RQ1 (replication):** Do the base paper's findings — stability falls with class imbalance and is lower for cost-sensitive models — hold on its public data and on an independent credit sample?
**RQ2 (noise and perturbation):** How much local instability remains once explainer stochasticity is removed, and does prediction stability indicate explanation stability?
**RQ3 (mechanism and prediction):** Do threshold/bin-edge crossings explain the instability, and can a structure-only score flag fragile explanations in advance?

| ID | Hypothesis (state it once in the paper, in your own words) | Type | Test (§6.5) |
|---|---|---|---|
| **H1a** | Local explanation instability (CoV, SRA) rises as the training default rate falls | Replication, directional | Jonckheere–Terpstra trend test; Spearman |
| **H1b** | Cost-sensitive variants (cslogit, csboost) are less stable than their traditional counterparts | Replication, directional | Paired Wilcoxon per instance; KS test as in the base paper |
| **E1** | *Estimation objective, not a hypothesis:* share of the base-style across-refit variance attributable to explainer stochasticity | Descriptive | Variance decomposition / ICC |
| **H2** | Under prediction-preserving perturbations, local explanation change exceeds the explainer's own noise floor (tree models: both explainers; LIME: all models) | Extension | One-sided paired Wilcoxon; exact explainers: share > 0 |
| **H3** | \|Δp\| is a weak indicator of reason-code change for tree models (AUROC upper 95% CI < 0.70) and a stronger one for LR + LinearSHAP | Extension | AUROC + cluster bootstrap; full and filtered sets |
| **H4a** | Crossing count mediates the effect of perturbation magnitude on explanation change (tree + TreeSHAP; discretised LIME) | Extension (mediator) | 5,000-resample cluster bootstrap of a·b |
| **H4b** | Double dissociation: TreeSHAP change tracks split-threshold crossings, not LIME-bin crossings; LIME change tracks bin crossings, not split crossings | Extension | Joint regression, coefficient pattern |
| **H4c** | Turning off LIME discretisation removes the bin-edge effect | Extension (intervention) | Paired comparison, on vs. off |
| **H5** | F(x) discriminates unstable instances better than perturbation-free baselines (decision margin; tree-variance uncertainty) | Extension (new method) | Paired AUROC comparison (DeLong or paired bootstrap), Holm |
| **H6** | *Exploratory:* adding crossing/F(x) variables to the base model's predictors (model class, IDCS, default rate) raises explained variance. IDCS models have higher mean F(x) | Bridge | Nested OLS F-test, ΔR² with CI |

Pre-specified thresholds (top-k = 4, τ = 0.05, instability label cut-off 0.2, AUROC 0.70) must be frozen in the analysis plan **before** the main runs (§6.10). The 0.70 cut-off follows the conventional "acceptable discrimination" band of Hosmer & Lemeshow. Cite their textbook, or drop the threshold and test AUROC against the LR contrast instead.

### 5.3 Conceptual model (specification for Fig. 1, drawn in draw.io)
- **Left block — base model (replicated):** Model class, Cost-sensitivity (IDCS), Class imbalance (default rate) → *Explanation instability across retraining* (CoV, SRA). Label each arrow H1a or H1b.
- **Right block — extension (instance level, fixed model):** Perturbation magnitude *s* → **Crossings C** (mediator; split thresholds for trees, bin edges for LIME) → *Local explanation change ΔE / reason-code change* (H4a).
  - Dashed arrow: Prediction change \|Δp\| → ΔE, labelled "weak" (H3).
  - **Explainer noise floor** as a control/baseline box feeding ΔE (H2, E1).
  - Moderators on the C → ΔE path: explainer type and model class (H4b). LIME discretisation switch (H4c).
  - **Fragility score F(x)**, computed from model structure only, → *Instance instability label Y* (H5).
- **Bridge arrow (dashed, H6):** base factors → F(x).
- Use one colour for base constructs and one for new ones. Keep the legend small.

### 5.4 Construct definitions (paper's construct table; each measure needs a source)

| Construct | Operational definition | Source of the measure |
|---|---|---|
| Explanation instability (retraining) | Instance-level CoV of attributions, and SRA at depth 10, across refits | Base paper, Eqs. 6–8 |
| Local explanation change ΔE | Top-k Jaccard distance (k = 4, primary); 1 − Spearman; normalised L1 | Top-k overlap as in A2 corpus (e.g., S42); Spearman as in S34 and others |
| Reason-code change RC | 1 if the top-4 feature set by \|φ\| changes beyond the noise floor | Our definition. Motivated by adverse-action practice (4 principal reasons) — **[ASSUMPTION]** check the regulatory source before citing it |
| Noise floor | Distribution of ΔE between K repeated explanations of the same x with independent seeds | Concept from A1. Run-to-run stability in S21, S46 |
| Prediction-preserving perturbation | Same predicted class and \|Δp\| ≤ τ | Label-preserving perturbations in Agarwal et al. (S09); τ-rule as in S62 |
| Perturbation magnitude s | Gaussian noise with SD = s · SD_j on perturbable continuous feature j | A1. Scale idea as in S62 (IQR-based) |
| Crossing count C | Number of split thresholds (trees) or LIME bin edges lying between x_j and x′_j | Our operationalisation of the S25 and S77 mechanism |
| Fragility F(x) | Expected crossing count under the perturbation distribution (§6.4, Eq. F) | **New** |
| Cost-sensitivity | Instance-dependent cost-sensitive (IDCS) loss vs. cross-entropy | Base paper; Höppner et al. 2022; Bahnsen et al. 2014/2015 |
| Class imbalance | Training default rate π ∈ {1, 3, 5, 10, 15, 20, 25, 30}% | Base paper |

### 5.5 Theoretical background (one page in the paper)
Suggested lens — confirm or replace:
1. **Explanations as measurement instruments.** In classical test theory, observed score = true score + error, and reliability limits validity. The noise floor estimates the error variance. Instability beyond it is a reliability failure. This gives the Track C "reliability/validity" language a principled home (cite a foundational CTT text).
2. **Piecewise-constant structure of tree ensembles and discretised LIME.** This is the structural reason crossings are necessary for change (Proposition P1, §6.4).
3. **Algorithmic accountability in credit decisions.** Adverse-action notices require specific reasons, so unstable local explanations have operational and regulatory cost. **[ASSUMPTION]** Use primary sources only (e.g., the CFPB circular on adverse-action notices for complex algorithms, or EU AI Act provisions). Verify each before citing.

---

## 6. Methodology

### 6.1 Data preparation (Part 0)
- **Splits:** for Part 1, follow the base paper: a fixed stratified test set of **300** instances per dataset, with the remainder used as the training pool. For Part 2, use a stratified 70/30 split at the natural default rate. For SGCS, the 300 test rows are reused because the dataset is small. All splits are seeded.
- **Encoding:** one-hot encode categoricals. Standardise numerics on training statistics only (base paper §3.7).
- **Missing values:** **[FACT]** not stated in the base paper. **[DECISION]** median (numeric) and mode (categorical) imputation fitted on the training fold. HMEQ has substantial missingness, especially `DEBTINC`; report counts in the data-screening table.
- **Data screening table (required):** missing values per feature, outliers (share beyond 1.5·IQR, retained), skewness. Skewness justifies the non-parametric tests.
- **Perturbable features:** continuous features only (e.g., HMEQ: LOAN, MORTDUE, VALUE, YOJ, CLAGE, DEBTINC; Taiwan: LIMIT_BAL, BILL_AMT1–6, PAY_AMT1–6; SGCS: duration, amount, age). Counts, ordinal and categorical features are held fixed. Mark a `perturbable` flag in the codebook.
- **Codebook (deliverable):** one CSV per dataset: variable, type, description, unit, transformation, perturbable flag, and the role in the cost model (A_i).

### 6.2 Part 1 — Replication protocol (Saif)

#### 6.2.1 Protocol (from the base paper, §3.8, Algorithm 2)
For each dataset, model variant and default rate π ∈ {1, 3, 5, 10, 15, 20, 25, 30}%:
1. Resample the training pool to rate π with fixed size = #defaults / 0.3. All defaults are kept; non-defaults are undersampled. **[ASSUMPTION]** The base paper implies undersampling but does not state it.
2. In iteration 1, tune hyperparameters by 5-fold CV on the base paper's grid (Table 5). Reuse them in later iterations.
3. Fit, then explain all 300 test instances with SHAP `PermutationExplainer` and LIME (ridge, exponential kernel). Rank features by absolute value.
4. Repeat for J iterations. Compute per instance: CoV = (1/P) Σ_p σ_ip / μ_ip, and SRA at d = 10.

#### 6.2.2 Cost model for the IDCS variants
C(0|1) = A_i · LGD with LGD = 0.75. C(1|0) = r_i + C_alt, with C_alt = −r̄·π0 + Ā·LGD·π1. π0 and π1 are fixed at the original priors.
- **[OPEN]** r_i (lost revenue) follows Bahnsen et al. (2014) but no formula is given. Read Bahnsen 2014. If it cannot be reproduced, use r_i = m · A_i with a stated margin m, and add a sensitivity row.
- **IDCS implementation:**
  1. First locate the Python implementation by Vanderschueren et al. (2022a) that the base paper builds on.
  2. Otherwise, implement cslogit as AEC + L1/L2 penalty minimised with scipy.
  3. Implement csboost as an XGBoost custom objective (gradient and Hessian of AEC with respect to the margin). The Hessian can be negative, so clip it to a small positive value and document this.
  4. **Sanity gate:** IDCS savings ≥ traditional savings on the same data. If not, the implementation is suspect.
- **Time box: 6 hours on Sat 14 Nov.** No-go fallback: replicate H1a with traditional models only, and report H1b as not replicated because of implementation constraints.

#### 6.2.3 Size and fidelity — data-driven choice on Fri 13 Nov
- **MVP:** 2 datasets (HMEQ, SGCS) × 8 rates × **J = 10** × 4–5 variants × 300 instances. Estimated 10–15 CPU-hours total; about 2–4 hours wall-clock with 8 cores per machine.
- **Full fidelity:** J = 25 (about 2.5× the cost). Run it if the pilot timing shows it fits overnight.
- Record the decision and timings in `admin/analysis_plan.md`.

#### 6.2.4 Known unknowns in the base paper (these drive the deviation table in the paper)
- A_i column per dataset (supplementary material).
- r_i formula.
- SHAP background data.
- LIME num_samples and discretisation settings.
- Seeds.
- Whether the 300-instance test set overlaps the CV folds.
- Whether CoV uses absolute or signed values.
- **SRA formula:** Eq. 8 as rendered on arXiv is inconsistent with the paper's Table 4. The form that reproduces Table 4 is SRA = (1/|S(d)|) Σ_{p∈S(d)} Var_L(R_p). Check the published PDF and **unit-test your SRA against Table 4's worked example**.
- **Stability values appear only in figures.** Compare curve shapes and orders of magnitude. If you digitise figures (e.g., WebPlotDigitizer), label them "approximate, read from Fig. X of [base]".
- **Deliverable:** a deviation table (setting | base paper | ours | reason) in the paper or its appendix.

#### 6.2.5 Estimation objective E1 (cheap bridge to the extension)
For 2 rates (1% and 30%) on one dataset, re-explain each fitted model K = 5 times with fresh explainer seeds. Decompose the across-refit attribution variance into between-refit and within-refit (explainer) parts, and report the explainer share with a bootstrap CI. This quantifies G2 for the base paper's own design.

### 6.3 Part 2 — Extension protocol (Shahoud)

1. **Models:** LR, RF and XGB at the natural default rate, tuned on the training split. Report AUC, AP and Brier next to base-paper Table 6. Note that Table 6 averages four datasets, so per-dataset values must come from the supplementary material.
2. **Operating threshold t:** **[DECISION]** t = training default prior, with 0.5 as a robustness check. Freeze before the main runs.
3. **Instances:** n = 200 per dataset, stratified 50/50 by predicted class so that rejected (adverse-action) cases are well represented. Weight back to the population when reporting rates.
4. **Noise floor:** K = 10 repeated explanations of x with independent seeds. For exact explainers the floor is 0 by construction; confirm this with a unit test.
5. **Perturbations:** for r = 1…R (R = 25) and s ∈ {0.01, 0.05} (s = 0.10 in robustness): x′ = clip(x + δ), δ_j ~ N(0, (s·SD_j)²) for perturbable j, 0 otherwise. Clip to the training [0.5th, 99.5th] percentiles (as in S62).
6. **Filter:** label each draw prediction-preserving if ŷ(x′) = ŷ(x) and \|p(x′) − p(x)\| ≤ τ (τ = 0.05). **Keep non-preserving draws as the contrast group** (needed for H3).
7. **Re-explain** x′. Compute ΔE, RC, C_tree, C_bin and \|Δp\| for each draw.
8. **Ex-ante scores (no perturbations):** F_tree(x), F_bin(x), margin, tree variance (RF), each computed once per x.
9. **Ablation runs:** LIME with discretisation off (H4c); LIME num_samples ∈ {1000, 5000}; SHAP PermutationExplainer on XGB (bridge to Part 1).

### 6.4 Model equations (write the final LaTeX yourselves; define every symbol and add a notation table, since there are more than 10 symbols)

- **Prediction-preserving filter:** keep x′ iff ŷ(x′) = ŷ(x) and \|p(x′) − p(x)\| ≤ τ.
- **Explanation change:** D_k(x, x′) = 1 − \|T_k(φ(x)) ∩ T_k(φ(x′))\| / \|T_k(φ(x)) ∪ T_k(φ(x′))\|, where T_k is the top-k feature set by \|φ\|. Secondary measures: D_ρ = 1 − ρ_S(φ(x), φ(x′)); D_1 = ‖φ(x′) − φ(x)‖₁ / ‖φ(x)‖₁.
- **Noise floor and instability event:** q_i = 95th percentile of {D_k(φ^(a)(x_i), φ^(b)(x_i))} over a < b. U_ir = 1[D_k(x_i, x′_ir) > q_i]. Instance instability rate Ū_i = mean over valid draws of U_ir. Label Y_i = 1[Ū_i ≥ 0.2], with sensitivity at 0.1 and 0.3.
- **Crossing counts:**
  - C_tree = Σ_t Σ_{n∈N_t} 1[x_{j(n)} and x′_{j(n)} lie on different sides of θ_n].
  - C_bin = Σ_j (number of LIME bin edges e ∈ E_j between x_j and x′_j).
  - Note: XGBoost uses `x < split`, scikit-learn uses `x ≤ threshold`. Handle both conventions.
- **Fragility score (Eq. F):**
  - F_tree(x) = (1/T) Σ_t Σ_{n∈N_t} Φ(−\|x_{j(n)} − θ_n\| / σ_{j(n)}), with σ_j = s·SD_j for perturbable j (term = 0 otherwise).
  - F_bin(x) = Σ_j Σ_{e∈E_j} Φ(−\|x_j − e\| / σ_j).
  - **Property:** by linearity of expectation, F equals the **expected crossing count** under the unclipped, unfiltered perturbation distribution. No independence assumption is needed. **Acceptance test:** the Monte-Carlo mean of C matches F within sampling error.
  - **[DECISION]** Pre-specify two variants only: unweighted, and weighted by node cover. No tuning of weights.
- **Proposition P1** (prove it in your own words; verify against the pinned library source):
  - (i) For a tree ensemble, exact TreeSHAP attributions are piecewise constant in x. They change only if some x_j crosses a split threshold on feature j.
  - (ii) With discretisation on, LIME's explanation depends on x only through its bin indices, plus the single sample at x itself and sampling noise. **[ASSUMPTION]** This rests on how `lime_tabular` builds its neighbourhood (`inverse[0] = data_row`). Verify it in the pinned version's source code.
  - **Implication:** for exact TreeSHAP, zero crossings ⇒ ΔE = 0 exactly (a unit test, not a hypothesis). For LIME, zero bin crossings ⇒ ΔE at noise-floor level (testable).
- **Mediation (H4a):** C_ir = a₀ + a·s_r + controls + u_i; ΔE_ir = b₀ + c′·s_r + b·C_ir + controls + v_i. Indirect effect = a·b. CI from 5,000 **cluster** bootstrap resamples over instances. Fit separately per model–explainer pair.
- **Double dissociation (H4b):** RC_ir regressed on C_tree and C_bin jointly, logistic, with cluster-robust SE. Predicted pattern for TreeSHAP: β_tree > 0, β_bin ≈ 0. For LIME: the reverse.
- **Nested models (robustness / H6):** at instance level, regress Ū_i:
  - M0: dataset + model + explainer (+ IDCS + π for H6)
  - M1: M0 + mean \|Δp\|
  - M2: M1 + mean C
  - M3: M2 + F
  - Compare adjusted R² and AUROC for Y_i. F-tests with HC3 standard errors.
- **Baselines for H5:** B_margin(x) = −\|p(x) − t\|. B_var(x) = variance of tree predictions (RF; for XGB an RF surrogate as in S52 — optional). Reference only: post-hoc \|Δp\| (not perturbation-free). Chance = 0.5.

### 6.5 Statistical analysis plan

| Hyp. | Unit | Test | Effect size | Multiplicity | Decision rule |
|---|---|---|---|---|---|
| H1a | Instance × rate | Jonckheere–Terpstra across ordered π; Spearman ρ(π, CoV) | ρ, median CoV ratio π = 1% vs 30% | Holm across dataset × model × explainer | Supported if a significant decreasing-stability trend appears in the majority of cells with the same sign as the base paper |
| H1b | Instance (paired at same π) | Wilcoxon signed-rank; two-sample KS (as in base) | Rank-biserial r | Holm | Supported if IDCS is less stable in the majority of cells |
| E1 | Instance | Random-effects variance components / ICC(1) | Explainer share of variance (95% CI) | — | Descriptive |
| H2 | Instance (paired: perturbed vs. floor) | One-sided Wilcoxon; exact explainers: share D > 0 with Clopper–Pearson CI | r; share | Holm across cells | Per cell |
| H3 | Draw, clustered by instance | AUROC of \|Δp\| for RC; 5,000-resample cluster bootstrap CI; full and filtered sets | AUROC | — | Supported for tree models if the upper CI < 0.70 on the full set |
| H4a | Draw, clustered | Mediation (Eqs. above); 5,000 cluster-bootstrap CI for a·b | a·b; proportion mediated | Holm across pairs | Supported if the CI excludes 0 |
| H4b | Draw, clustered | Logistic, cluster-robust SE | Odds ratios | Holm | Supported if the predicted coefficient pattern holds |
| H4c | Instance | Wilcoxon on Ū_i with discretisation on vs. off, conditioned on C_bin > 0 | r | — | Supported if instability drops |
| H5 | Instance | Paired AUROC difference: DeLong or paired cluster bootstrap; also Spearman(F, Ū_i) | ΔAUROC (95% CI) | Holm across baselines × datasets | Supported if F beats both baselines in the majority of datasets |
| H6 | Instance | Nested OLS F-test; ΔR² with bootstrap CI | ΔR² | — | Exploratory, report only |

- **Significance level:** α = 0.05, two-sided unless the direction is pre-specified (H2, H3, H5 one-sided). Report effect sizes and CIs for everything, not only p-values. With thousands of draws almost everything is "significant", so the effect sizes carry the argument.
- **Clustering:** draws are nested in instances, so never treat draws as independent. Use cluster bootstrap or cluster-robust SEs.
- **Bootstrap:** 5,000 resamples, seed fixed and reported.

### 6.6 Measurement quality — mapping Track C's psychometric criteria (confirm with instructor, §15 Q2)

| Brief asks for | Why it does not apply directly | What we report instead |
|---|---|---|
| Cronbach's α, composite reliability | No multi-item survey scales | **Test–retest reliability** of each explainer: ICC across the K repeated calls (the noise floor) |
| AVE (convergent validity) | No latent constructs | **Convergent validity** of ΔE measures: Spearman correlations among D_k, D_ρ, D_1 (expected moderate to high) |
| HTMT (discriminant validity) | No reflective constructs | **HTMT-style logic:** correlation between explanation change and prediction change (heterotrait) should be below the correlations among explanation-change measures (monotrait) |
| Common-method-bias check | No self-report | **Shared-randomness control:** independent seeds for baseline vs. perturbed explanations; a check that the explainer seed alone does not drive ΔE |
| Informed consent | No human participants (public, anonymised secondary data) | Ethics statement, licences, protected-attribute note (§11) |
| Bootstrap CIs for indirect effects | Applies | 5,000-resample cluster bootstrap for a·b (H4a) |
| A-priori power analysis | Applies | G*Power (§6.7) |

In the paper, title these tables "Measurement quality" and "Structural (hypothesis) results". Say explicitly that SEM indices are replaced by these analogues, and why.

### 6.7 Sample size and power
- **G*Power:** "Exact — Correlation: Bivariate normal model", ρ = 0.20, α = 0.05, power = 0.80, two-tailed → **about 194 instances**. Run it, screenshot it, and report the exact output. This justifies n = 200 instances per dataset for the instance-level analyses (H2, H5, Spearman(F, Ū)).
- **Draws per instance:** R = 25 gives an SE of Ū_i ≤ 0.10 (worst case p = 0.5).
- **Part 1:** J = 10 refits gives a CoV estimate per instance. Note in the limitations that J = 25 is the base paper's value.

### 6.8 Robustness plan (Track C item 5, 7 marks)

| ID | Variant | What changes | Report |
|---|---|---|---|
| R1 | **Nested models** (M0 → M3) and **component ablation** of F (drop cover weighting; drop the F_bin term; drop the F_tree term) | One component at a time | Table + bar/forest plot of ΔR² and ΔAUROC (required) |
| R2 | τ ∈ {0.02, 0.05, 0.10} | Filter strictness | Table |
| R3 | s ∈ {0.01, 0.05, 0.10} | Perturbation size | Plot: instability vs. s |
| R4 | k ∈ {3, 4, 5}; metric D_k → D_ρ → D_1 | Outcome definition | Table |
| R5 | LIME num_samples 1000 vs. 5000; discretisation on/off | Explainer settings | Table (also H4c) |
| R6 | Exact TreeSHAP vs. PermutationExplainer | SHAP estimator | Table |
| R7 | **Subsamples:** defaulters vs. non-defaulters; per dataset; leave-one-dataset-out for the F cut-off used in the case study | Sample | Table |
| R8 | Operating threshold t = prior vs. 0.5 | Classification rule | Table |
| R9 | Replication: J = 10 vs. 25 (if run); r_i margin assumption | Part 1 settings | Table |

### 6.9 Compute budget (CPU only; no GPU needed)

| Block | Size | Estimate | Plan |
|---|---|---|---|
| Part 1 MVP | 2 datasets × 8 rates × J = 10 × 4–5 variants × 300 instances × 2 explainers | 10–15 CPU-h | Machine A, joblib over instances, overnight Sun 15 Nov |
| Part 2 main | 3 datasets × 3 models × 200 instances × (2 s × 25 draws + 10 floor) × explainers | ≈ 3–4 CPU-h (LIME dominates) | Machine B, overnight Sun 15 Nov |
| Robustness | R2–R8 | ≈ 3–6 CPU-h | Mon 16 Nov |

These estimates are **[ASSUMPTION]**. Measure real timings in the Fri 13 Nov pilot and re-plan if they are off by more than 2×. Cache every explanation to disk (parquet) so figures never require recomputation.

### 6.10 Analysis-plan freeze (prevents HARKing; strengthens the publication case)
By **Sat 14 Nov, 9 PM**, commit `admin/analysis_plan.md` containing:
- hypotheses H1–H6 and their directions;
- all thresholds (t, τ, s, k, Y cut-off, AUROC 0.70);
- the definitions of F (both variants) and the baselines;
- the tests, the α level and the multiplicity corrections;
- the instance sampling rule.

Record the git commit hash in the paper's reproducibility statement. Any later change goes into a **deviation log** with the reason. Optional: post the frozen plan on OSF for a timestamp.

---

## 7. Implementation architecture (team-written code)

### 7.1 Repository layout (inside `Assignment 3/`)
```
Assignment 3/
├── plan.md                       (this file; internal, not submitted)
├── admin/                        (quartile screenshot, instructor emails, analysis_plan.md,
│                                  deviation_log.md, novelty_log.md, power_analysis.png, turnitin report)
├── analysis/
│   ├── README.md                 (names + roll numbers, setup, run steps, expected runtime, outputs)
│   ├── requirements.txt          (exact pinned versions)
│   ├── config.yaml               (all seeds, sizes, thresholds — single source of truth)
│   ├── data/raw/                 (git-ignored if the licence forbids redistribution; download script instead)
│   ├── data/processed/  data/codebook/
│   ├── src/  data.py  models.py  idcs.py  explain.py  perturb.py
│   │         metrics.py  crossings.py  fragility.py  stats.py  plots.py
│   ├── scripts/  01_clean.py  02_replicate.py  03_extend.py  04_robustness.py  05_plots.py  run_demo.py
│   ├── notebooks/  <leader-roll>_<SECTION>.ipynb   (mirrors paper order; calls src/)
│   ├── results/   figures/  (PNG 400 DPI + PDF vector, same files as the paper)
│   └── tests/
├── diagrams/                     (.drawio sources: conceptual model, design flowchart, case study)
└── paper/                        (Overleaf export: main.tex, references.bib, figures/)
```

### 7.2 Module contracts and acceptance tests (definition of done)

| Module | Must do | Acceptance test |
|---|---|---|
| `data.py` | Load, clean, impute, encode, split; emit codebook | Row counts match the source; no test-set leakage (scaler fitted on train only); codebook complete |
| `models.py` / `idcs.py` | Train LR/RF/XGB and cslogit/csboost; resample to π | Fixed seed ⇒ identical predictions; IDCS savings ≥ traditional on the same split |
| `explain.py` | LinearSHAP, TreeSHAP (interventional, fixed background), PermutationExplainer, LIME; per-call seed | Exact explainers: repeated calls are bit-identical; SHAP local accuracy (sum of φ = f(x) − E f) within tolerance |
| `perturb.py` | Gaussian perturbations, clipping, prediction-preserving filter | All x′ within the clip range; only perturbable features change; filter pass rate logged |
| `metrics.py` | D_k, D_ρ, D_1, RC, CoV, SRA | Identical vectors ⇒ 0. **SRA reproduces the base paper's Table 4 example** |
| `crossings.py` | Extract thresholds (sklearn `tree_.threshold`, XGBoost `trees_to_dataframe`); LIME bin edges; count crossings | Hand-built 1-tree example; `<` vs. `≤` convention handled |
| `fragility.py` | F_tree, F_bin (two variants), baselines | **Monte-Carlo mean of C ≈ F** (unfiltered, unclipped) within 3 standard errors |
| (P1 test) | — | **Zero crossings ⇒ exact TreeSHAP unchanged** on 1,000 random draws |
| `stats.py` | Wilcoxon, JT trend, cluster bootstrap, DeLong/paired AUROC, mediation, nested OLS | Recovers known effects on a simulated dataset |
| `plots.py` | Consistent style; 400 DPI PNG + PDF | Fonts ≥ 8 pt at column width; axis labels with units |

### 7.3 Reproducibility and the demo
- **One command per stage:** `python scripts/0X_*.py --config config.yaml`. The notebook calls the same functions, so there is no duplicated logic.
- **Demo mode (`run_demo.py --dataset hmeq --applicant <id>`), under 3 minutes:**
  1. Train or load XGB.
  2. Show the applicant's prediction and top-4 reasons.
  3. Compute F(x).
  4. Run 25 live perturbations with exact TreeSHAP.
  5. Print how often the reason codes flip while the decision is unchanged.
  6. Show the gate decision (§10).
- **Tiny end-to-end config** (`config_demo.yaml`, n = 10) proves the pipeline runs live. Full figures load from cached results.
- **Fresh-environment test** on the *other* member's machine on Fri 20 Nov (checklist item 17).

---

## 8. Paper blueprint (8–10 pages, IEEE conference)

| § | Section | Content requirements (brief) | Figures/tables | Pages | Owner |
|---|---|---|---|---|---|
| — | Title, Abstract (150–250 words), Index Terms (4–6) | Problem, method, **key numbers** | — | 0.3 | Joint (write last) |
| I | Introduction | Problem, motivation, base-paper summary, gap (2–3 sentences) → novelty → numbered contributions → organisation | — | 1.0 | Joint |
| II | Literature Review and Hypotheses Development | Grouped by theme; each hypothesis justified; ends with the gap table + G1–G4 paragraph | **T1** gap table (12–14 rows + "Proposed") | 1.5 | Saif |
| III | Theoretical Background | Lens (§5.5), P1 | **T2** construct definitions | 0.6 | Shahoud |
| IV | Conceptual Model and Research Design | Model diagram, flowchart, H1–H6, equations, sampling, instrument sources, data procedure | **Fig. 1** model, **Fig. 2** flowchart, equations, **T3** notation (appendix if space is short) | 1.4 | Shahoud |
| V | Base vs. Extended Model Comparison | Side-by-side constructs, logic, design | **T4** | 0.3 | Saif |
| VI | Data and Analysis Setup | Sample profile, screening, software and versions, settings, criteria with formulas and thresholds | **T5** sample profile, **T6** analysis configuration | 0.7 | Saif |
| VII | Results and Discussion | Measurement quality; structural/hypothesis results; base vs. extended on **every** sample; explain why | **T7** measurement, **T8** hypotheses (supported/not), **T9** replication vs. published; **Fig. 3** CoV/SRA vs. π (ours vs. base), **Fig. 4** ΔE vs. noise floor, **Fig. 5** ROC: F vs. baselines vs. \|Δp\|, Fig. 6 dissociation (optional) | 2.2 | Both (own hypotheses) |
| VIII | Robustness Analysis | Each variant | **T10** + **Fig. 7** (nested ΔR²/ΔAUROC) | 0.6 | Saif |
| IX | Practical Implications and Case Study | Scenario diagram + step-by-step walkthrough | **Fig. 8** scenario diagram | 0.6 | Joint |
| X | Limitations and Threats to Validity | Construct, internal, external, conclusion validity | — | 0.4 | Joint |
| XI | Conclusion and Future Work | — | — | 0.2 | Joint |
| — | Acknowledgment (AI-use declaration), CRediT statement, References (25–40) | — | — | ~1.0 | Joint |

**Writing order:** Methods (Thu 15–16 Nov, as the code stabilises) → Literature → Results → Robustness/Case → Discussion/Limitations → Intro → Conclusion → Abstract.

**Figure rules:** draw.io for Figs. 1, 2 and 8 (export as PDF vector); matplotlib for plots (one shared style file). Captions below figures; table captions above; booktabs; best result in bold. Every float referenced before it appears. **[OPEN] Q5:** whether references and appendices count toward the 8–10 pages.

---

## 9. Literature plan (25–40 references)

### 9.1 Gap-table candidates (pick 12–14; at least two-thirds peer-reviewed; label preprints)
Every row must come from a paper a team member has **actually read**: abstract + method + results at minimum. A2 extractions are only a starting point (`AI_AUDIT_STATUS.md` lists known correction issues).

| Candidate | Venue (year) | Why in the table | Reader |
|---|---|---|---|
| **Ballegeer et al. (S29) — BASE** | EJOR (2025) | Base model | Both |
| Lin & Wang (S34) | Risks (2025) | Seed-retraining SHAP stability, Taiwan credit | Saif |
| Bhuiyan et al. (S42) | FinTech (2026) | SHAP stability under distribution shift, fraud | Saif |
| Qi et al. (S58) | Mathematics (2026) | Optimising explanation stability, HR attrition | Saif |
| Alkan (S41) | J. Expert Syst. Sustain. Dev. (2026) | Imbalance strategies and TreeSHAP stability | Saif |
| Nnadi et al. (S54) | IEEE Access (2026) | LIME variants with LR/RF/XGB, repeated runs | Saif |
| Tzionis et al. (S64) | Applied Artificial Intelligence (2026) | LIME stability on tree ensembles | Shahoud |
| Hasnain et al. | Applied AI Letters (2026), doi:10.1002/ail2.70030 | Prediction-preserving perturbations (gradient models) | Shahoud |
| Kelodjou et al. (S21) | AAAI (2024) | SHAP sampling instability (noise-floor motivation) | Shahoud |
| Hill et al. (S19) | AISTATS (2024) | Boundary-aware explanation uncertainty | Shahoud |
| Khan et al. (S23) | AISTATS (2024) | Astuteness; discretised LIME | Shahoud |
| Mikriukov et al. (S52) | arXiv preprint (2026) | Uncertainty gating (H5 baseline) | Shahoud |
| Tolay (S62) | arXiv preprint (2026) | Prediction-preserving TreeSHAP perturbations | Shahoud |
| Breeden (S77) | arXiv preprint (2026) | Boundary crossings / forecast uncertainty | Shahoud |
| Pala & García-Cuesta (S25) | arXiv preprint (2024) | Split crossing → RF SHAP divergence | Shahoud |

### 9.2 Other references (foundational or method sources; justify each older item)
Ribeiro et al. 2016 (LIME); Lundberg & Lee 2017 (SHAP); Lundberg et al. 2020 (TreeSHAP, *Nature Machine Intelligence*); Štrumbelj & Kononenko 2014 (permutation Shapley); Bahnsen et al. 2014/2015 and Höppner et al. 2022 (IDCS); Vanderschueren et al. 2022 (IDCS implementation); Breiman 2001 (RF); Chen & Guestrin 2016 (XGBoost); DeLong et al. 1988 (AUROC comparison); Holm 1979; a mediation methods source (e.g., Preacher & Hayes 2008); a classical-test-theory source; dataset sources (UCI SGCS 2019; Yeh & Lien 2009 for Taiwan; Baesens et al. 2016 for HMEQ); Agarwal et al. 2022 (S09, RIS; workshop). **Every reference needs a working DOI or publisher link. Export BibTeX from the publisher and check authors, volume, pages and DOI by hand.**

### 9.3 Novelty re-check (Thu 19 Nov; log in `admin/novelty_log.md`)
Search Google Scholar and arXiv for:
- `"split threshold" SHAP stability`
- `"expected number" crossings explanation`
- `LIME discretization stability perturbation`
- `"prediction-preserving" SHAP LIME tabular`
- `"reason code" stability perturbation credit`
- `fragility TreeSHAP`

Also check whether S52, S62, S75 or S77 have since been published in a journal or conference, and cite the published version if so. Inspect the GitHub project noted in §3.2.

---

## 10. Case study plan (6 marks)
- **Setting:** a lender's home-equity loan decisions (HMEQ). A rejected applicant receives an adverse-action notice listing the top-4 reasons.
- **Selection rule (pre-specify):** the rejected test applicant with the highest F_tree whose decision is preserved under every one of the R draws but whose reason set changes in at least 20% of draws. Also show one low-F applicant as a contrast.
- **Walkthrough (step by step):**
  1. Applicant profile and decision.
  2. TreeSHAP and LIME reasons.
  3. Realistic measurement noise (e.g., DEBTINC or CLAGE recorded slightly differently).
  4. The reason codes flip while the decision stays the same.
  5. F(x) would have flagged this applicant *before* any perturbation.
  6. **Gate policy:** if F(x) exceeds the cut-off chosen on development data (R7), route to manual review or report only reasons stable across a perturbation neighbourhood.
  7. Operational cost: the share of applicants flagged and the share of unstable notices avoided (precision and recall at the cut-off).
- **Fig. 8 (draw.io):** applicant → model → explainer → fragility gate → {issue notice | manual review}.
- **Practical implications:** for model validators (add a fragility check to the model-risk review), compliance (reason-code stability as a documented control), and data teams (input precision matters near thresholds). Do not claim legal compliance.

---

## 11. Ethics, licensing, integrity

- **Data ethics:** public, anonymised secondary data; no human participants; no consent form needed. **[OPEN]** Confirm this with the instructor (checklist item 18 then reads "n/a — no primary data"). SGCS contains personal-status/sex and age attributes. State that fairness is out of scope and that the stability findings should not be read as fairness evidence.
- **Licences:** record each dataset's licence in the README. Redistribute only where it is allowed; otherwise ship download scripts plus SHA-256 checksums of the raw files.
- **Integrity checklist (both members sign off on Fri 20 Nov):**
  - All paper text was typed by us in Overleaf. No text from A2, AI_SESSION_NOTES.md or this plan.
  - All analysis code was written by us. AI was used only for debugging, and that is declared.
  - Every reference was opened and checked (working DOI or publisher link).
  - Every number in the paper comes from `analysis/results/`.
  - Figures are self-drawn; any figure adapted from a paper is captioned "Adapted from [n]".
  - The Acknowledgment states the AI use truthfully, including this planning assistance.
  - The CRediT paragraph matches who actually did what.
- **Turnitin/AI check:** run on Fri 20 Nov, leaving a day to rewrite flagged passages. **[OPEN] Q7:** whether a draft check is allowed earlier.
- **Do not post the paper publicly (arXiv, ResearchGate) before grading.** Turnitin could match the submission against your own preprint.

---

## 12. Timeline and milestones (10 days)

| Date | Shahoud (Part 2) | Saif (Part 1) | Joint / exit criterion |
|---|---|---|---|
| **Wed 11 Nov** (evening) | Read the plan; read the base paper | Read the plan; read the base paper | Decision log D1–D7 filled; **instructor email sent (Q1–Q8)**; SCImago screenshot saved; Overleaf project created from the *IEEE Bare Demo Template* (conference) and view link shared; repo folders created |
| **Thu 12 Nov** | Environment + pinned `requirements.txt`; download/clean HMEQ, SGCS, Taiwan; codebooks; data-screening table | Base-paper close read → replication spec sheet (every setting, unknowns marked); get the EJOR PDF + supplementary via HEC; read Bahnsen 2014 & Höppner 2022 for costs; G*Power analysis | **Exit:** processed data + codebooks committed; spec sheet; power screenshot. Analysis plan v0 drafted together (1 h) |
| **Fri 13 Nov** | `perturb`, `metrics`, `explain`, `crossings`, `fragility` + all unit tests (P1, MC check of F) | Traditional models + resampling loop + CoV/SRA (SRA unit test vs. Table 4); pilot on SGCS (2 rates × 3 iterations); **measure timings** | **Exit:** tests pass; timing table → choose J = 10 or 25 |
| **Sat 14 Nov** | Extension pilot on HMEQ (n = 30, R = 10): filter pass rates, floor sizes, sanity plots; finalise s levels | IDCS time box (6 h): locate or implement cslogit/csboost; savings sanity gate. **GO/NO-GO at 8 PM** | **9 PM: FREEZE analysis plan v1 (commit hash)** |
| **Sun 15 Nov** | Launch Part 2 full runs (Machine B, overnight). Write §III Theoretical Background + §IV Design (equations, notation); draw Fig. 1 and Fig. 2 | Launch Part 1 full runs (Machine A, overnight). Write §II Literature + read and build the gap table (T1) | **Exit:** runs launched; lit + methods drafts; Figs 1–2 drawn |
| **Mon 16 Nov** | H2, H3 analyses; launch robustness runs R2–R6, R8 | H1 replication analysis; T9 comparison with published results (digitise figures if needed); E1 | **9 PM CHECKPOINT:** any core result missing → apply the cut list |
| **Tue 17 Nov** | H4a–c, H5; Figs 4–6 | R1 nested/ablation, R7, R9; T10 + Fig. 7; case-study applicant selection + Fig. 8 | **Exit:** all tables and figures final-candidate (400 DPI + PDF) |
| **Wed 18 Nov** | Write §VII (H2–H5 results), measurement table T7 | Write §V, §VI (T5, T6), §VII (H1/E1), §VIII | Draft §IX case study, §X limitations together |
| **Thu 19 Nov** | Cross-review Saif's sections against the checklist | Cross-review Shahoud's sections | Write Intro (gap → novelty → contributions with numbers → organisation), Conclusion, Abstract; **reference audit** (click every DOI); **novelty re-check**; fit to 10 pages; **teach-back session** (each explains the other's half) |
| **Fri 20 Nov** | Fresh-environment end-to-end run on Saif's machine; fix | Turnitin + AI report; rewrite flagged passages | Package: notebook, zip, requirements, data/codebooks, diagram sources, README; CRediT + Acknowledgment; integrity sign-off |
| **Sat 21 Nov** | Final read | Final read | **Submit by 6:00 PM**; confirm upload receipts; Overleaf view link re-checked |
| After 21 Nov | Demo rehearsal (time it), viva Q&A drill | Same | Demo slot (date TBA) |

### 12.1 Dependencies (critical path)
Base-paper read → replication spec → models + metrics → pilot timings → **plan freeze (Sat)** → full runs (Sun night) → analyses (Mon–Tue) → results writing (Wed) → Intro/Abstract (Thu) → integrity checks (Fri) → submit (Sat).
Part 2 does not depend on IDCS, so a failed IDCS time box never blocks the extension.

### 12.2 Cut list (apply in this order at the Mon 16 Nov checkpoint, or earlier if a gate fails)
1. Drop GMSC and the PermutationExplainer bridge (R6).
2. Drop RF from Part 1.
3. Reduce Part 1 rates to {1, 5, 10, 20, 30}%.
4. Replace the H4a bootstrap mediation with crossing-stratified comparisons (C = 0 vs. C > 0). Keep H4b.
5. Drop H6.
6. Drop the IDCS replication (H1b reported as not replicated, with the reason).

**Never cut:** replication of at least one base finding (H1a) on base-paper data; H2, H3 and H5 on ≥ 2 datasets; robustness table + plot; case study; every required figure and table; the integrity checks.

---

## 13. Roles and responsibilities (summary)

| Work package | Responsible | Reviewer |
|---|---|---|
| WP0 Admin (instructor email, quartile proof, Overleaf, decision log) | Shahoud | Saif |
| WP1 Data + codebooks + screening | Shahoud | Saif |
| WP2 Replication + IDCS + E1 + H1 | Saif | Shahoud |
| WP3 Extension + crossings + F(x) + H2–H5 | Shahoud | Saif |
| WP4 Robustness R1–R9 | Saif (R1, R7, R9), Shahoud (R2–R6, R8) | Other |
| WP5 Literature + gap table | Saif (with Shahoud's 7 readings) | Shahoud |
| WP6 Diagrams | Shahoud (Figs 1–2), Saif (Fig. 8) | Other |
| WP7 Writing | As in §8 | Other |
| WP8 Packaging, README, demo | Joint | — |
| WP9 Integrity (Turnitin, references, AI declaration) | Saif | Shahoud |

---

## 14. Risk register

| # | Risk | Likelihood | Impact | Mitigation | Fallback |
|---|---|---|---|---|---|
| R1 | AI-content or similarity > 15% → zero for the group | Medium | **Critical** | Write all text ourselves; reuse no A2 text; Turnitin on Fri 20 Nov with a day to fix; declare AI use | Rewrite flagged sections; never submit above the limit |
| R2 | AI-use policy covers the planning or code assistance | Medium | High | Ask the instructor (Q3) on 11 Nov; write the code ourselves | Follow the instructor's ruling; adjust the Acknowledgment |
| R3 | IDCS implementation fails or eats time | High | Medium | 6-h time box; sanity gate | Traditional-only replication (H1a); report H1b as not replicated |
| R4 | Base results only in figures → weak "next to published" comparison | High | Medium | Get the supplementary material via HEC; digitise figures with labelling | Qualitative shape comparison + deviation table |
| R5 | Compute overruns | Medium | Medium | Pilot timings Fri; caching; two machines; joblib | Cut list §12.2 |
| R6 | Null results (e.g., F no better than the margin) | Medium | Low–Medium | Pre-registered plan makes nulls reportable | Report honestly; discuss why. Still valid for marks |
| R7 | Instructor rejects public secondary data as "own data" or wants SEM criteria | Low–Medium | High | Ask on day 0 (Q1, Q2) | Add a small primary component, e.g., a ≥30-respondent survey of finance/data students rating trust in stable vs. flipping reason-code notices (validated trust scale; consent form). **Costs about 3 days.** Only if required |
| R8 | Dataset licence forbids redistribution | Medium | Low | Download scripts + checksums | — |
| R9 | Novelty pre-empted by a new paper before 21 Nov | Low | Medium | Re-check on Thu 19 Nov | Narrow the claims; cite it |
| R10 | Library behaviour differs from assumptions (LIME internals, SHAP versions) | Medium | Medium | Pin versions; verify P1(ii) in source; unit tests | Restate P1 for the verified version |
| R11 | Uneven Overleaf contribution or viva weakness | Low | Medium | Section ownership; teach-back Thu 19 Nov | — |
| R12 | Base-paper quartile not Q1/Q2 on SCImago | Low | High | Screenshot today | Switch to a verified backup by Thu 12 Nov (§4.1) |

---

## 15. Open questions and blockers

**Send to the instructor on Wed 11 Nov (one email):**
- **Q1** Track C with public secondary credit datasets: is a dataset we prepared and experimented on acceptable as "your own data", with the base paper's public data as the additional samples?
- **Q2** May SEM/questionnaire criteria (α, CR, AVE, HTMT, CMB, consent) be replaced by the computational analogues in §6.6, with a note that they are not applicable?
- **Q3** Is AI assistance for research planning acceptable if declared? Is AI help limited to *debugging* code we write?
- **Q4** What is "SECTION" in the file name, and should the prefix be our actual roll format (23i-2515) rather than "i22-xxxx"?
- **Q5** Do references and appendices count toward the 8–10 pages?
- **Q6** May we read the base paper from its arXiv accepted manuscript, while citing the EJOR version?
- **Q7** Can Turnitin be run on a draft before final submission?
- **Q8** When is the demo slot, and in what format (own laptop, internet access)?

**Internal blockers:**
- **B1** Which member is the group leader (file naming)? Decide today.
- **B2** The brief refers to "Part A (research) and Part B (analysis) of the original assignment". That document is not in the repository. Locate it (course portal) and check it for anything this plan misses.
- **B3** HEC Digital Library access for the EJOR PDF and supplementary material. Who has it?
- **B4** Machines: cores and RAM on each laptop (for the compute plan).
- **B5** Whether A2's human screening and verification were completed. Matters only if A3 refers to A2 as human-screened.

---

## 16. Requirements traceability

### 16.1 Brief → plan

| Requirement (brief) | Plan § | Owner | Done when |
|---|---|---|---|
| Q1/Q2 2025–26 journal base paper + screenshot | 4.1 | Shahoud | Screenshot in `admin/` |
| References 2021–2026 (older only if foundational); 25–40 | 9 | Saif | Bib count 25–40; older items justified |
| Own data + ≥1 more sample + base public data | 4.2 | Shahoud | 3 datasets processed |
| Script-based tools, justified | 4.6, 7 | Both | Tool justification in §VI |
| Intro ends with gap → novelty → contributions → organisation | 5.1, 8 | Joint | Order checked on 19 Nov |
| Gap table 10–15 rows, base included, "Proposed" last row, gaps paragraph | 9.1 | Saif | T1 complete |
| Conceptual model, flowchart, case-study diagrams | 5.3, 10 | Shahoud / Saif | Figs 1, 2, 8 exported (vector) |
| Results figures base vs. proposed, robustness | 6.8, 8 | Both | Figs 3–7 |
| Self-drawn, 400 DPI/vector, readable, no AI images, sources kept | 7.3, 8 | Both | Figure audit on 19 Nov |
| Track C: replicate base model, report next to published, explain differences | 6.2 | Saif | T9 + deviation table |
| Gap table gap → evidence → how addressed | 5.1, 9 | Saif | — |
| Extended model with real novelty (mediator, new method) | 5.2–5.3, 6.4 | Shahoud | H4/H5 specified and frozen |
| Base vs. extended on main + ≥1 additional sample, same criteria | 6.3, 6.4 (M0–M3) | Both | T8 per dataset |
| Robustness: one-at-a-time removal (table + plot) + alternative spec/subsample | 6.8 | Saif | T10 + Fig. 7 |
| Statistical rigour: power, reliability/validity, CMB, bootstrap 5,000, α stated | 6.5–6.7 | Both | T7, power screenshot |
| Ethics | 11 | Joint | Ethics paragraph |
| Practical implications + detailed case study | 10 | Joint | §IX + Fig. 8 |
| Outline sections 1–13 in order | 8 | Joint | ToC check |
| Min figures/tables list (Track C) | 8 | Joint | Register complete |
| Analysis configuration table | 8 (T6) | Saif | — |
| File structure mirrors paper; seeds; dependency file; README; codebook; comments; 400-DPI plots | 7 | Both | Fresh-env run passes |
| IEEEtran conference template, unchanged | 8 | Shahoud | Template from Overleaf gallery |
| Formatting rules (captions, booktabs, equations, notation table, hypotheses table, cross-refs, BibTeX, style) | 8 | Both | Format pass on 19 Nov |
| Integrity: own words, cite everything, AI only grammar/LaTeX/code-debug, declared; no fake refs; own results; CRediT | 0.1, 11 | Both | Integrity sign-off on 20 Nov |
| Deliverables 1–4, naming, names and roll numbers on p. 1 + README, Overleaf view link | 7, 12 | Joint | Submission receipts |

### 16.2 Submission checklist (brief pp. 10–11) with owner

| # | Item | Owner |
|---|---|---|
| 1 | Base paper Q1/Q2 2025/26, quartile proof attached | Shahoud |
| 2 | Names and roll numbers on page 1 and in the README | Joint |
| 3 | IEEEtran template unchanged | Shahoud |
| 4 | Abstract: problem, method, key numeric results | Joint |
| 5 | Intro ends with gap, novelty, numbered contributions, organisation | Joint |
| 6 | Lit review ends with gap table, proposed work as the last row | Saif |
| 7 | Conceptual model diagram, research-design flowchart, case-study diagram | Shahoud / Saif |
| 8 | Numbered hypotheses and model equations | Shahoud |
| 9 | Base vs. proposed comparison table and result graphs | Saif |
| 10 | Robustness analysis with table and plot | Saif |
| 11 | At least one detailed case study | Joint |
| 12 | All figures self-drawn, readable, ≥400 DPI | Both |
| 13 | Every figure, table and equation referenced in the text | Both |
| 14 | Every reference has a DOI or working link; IEEE BibTeX | Saif |
| 15 | Plagiarism + AI report < 15%, attached | Saif |
| 16 | File naming; Overleaf zip; dependency file; data + codebook; diagram sources | Joint |
| 17 | Analysis runs end-to-end on a fresh setup from the README | Shahoud |
| 18 | Informed consent (primary data only) → "not applicable" note | Joint |
| 19 | Contribution statement for each member | Joint |

---

## 17. After submission: path toward a conference paper (not part of the 21 Nov scope)

**What the assignment version will lack for a competitive venue:**
- **Breadth:** 3 datasets is thin. Add 6–10 public tabular credit and risk datasets.
- **Fidelity:** a full J = 25, 8-rate replication including csforest and the neural nets.
- **Theory:** formal statement and proof of P1. Analyse F(x) as an expected crossing count (e.g., bounds linking crossings to attribution change).
- **Baselines:** S52-style uncertainty gating for every model; S77-style conformal uncertainty; local Lipschitz estimates as an upper reference.
- **Other explainers:** KernelSHAP; a stabilised LIME such as S-LIME.
- **Artefact:** a released package (GitHub + Zenodo DOI).

**Steps:**
1. **Late Nov – mid Dec:** incorporate viva feedback, run the extensions above, re-check novelty.
2. **Dec – Jan:** rewrite for the target venue's format and page limit. The course paper's text can be reused here because it is your own, but check the venue's prior-publication policy.
3. **Venues to evaluate** (check current calls for papers and deadlines yourselves; none are listed here as facts):
   - the World Conference on Explainable AI (Springer CCIS proceedings);
   - XAI workshops at major ML/data-mining conferences, e.g., XKDD at ECML-PKDD (S45 reports acceptance there in 2026);
   - the EXPLAINS conference (S46 reports acceptance in 2026);
   - FAccT and AAAI/IJCAI workshops.
   Given the base paper, an OR/IS **journal** route (e.g., decision-support or expert-systems journals) is also realistic.
4. **Preprint only after grading.** Authorship decided in writing; add the instructor only for substantive contribution.
5. **AI policies differ by venue.** Disclose according to the venue's policy.

**Honest outlook:** a careful, pre-registered, reproducible study with a clear mechanism and validated score is a credible workshop or specialised-conference submission. A top-tier main track would need the breadth and theory upgrades above, and even then acceptance is uncertain.

---

## 18. Evidence log (what this plan rests on)

| Claim | Source | How verified |
|---|---|---|
| All brief requirements | `Assignment_3_Research_Paper.pdf` | Read in full, 11 Nov |
| Base paper protocol, datasets, metrics, Table 6 values, limitations | arXiv:2509.01409 (HTML v1) | Read by a fetch agent on 11 Nov; Sections 1–6 and Tables 1–7 verbatim, figures not readable. Supplementary material not accessed |
| Base paper is a 2025 EJOR journal article | Crossref (A2 workbook, `Included` sheet, S29) | A2 record |
| Our gaps match the base paper's limitations | A2 extraction of S29 + base-paper read | Cross-checked |
| S52, S62, S75, S77 do not close the gap | Full-text reads on 6 Nov (S75, S77 local; S52, S62 partly via fetch summaries) | See the 6 Nov session; S52 and S62 only partly verbatim |
| Crossing mechanism already noted | A2 extractions of S25, S23, S43, S77 | A2 records (partly AI-extracted; re-read before citing) |
| No direct prior work on the full combination (as of 11 Nov) | Two web searches, 11 Nov | Searches; not exhaustive. Re-check on 19 Nov |
| Hasnain et al. 2026 bibliographic data | A1 proposal ref [21]; Wiley page in today's search | Two sources agree |
| EJOR quartile Q1 | — | **Not verified. Screenshot required** |

---

## Appendix A — Draft notation (finalise in the paper's notation table)
x, x′: instance and its perturbation · p(·): predicted default probability · t: operating threshold · ŷ: predicted class · τ: prediction-change tolerance · s: perturbation scale · σ_j = s·SD_j · φ^e(x): attribution vector of explainer e · T_k: top-k set by \|φ\| · D_k, D_ρ, D_1: explanation-change measures · RC: reason-code change · K: noise-floor repeats · q_i: noise-floor 95th percentile · U_ir, Ū_i, Y_i: instability event, rate, label · R: draws per instance · C_tree, C_bin: crossing counts · θ_n: split threshold of node n · E_j: LIME bin edges of feature j · F_tree, F_bin: fragility scores · Φ: standard normal CDF · π: training default rate · J: refits per rate · CoV, SRA: base-paper stability measures · A_i, r_i, LGD: cost-model terms · a, b, c′: mediation paths.
