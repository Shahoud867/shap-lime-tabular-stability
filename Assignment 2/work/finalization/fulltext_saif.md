# AI-assisted full-text retrieval and eligibility assessment — saif

22 records: **4 Include, 18 Not retrieved, 0 Exclude**. Original screening labels unchanged. Human verification pending.

Primary PDF/HTML evidence was read through the web tool. Shell downloads failed (DNS unavailable). Access failure never caused exclusion. U302 numerical inconsistencies prevent uncritical quantitative synthesis.

## N039 — Explainable Ensemble Machine Learning Approach for severity prediction of Hemophilia A

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://ieeexplore.ieee.org/document/11405080). Not established

Evidence: No full-paper exclusion evidence available.

Access: IEEE cache miss. Secondary bibliographic/abstract reproduction only; no legal full text found.

Searches: "10.1109/ised67359.2025.11405080" pdf

## N052 — Balancing Accuracy and Interpretability: Machine Learning Approaches to Predict Heart Failure Readmissions

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://ieeexplore.ieee.org/document/11504054). Not established

Evidence: No full-paper exclusion evidence available.

Access: IEEE cache miss; matching ResearchGate record explicitly says no full text available.

Searches: "10.1109/spicscon69221.2025.11504054" pdf

## N074 — Bus passenger flow prediction using BO-optimized XGBoost

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://doi.org/10.1109/bdicn58493.2023.00035). Not established

Evidence: No full-paper exclusion evidence available.

Access: DOI inaccessible; exact-title search found proceedings contents and secondary records, no article text.

Searches: "Bus passenger flow prediction using BO-optimized XGBoost" pdf

## N088 — A SHAP-Guided Machine Learning Framework for Predicting Intravenous Immunoglobulin Resistance

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://ieeexplore.ieee.org/document/11735905). Not established

Evidence: No full-paper exclusion evidence available.

Access: Exact-title/DOI search did not locate matching accessible primary content; IEEE route inaccessible.

Searches: "A SHAP-Guided Machine Learning Framework for Predicting Intravenous Immunoglobulin Resistance" pdf; "10.1109/icosec70158.2026.11735905"

## N096 — Intelligent Assessment of College Students' Mental Health based on Kendall’s Rank Correlation Coefficient Based J48 Decision Tree

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://ieeexplore.ieee.org/document/10934250). Not established

Evidence: No full-paper exclusion evidence available.

Access: IEEE robot-verification page; matching ResearchGate record offers request-only full text.

Searches: "Intelligent Assessment of College Students" "J48" pdf

## N114 — XAI-RiskNet: Explainable Ensemble Learning for Robust Firm Default Prediction Under Economic Stress

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://ieeexplore.ieee.org/document/11319052). Not established

Evidence: No full-paper exclusion evidence available.

Access: IEEE robot verification; matching ResearchGate record offers request-only full text.

Searches: "XAI-RiskNet" pdf

## N121 — Takeover Time Prediction for Conditionally Automated Driving Vehicles: Considering Mixed Traffic Flow Environment

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://doi.org/10.1109/tits.2025.3557772). Not established

Evidence: No full-paper exclusion evidence available.

Access: DOI inaccessible. TRID reproduces abstract; author/repository search returned a different Delft paper citing this study, not the requested full text.

Searches: "Takeover Time Prediction for Conditionally Automated Driving Vehicles" pdf

## N127 — Data Leakage Detection in Cloud Computing Environment

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://ieeexplore.ieee.org/document/10913619). Not established

Evidence: No full-paper exclusion evidence available.

Access: IEEE robot verification. Search found a third-party licensed-use upload; not used as a verified legal author/repository copy.

Searches: "Data Leakage Detection in Cloud Computing Environment" "10913619"; "10.1109/cybercomp60759.2024.10913619" pdf repository

## N207 — Bridging Predictive Performance and Explainability in Employee Attrition Through Imbalanced Learning and XAI

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://ieeexplore.ieee.org/document/11567396). Not established

Evidence: No full-paper exclusion evidence available.

Access: IEEE cache miss; J-GLOBAL bibliographic metadata only.

Searches: "Bridging Predictive Performance and Explainability in Employee Attrition" pdf

## N234 — Machine Learning-Driven Models for Predictive Diagnosis of Depression and Anxiety in Clinical Settings

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://ieeexplore.ieee.org/document/11717141). Not established

Evidence: No full-paper exclusion evidence available.

Access: Exact-title/DOI search found no matching primary full text; IEEE inaccessible.

Searches: "Machine Learning-Driven Models for Predictive Diagnosis of Depression and Anxiety" pdf; "10.1109/aic70896.2026.11717141"

## N240 — Machine Unlearning in Smart Grids: A Cautionary Tale of Infrastructure Vulnerability

**Include**. English full research report within 2016–2026; tabular data, SHAP-family explanations, and a separately evaluated attribution-change/stability outcome.

[Source](https://www.researchgate.net/publication/406709873_Machine_Unlearning_in_Smart_Grids_A_Cautionary_Tale_of_Infrastructure_Vulnerability). CC-BY 4.0 accepted author manuscript; IEEE DOI stated in footer

Evidence: §IV p 3/Fig 5: mean absolute SHAP shifts at 0%,10%,90%label manipulation, separate from modelF 1.

Access: Primary full text inspected through web text extraction. Local PDF download attempted but DNS unavailable; source URL and locations retained. QA is an AI appraisal, not independent human verification.

Searches: "Machine Unlearning in Smart Grids: A Cautionary Tale" pdf

Extraction:

- problem: Attribution drift from malicious retraining in smart grids.
- rq: RQ 1;RQ 2;RQ 3
- method: Label manipulation/retraining; XGBoost mean absolute SHAP comparison.
- data: Electricalfault tabular measurements; IEEE 24 bus simulated contingencies with 38 binary line features.
- evaluation: Three-repetition fivefoldCV; attribution comparisons 0/10/90%relabeling; Fig 5 top-10 shifts.
- findings: Line 7 mean absolute SHAP approximately 1.2→0.75→0.25 as attack increases.
- limitations: Simulation/selectedcases; no normalized stabilityindex or uncertainty interval for SHAP shifts.
- repro: Public/simulation sources cited; exact seeds/code not located.
- evidence: §III–IV/Fig 5;§V discusses defense tradeoffs.
- read_from: Author CC-BYfulltextHTML onResearchGate.

QA:

- Q1 = 1: Aim§I.
- Q2 = 0.5: Pipeline clear; SHAP settings incomplete.
- Q3 = 1: §III feature descriptions.
- Q4 = 1: Attack levels/baseline.
- Q5 = 0.5: MeanabsoluteSHAP shifts; no aggregate index.
- Q6 = 0.5: Repeated CV; SHAP uncertainty absent.
- Q7 = 1: §V defense drawbacks.
- Q8 = 0.5: Source datasets cited; no project code located.
- Q9 = NA: Grid simulation; clinical/privacy criterion inapplicable.
- Q10 = 0.5: Repeated CV; seeds not found.

## N359 — Neural Emulators vs. Machine Learning Regressors: District‑Scale Pre‑Monsoon Temperature Prediction in Telangana

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://link.springer.com/article/10.1007/s41748-026-01220-5). Not established

Evidence: No full-paper exclusion evidence available.

Access: Springer inaccessible; matching ResearchGate offers request-only text; secondary metadata does not establish eligibility.

Searches: "Neural Emulators vs. Machine Learning Regressors" pdf; "10.1007/s41748-026-01220-5" pdf

## N594 — Identifying key bioprocess variables using explainable machine learning to enhance culture efficiency and viability of umbilical cord-derived mesenchymal stem cells

**Include**. English full research report within 2016–2026; tabular data, SHAP-family explanations, and a separately evaluated attribution-change/stability outcome.

[Source](https://www.medsci.org/v23p1808.htm). Publisher version of record,30 March 2026

Evidence: Results afterTable 4/Fig 5:70%retention of top 10 SHAP ranking when outcome threshold changes 600→564 h.

Access: Primary full text inspected through web text extraction. Local PDF download attempted but DNS unavailable; source URL and locations retained. QA is an AI appraisal, not independent human verification.

Searches: "Identifying key bioprocess variables" "127764"

Extraction:

- problem: Reliability of bioprocess drivers under outcome-threshold choice.
- rq: RQ 1;RQ 2
- method: XGBoost+SHAP; threshold sensitivity analysis.
- data: 203 UC-MSC manufacturing cases, single facility; tabular neonatal/process variables.
- evaluation: 80/20 split; four-fold CV; top 10 SHAP retention at 600 versus 564 h threshold.
- findings: 70%top 10 retention; AUROC change 0.08. This is target-definition sensitivity, not repeated-call or local-input stability.
- limitations: Retrospective single facility; one threshold comparison; rank-retention formula incompletely specified.
- repro: Table 4 seed 42/hyperparameters; data on request; code not located.
- evidence: Results afterTable 4;Discussion;Dataavailability.
- read_from: Complete publisherHTML.

QA:

- Q1 = 1: AimIntroduction.
- Q2 = 0.5: Model described; stability protocol limited.
- Q3 = 1: Methods/Table 1.
- Q4 = 0.5: Two thresholds, no explainer comparator.
- Q5 = 0.5: Top-10 retention, limited specification.
- Q6 = 0.5: Single contrast, no stability interval.
- Q7 = 1: Discussion explicit threats.
- Q8 = 0.5: Data on request; no code located.
- Q9 = 1: IRBA 202505142/consentwaiver.
- Q10 = 1: Table 4 seed 42.

## N687 — Hierarchical Shrinkage: improving the accuracy and interpretability of tree-based methods

**Include**. English full research report within 2016–2026; tabular data, SHAP-family explanations, and a separately evaluated attribution-change/stability outcome.

[Source](https://proceedings.mlr.press/v162/agarwal22b/agarwal22b.pdf). ICML 2022 PMLR proceedings PDF,25 pages incl.appendix

Evidence: §5 p 8/Fig 6; AppendixS 5.3 pp 23–25: SHAP variance under training-data resampling.

Access: Primary full text inspected through web text extraction. Local PDF download attempted but DNS unavailable; source URL and locations retained. QA is an AI appraisal, not independent human verification.

Searches: "Hierarchical Shrinkage: improving the accuracy and interpretability of tree-based methods"

Extraction:

- problem: Unstable tree explanations.
- rq: RQ 1;RQ 2;RQ 3
- method: Hierarchical shrinkage of node predictions; RF SHAP comparison.
- data: 16 tabular benchmarks; SHAP experiments on 8 classification datasets (Table 1).
- evaluation: AppendixS 5.3:50 held-out samples;100 fits using random two-thirds training subsets; mean per-feature SHAP variance.
- findings: Shrinkage reduces SHAP variance, including datasets with little prediction improvement.
- limitations: Reviewer: no local input-noise/noise-floor assessment; §6 leaves broader robustness questions open.
- repro: imodels and Yu-Group/imodels-experiments repositories, p 1.
- evidence: Table 1;§5;AppendixS 5.3.
- read_from: Primary PMLR PDF and landing page.

QA:

- Q1 = 1: Clear aim§1.
- Q2 = 1: Algorithm§2–3/appendix.
- Q3 = 1: Table 1.
- Q4 = 1: RF with/without HS.
- Q5 = 1: Variance definedS 5.3.
- Q6 = 1: 100 fits/50 held-out samples.
- Q7 = 0.5: §6 futurework; limited critical threats.
- Q8 = 1: Repositories p 1.
- Q9 = 0.5: Sensitive benchmarks; limited application ethics discussion.
- Q10 = 0.5: Repetitions explicit; seed values not located.

## U008 — From Symptoms to Clinical Insight: An Explainable Machine Learning Framework for Early Diabetes Screening

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://ieeexplore.ieee.org/document/11546646). Not established

Evidence: No full-paper exclusion evidence available.

Access: IEEE inaccessible; matching ResearchGate offers request-only full text. Supplied abstract cannot substitute for reading methods/results.

Searches: "From Symptoms to Clinical Insight" "Diabetes" pdf

## U033 — Self-Adaptive Genetic Boosted Explainable Framework for Employee Performance Classification

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://ieeexplore.ieee.org/document/11507650). Not established

Evidence: No full-paper exclusion evidence available.

Access: IEEE inaccessible; J-GLOBAL metadata only. Separate stability assessment remains unverified.

Searches: "Self-Adaptive Genetic Boosted Explainable Framework" pdf

## U048 — Explainability and Verifiability of Artificial Intelligence in ICU Mortality Prediction

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://ieeexplore.ieee.org/document/11706859). Not established

Evidence: No full-paper exclusion evidence available.

Access: IEEE inaccessible; official conference program confirms title but supplies no full report.

Searches: "Explainability and Verifiability of Artificial Intelligence in ICU Mortality Prediction" pdf

## U054 — A Robustness-Oriented Quantum–Classical Hybrid Machine Learning Pipeline for Breast Cancer Diagnosis: External Validation, Explainability, and Rigorous Benchmarking in the NISQ Era

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://www.mdpi.com/2075-4418/16/13/1996). Not established

Evidence: No full-paper exclusion evidence available.

Access: MDPI429; alternate /htm route inaccessible; PMC reCAPTCHA; EuropePMC unavailable. PubMed abstract cannot establish separate attribution-stability evaluation.

Searches: "10.3390/diagnostics16131996"; "diagnostics16131996" pdf

## U302 — Explainable Machine Learning Framework for Distributed Denial-of-Service (DDoS) Attack Detection using Comparative Evaluation and SHAP Analysis

**Include**. English full research report within 2016–2026; tabular data, SHAP-family explanations, and a separately evaluated attribution-change/stability outcome.

[Source](https://ejournals.itda.ac.id/index.php/avitec/article/download/3594/pdf). Publisher version of record; Vol 8 No 1 Feb 2026; online 22 Dec 2025

Evidence: §2.8 Eq 5 p 100;§3.1.3 Table 7 pp 106–107: separate SHAP fold-consistency score.

Access: Primary full text inspected through web text extraction. Local PDF download attempted but DNS unavailable; source URL and locations retained. QA is an AI appraisal, not independent human verification.

Searches: "10.28989/avitec.v8i1.3594"

Extraction:

- problem: DDoS attribution consistency versus predictive robustness.
- rq: RQ 1;RQ 2
- method: Six classifiers; tree-model SHAP; FSS=1−SD/mean importance across folds.
- data: CIC-DDoS 2019 flow features; Table 5.
- evaluation: Five-fold CV; four tree models; FSS versus F 1 variability.
- findings: Table 7 RF has highest FSS; prediction-variance ordering differs. Reported correlation sign/magnitude andp-value conflict with table/text; do not synthesize unverified rho.
- limitations: One benchmark; reviewer flags small model-level sample, correlation inconsistency and undefined zero-mean handling.
- repro: Seed 42; package versions Table 5; public benchmark; code link not located.
- evidence: Eq 5;Table 7;Figures 13–14.
- read_from: Publisher 20-pagePDF.

QA:

- Q1 = 1: Aim§1.
- Q2 = 0.5: Implementation settings; FSS aggregation unclear.
- Q3 = 1: §2/Table 5.
- Q4 = 1: Shared protocol/Table 7.
- Q5 = 0.5: Eq 5; zero-mean/aggregation incomplete.
- Q6 = 0.5: Fivefolds; contradictory statistics.
- Q7 = 0.5: Discussion needs diverse/adversarial validation.
- Q8 = 0.5: Public data; no code located.
- Q9 = NA: Public network benchmark; personal clinical data inapplicable.
- Q10 = 1: Table 5 seed 42.

## U407 — Decision support for sustainable maritime transportation under multiple objectives

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://www.sciencedirect.com/science/article/pii/S221053952600204X). Not established

Evidence: No full-paper exclusion evidence available.

Access: Publisher403. Cached abstract/introduction and section snippets support likely eligibility, but complete methods/results and stability-score details unavailable; no complete extraction/QA justified.

Searches: "Decision support for sustainable maritime transportation under multiple objectives" pdf

## U411 — Enhancing Clinical Insight Of An Artificial Intelligence Test For Pulmonary Capillary Wedge Pressure Elevation By Unmasking Physiological Mechanisms

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://doi.org/10.1016/j.cardfail.2025.11.131). Not established

Evidence: No full-paper exclusion evidence available.

Access: DOI inaccessible. Secondary metadata gives one-page pagination226 but report type cannot be confirmed from primary full text. No exclusion inferred.

Searches: "10.1016/j.cardfail.2025.11.131"

## U430 — A machine learning framework for coal gasification process simulation and operation optimization

**Not retrieved**. Full text/relevant complete methods and results not retrieved; eligibility remains pending. Access failure is not an exclusion criterion.

[Source](https://www.sciencedirect.com/science/article/pii/S001623612501659X). Not established

Evidence: No full-paper exclusion evidence available.

Access: Publisher403. Cached complete abstract/introduction and abbreviated method/results section snippets, but relevant complete explanation analysis unavailable; no exclusion inferred.

Searches: "A machine learning framework for coal gasification process simulation and operation optimization" pdf
