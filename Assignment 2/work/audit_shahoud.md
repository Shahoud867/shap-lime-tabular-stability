# AI audit by agent shahoud: S01–S35

Audit date: 6 November 2026. This is AI assistance, not the work or independent screening of the human Shahoud. No human screening columns, canonical data, workbook or report were modified.

## Coverage and limits

All 35 assigned master records and their source JSONL locations were reviewed for auditability. Primary-source excerpts or contents were accessible for 28 studies; seven did not have primary paper text inspected through the attempted routes. Specifically, nine received targeted section checks (S01, S03, S05, S08, S09, S10, S18, S24, S29), sixteen received abstract/introduction-level checks, and three had only title/contents-level inspection (S26, S27, S30). None received a complete independent paper-and-QA re-extraction. This is a targeted evidence audit, not a completed full-paper re-extraction or a fresh independent Q1–Q10 appraisal. An accessible full text is not the same as every field verified. No record is declared wholly verified. Existing study selection is retained pending review of scope concerns.

The 11 local PDF/text files belong to supplementary N-records rather than these assigned studies, so primary web sources were used. Source URLs below distinguish paper evidence from software evidence.

## Concrete corrections for parent review

1. **S10/U155: confirmed dataset omission.** arXiv:2001.11757v2 Section 6 dataset-composition table provides train 39,418, test 16,893 and total 56,311. Replace “sample size not reported”; Q3 rationale is invalid and should be rescored against the full dataset description. [Primary paper](https://arxiv.org/html/2001.11757).

2. **S08/U483: confirmed incomplete limitation.** Appendix B.2/Table 5 evaluates 50 randomly selected instances per breast-cancer/sepsis application, each with 20 repeats. Replace single-instance-only characterization. Reassess Q6 rather than automatically awarding 1. [Primary paper](https://arxiv.org/html/2106.07875).

3. **S05/C058: confirmed reproducibility omission.** Paper footnote links [author project](https://dylanslacks.website/reliable/index.html), whose GitHub link resolves to [author code](https://github.com/dylan-slack/Modeling-Uncertainty-Local-Explainability). Add link; re-evaluate Q8 (currently 0.5 attributed to no code link).

4. **S18/U556: overgeneralized numerical finding.** Section 5 describes substantial variability across settings, including <25% reductions for some independent-feature NN cases and >50% in others. Replace unqualified 80–95% range with “ControlSHAP reduced Monte Carlo variability across evaluated settings, with size depending on model, dataset and feature-dependence assumption; selected examples reached about 90%, and a displayed example matched 1,000 plain iterations after 100 adjusted iterations.” [Primary Sections 5–6](https://arxiv.org/html/2310.07672).

5. **S35/U042: software availability update.** [Author repository](https://github.com/ntakafumi/aime) now explicitly implements BayesianAIME. Record as software located on audit date; do not pretend it was checked during original search or that the paper’s numerical findings were verified.

## Scope concerns requiring a consistent decision

- **S03:** attack-detection top-k accuracy is a robustness outcome, but is not repeated-call/input-neighborhood stability. Keep adversarial evidence distinct; clarify IC3/EC1 interpretation before any exclusion. [Primary Section 4.3](https://arxiv.org/pdf/2006.12302).
- **S24:** CE is a different method; paper makes conceptual SHAP/LIME comparisons while numerical stability experiments focus on CE. Check whether this meets the documented IC2 “studied or compared”; do not describe its CE variance as a measured SHAP/LIME result. [Primary Sections 5–6](https://arxiv.org/html/2305.02305).

## Per-study audit trail

### S01 — On the Robustness of Interpretability Methods
- Record: `C069`; source: `Assignment 2/work/extract/ft_003.jsonl:13`.
- Inspection depth: Targeted sections, not complete full text.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/1806.08049
- Finding: Primary abstract and Section 2 support local-Lipschitz framing and nearby-input instability. Partial check; all dataset details and Q1–Q10 not independently validated.
- Existing Q1–Q10: `[1, 1, 0.5, 1, 1, 0.5, 0.5, 0.5, 'NA', 0.5]`. Existing rationale: Q3 dataset details brief; Q6 small-scale; Q8 libraries only; Q10 sampling described, no repeated seeds.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S02 — "Why Should You Trust My Explanation?" Understanding Uncertainty in LIME Explanations
- Record: `C068`; source: `Assignment 2/work/extract/ft_003.jsonl:12`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/1904.12991
- Finding: Primary abstract identifies three uncertainty sources and synthetic/tabular COMPAS plus text data. Partial check; 100-run count and every quality item remain unverified.
- Existing Q1–Q10: `[1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 'NA', 1]`. Existing rationale: Q3 data preprocessing brief; Q4 no baseline; Q5 selection probability only; Q6 descriptive; Q8 none; Q10 100 repeated runs.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S03 — Improving LIME Robustness with Smarter Locality Sampling
- Record: `C061`; source: `Assignment 2/work/extract/ft_003.jsonl:9`.
- Inspection depth: Targeted sections, not complete full text.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/pdf/2006.12302
- Finding: Primary PDF Sec 4.3, pp 3–4 measures top-k detection of biased behavior. Finding is supported, but outcome is attack detection rather than repeated-call or neighboring-input stability. Flag scope/metric compatibility; do not automatically exclude.
- Existing Q1–Q10: `[1, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1, 'NA', 0.5]`. Existing rationale: Q2/Q3 brief; Q5 detection accuracy only; Q6 small experiments; Q10 no repetitions.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S04 — Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods
- Record: `C062`; source: `Assignment 2/work/extract/ft_003.jsonl:10`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/1911.02508
- Finding: Primary abstract supports biased scaffolding of SHAP/LIME and preserved biased predictions. Attack robustness must remain separate from ordinary perturbation stability. Remaining extraction/QA only partially checked.
- Existing Q1–Q10: `[1, 1, 1, 1, 0.5, 1, 0.5, 1, 1, 0.5]`. Existing rationale: Q5 rank-share outcome; Q7 brief; Q9 discrimination and fairwashing discussed; Q10 repetitions not reported.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S05 — Reliable Post hoc Explanations: Modeling Uncertainty in Explainability
- Record: `C058`; source: `Assignment 2/work/extract/ft_003.jsonl:8`.
- Inspection depth: Targeted sections, not complete full text.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2008.05030
- Finding: Primary introduction supports 53% average improvement. First-page footnote links author project page, which links working code repository. Correct reproducibility omission and reassess Q8.
- Existing Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 0.5, 'NA', 1]`. Existing rationale: Q8 no code link in text; Q10 repeated sampling with significance tests.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S06 — On the Trustworthiness of Tree Ensemble Explainability Methods
- Record: `U373`; source: `Assignment 2/work/extract/ft_001.jsonl:5`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2110.00086
- Finding: Primary abstract supports global feature importance and input/model perturbations. Distinguish global from local stability. QA and exact repetition counts not fully checked.
- Existing Q1–Q10: `[1, 1, 0.5, 1, 1, 1, 0.5, 0.5, 'NA', 1]`. Existing rationale: Q3 proprietary dataset undisclosed; Q7 limitations brief; Q8 no code link; Q10 50 seeded repetitions.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S07 — Deterministic Local Interpretable Model-Agnostic Explanations for Stable Explainability
- Record: `U577`; source: `Assignment 2/work/extract/ft_005.jsonl:12`.
- Inspection depth: No primary paper inspection.
- Coverage: Primary paper not inspected; unverified.
- Source attempted/checked: https://doi.org/10.3390/make3030027
- Finding: Publisher returned HTTP 429. No primary full text inspected in this run. Existing extraction and scores remain unverified.
- Existing Q1–Q10: `[1, 1, 1, 1, 1, 1, 0.5, 1, 'NA', 1]`. Existing rationale: Q7 brief; Q10 repeated runs.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S08 — S-LIME: Stabilized-LIME for Model Explanation
- Record: `U483`; source: `Assignment 2/work/extract/ft_001.jsonl:3`.
- Inspection depth: Targeted sections, not complete full text.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2106.07875
- Finding: Primary Appendix B.2/Table 5 explicitly includes 50 test instances per application with 20 repetitions; existing single-instance limitation is incomplete and Q6 rationale needs review.
- Existing Q1–Q10: `[1, 1, 1, 1, 1, 0.5, 0.5, 1, 'NA', 1]`. Existing rationale: Q6 few instances; Q7 scope limits noted; Q8 code and data public; Q9 no personal data beyond public benchmark (N/A); Q10 repetitions control sampling noise.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S09 — Rethinking Stability for Attribution-based Explanations
- Record: `U498`; source: `Assignment 2/work/extract/ft_001.jsonl:4`.
- Inspection depth: Targeted sections, not complete full text.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2203.06877
- Finding: Primary Sec 4 supports 233% theoretical-bound difference and comparison across seven methods. Exact metric values/QA not fully checked.
- Existing Q1–Q10: `[1, 1, 0.5, 1, 1, 0.5, 0, 0.5, 'NA', 0.5]`. Existing rationale: Q3 datasets named, sizes not given; Q6 single figure of medians; Q7 no limitations section; Q8 data public, code not linked; Q10 number of perturbations/seeds not detailed.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S10 — Statistical stability indices for LIME: Obtaining reliable explanations for machine learning models
- Record: `U155`; source: `Assignment 2/work/extract/ft_001.jsonl:2`.
- Inspection depth: Targeted sections, not complete full text.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2001.11757
- Finding: Primary v2 Sec 6 dataset-composition table reports 56,311 total (39,418 train; 16,893 test), contradicting sample-size-not-reported extraction and Q3 rationale. Correct dataset entry; reassess Q3.
- Existing Q1–Q10: `[1, 1, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 1]`. Existing rationale: Q3 dataset size not reported; Q4 no competing stability measure; Q6 case-study evidence; Q8 code yes, data no; Q9 regulation discussed; Q10 repeated calls isolate sampling noise.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S11 — Uncertainty Quantification for Local Model Explanations Without Model Access
- Record: `C040`; source: `Assignment 2/work/extract/ft_003.jsonl:3`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2301.05761
- Finding: Primary abstract supports bootstrap interval width/coverage claims and author code link. QA numerical judgment not independently validated.
- Existing Q1–Q10: `[1, 1, 1, 1, 1, 0.5, 0.5, 1, 'NA', 1]`. Existing rationale: Q6 real-data results visual only; Q7 brief; Q10 simulation over parameter grid.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S12 — Locally Invariant Explanations: Towards Stable and Unidirectional Explanations through Local Invariant Learning
- Record: `U493`; source: `Assignment 2/work/extract/ft_001.jsonl:34`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/pdf/2201.12143
- Finding: Primary PDF abstract supports invariant-risk-minimization LIME variant and stable explanations. Exact 21/40 and 5/40 findings and all quality items remain unverified.
- Existing Q1–Q10: `[1, 1, 1, 1, 1, 1, 0.5, 0.5, 'NA', 1]`. Existing rationale: Q7 limitations in supplement; Q8 no code link; Q10 multiple environments and reported SDs.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S13 — Towards Quality Measures for xAI algorithms: Explanation Stability
- Record: `U017`; source: `Assignment 2/work/extract/ft_006.jsonl:3`.
- Inspection depth: No primary paper inspection.
- Coverage: Primary paper not inspected; unverified.
- Source attempted/checked: https://doi.org/10.1109/dsaa60987.2023.10302535
- Finding: DOI full-text route inaccessible. Metadata alone does not verify extracted results or QA. Recheck original p10 and seed claim.
- Existing Q1–Q10: `[1, 1, 1, 0.5, 1, 1, 1, 0.5, 'NA', 0.5]`. Existing rationale: Q4 one model; Q8 data only; Q10 seeds not reported.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S14 — An empirical study of the effect of background data size on the stability of SHapley Additive exPlanations (SHAP) for deep learning models
- Record: `U481`; source: `Assignment 2/work/extract/ft_001.jsonl:6`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/pdf/2204.11351
- Finding: Primary PDF abstract supports MIMIC-III background-sample fluctuation. Publication header states ICLR 2023 Tiny Paper. Every quality score and 100-resampling count remain unverified.
- Existing Q1–Q10: `[1, 1, 1, 0.5, 1, 1, 0.5, 1, 0.5, 1]`. Existing rationale: Q4 compares background sizes, no alternative explainer; Q7 brief; Q8 code public; Q9 clinical data, ethics not discussed; Q10 100 resamplings per size.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S15 — Enhancing Model Explainability with CTGAN-LIME: A Novel Approach for Interpretable Machine Learning
- Record: `U258`; source: `Assignment 2/work/extract/ft_006.jsonl:8`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://avestia.com/EECSS2024_Proceedings/files/paper/CIST/CIST_158.pdf
- Finding: Primary proceedings PDF abstract supports CTGAN-LIME improvement claims and tabular setting. Exact Fig3 stability procedure/metric needs closer check; avoid treating abstract assertion as full verification.
- Existing Q1–Q10: `[1, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0, 'NA', 0.5]`. Existing rationale: Q2 brief; Q4 LIME only; Q5 brief; Q8 none; Q10 not detailed.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S16 — Adversarial attacks and defenses in explainable artificial intelligence: A survey
- Record: `U466`; source: `Assignment 2/work/extract/ft_001.jsonl:28`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2306.06123
- Finding: Primary abstract and introduction support adversarial-XAI survey, tabular coverage, and security scope. It is not a benign-perturbation empirical experiment. QA review adaptation remains a judgment.
- Existing Q1–Q10: `[1, 1, 'NA', 'NA', 'NA', 1, 1, 1, 1, 'NA']`. Existing rationale: Review: dataset/baseline/metric items N/A; ethics (fairwashing) discussed.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S17 — Improving Local Interpretable Model-agnostic Explanations Stability
- Record: `U362`; source: `Assignment 2/work/extract/ft_006.jsonl:7`.
- Inspection depth: No primary paper inspection.
- Coverage: Primary paper not inspected; unverified.
- Source attempted/checked: https://doi.org/10.22266/ijies2024.1231.81
- Finding: DOI and attempted publisher PDF unavailable. Search surfaced a paper copy but no primary full text inspected; all detailed scores remain unverified. Do not mark full-text verified.
- Existing Q1–Q10: `[1, 1, 0.5, 1, 1, 0.5, 0.5, 0, 'NA', 1]`. Existing rationale: Q3 dataset sizes brief; Q6 small; Q8 none; Q10 10 repeated explanations.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S18 — Stabilizing Estimates of Shapley Values with Control Variates
- Record: `U556`; source: `Assignment 2/work/extract/ft_002.jsonl:6`.
- Inspection depth: Targeted sections, not complete full text.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2310.07672
- Finding: Primary Sec5–6 shows variability improvements depend on model/features/dataset: some cases below 25%, some above 50%, high reductions in selected settings. Existing unqualified roughly 80–95% statement overgeneralizes; use qualified wording.
- Existing Q1–Q10: `[1, 1, 1, 1, 1, 1, 0.5, 1, 'NA', 1]`. Existing rationale: Q7 limited; Q10 repeated estimation runs.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S19 — Boundary-Aware Uncertainty for Feature Attribution Explainers
- Record: `C049`; source: `Assignment 2/work/extract/ft_003.jsonl:6`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2210.02419
- Finding: Primary abstract supports decision-boundary-aware GPEC uncertainty. Existing Q10 note explicitly refers to sections read, so scores are not complete-paper verified.
- Existing Q1–Q10: `[1, 1, 1, 1, 0.5, 0.5, 1, 1, 'NA', 0.5]`. Existing rationale: Q5 uncertainty estimate rather than a stability metric; Q6 largely visual comparisons; Q10 seeds not reported in sections read.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S20 — Influence of Data Characteristics on Machine Learning Classification Performance and Stability of SHapley Additive exPlanations
- Record: `U490`; source: `Assignment 2/work/extract/ft_001.jsonl:32`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://www.esann.org/sites/default/files/proceedings/2024/ES2024-107.pdf
- Finding: Primary proceedings abstract supports training/background data size and imbalance link. Detailed sample count and QA not fully checked.
- Existing Q1–Q10: `[1, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1]`. Existing rationale: Q4 three models but no stability baseline; Q5 variance only; Q8 code, no data; Q9 health data, ethics not discussed; Q10 cross-validation folds.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S21 — Shaping Up SHAP: Enhancing Stability through Layer-Wise Neighbor Selection
- Record: `U546`; source: `Assignment 2/work/extract/ft_002.jsonl:5`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2312.12115
- Finding: Primary abstract supports stochastic neighbor-selection cause and fully stable ST-SHAP for repeat executions. Do not interpret full stability as guarantee under changed input/model.
- Existing Q1–Q10: `[1, 1, 1, 1, 1, 1, 0.5, 1, 'NA', 1]`. Existing rationale: Q7 limited; Q10 repeated executions.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S22 — A Methodology for Reliability Analysis of Explainable Machine Learning: Application to Endocrinology Diseases
- Record: `C033`; source: `Assignment 2/work/extract/ft_004.jsonl:4`.
- Inspection depth: No primary paper inspection.
- Coverage: Primary paper not inspected; unverified.
- Source attempted/checked: https://doi.org/10.1109/access.2024.3431691
- Finding: DOI full-text route inaccessible. Original Q3 says dataset sizes not checked and Q8 not verified: these remain outstanding and cannot be treated as evidence-backed scores.
- Existing Q1–Q10: `[1, 1, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, 1]`. Existing rationale: Q3 dataset sizes not checked; Q6 two case studies; Q8 not verified; Q9 health data; Q10 multiple k values.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S23 — Analyzing Explainer Robustness via Probabilistic Lipschitzness of Prediction Functions
- Record: `C048`; source: `Assignment 2/work/extract/ft_003.jsonl:5`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2206.12481
- Finding: Primary abstract supports predictor probabilistic-Lipschitzness and explainer-astuteness connection. Exact experiment rankings and Q10 need full recheck.
- Existing Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 0.5, 'NA', 1]`. Existing rationale: Q8 libraries public, no paper code; Q10 20 explainer runs.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S24 — Calibrated Explanations: with Uncertainty Information and Counterfactuals
- Record: `C045`; source: `Assignment 2/work/extract/ft_003.jsonl:4`.
- Inspection depth: Targeted sections, not complete full text.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2305.02305
- Finding: Primary Secs3.2 and6.2 describe CE as distinct from SHAP/LIME with conceptual characteristic comparison. Numerical stability results concern CE. Scope concern under IC2: explain why conceptual comparison is sufficient or reconsider include; no automatic recoding.
- Existing Q1–Q10: `[1, 1, 1, 0.5, 0.5, 1, 0.5, 1, 'NA', 1]`. Existing rationale: Q4 no competing explainer in stability tests; Q5 variance only; Q7 brief; Q10 30 runs per dataset.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S25 — Combination of Weak Learners eXplanations to Improve Random Forest eXplicability Robustness
- Record: `U159`; source: `Assignment 2/work/extract/ft_001.jsonl:10`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2402.19025
- Finding: Primary abstract supports discriminative averaging of random-forest SHAP explanations. Exact split-divergence claim and quantitative experiment details require further check.
- Existing Q1–Q10: `[1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 'NA', 0.5]`. Existing rationale: Q5 Lipschitz units hard to compare; Q6 small-scale; Q8 no code; Q10 sampling of neighbourhoods described but no repeated seeds.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S26 — How Reliable and Stable are Explanations of XAI Methods?
- Record: `U560`; source: `Assignment 2/work/extract/ft_002.jsonl:8`.
- Inspection depth: Title and contents only; no substantive results checked.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2407.03108
- Finding: Primary contents identifies perturbation stability experiments and conclusion. Q7 note says none explicit despite 0.5 score: evidence rationale requires clarification. Findings/QA not fully revalidated.
- Existing Q1–Q10: `[1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 'NA', 0.5]`. Existing rationale: Q2/Q3 limited detail; Q6 descriptive; Q7 none explicit; Q10 perturbation levels but no repetitions reported.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S27 — Probabilistic Lipschitzness and the Stable Rank for Comparing Explanation Models
- Record: `U557`; source: `Assignment 2/work/extract/ft_002.jsonl:7`.
- Inspection depth: Title and contents only; no substantive results checked.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2402.18863
- Finding: Primary contents includes normalised astuteness and stable-rank experiments. Recheck title/abstract scope and exact comparative rankings before using quantitative synthesis.
- Existing Q1–Q10: `[1, 1, 0.5, 1, 1, 0.5, 1, 0, 'NA', 0.5]`. Existing rationale: Q3 toy datasets; Q6 limited empirical scope; Q8 none; Q10 sampling not detailed.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S28 — Comparative Analysis of Perturbation Techniques in LIME for Intrusion Detection Enhancement
- Record: `U176`; source: `Assignment 2/work/extract/ft_005.jsonl:3`.
- Inspection depth: No primary paper inspection.
- Coverage: Primary paper not inspected; unverified.
- Source attempted/checked: https://doi.org/10.3390/make7010021
- Finding: DOI route unavailable. Dataset and exact R2/stability findings remain unverified against primary full text.
- Existing Q1–Q10: `[1, 1, 1, 1, 1, 0.5, 0.5, 1, 'NA', 1]`. Existing rationale: Q6 one dataset; Q7 brief; Q10 repeated LIME runs.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S29 — Evaluating the stability of model explanations in instance-dependent cost-sensitive credit scoring
- Record: `U291`; source: `Assignment 2/work/extract/ft_001.jsonl:1`.
- Inspection depth: Targeted sections, not complete full text.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2509.01409
- Finding: Primary Sec4.2 supports main class-imbalance conclusion but also reports exceptions/nonmonotonic SRA and GMSC rebalance effects. Avoid implying uniform monotonic decline in every model/dataset/metric.
- Existing Q1–Q10: `[1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]`. Existing rationale: Q7 future-work only; Q8 data public, code not linked; Q9 regulatory context (GDPR/AI Act) but no privacy analysis; Q10 25 iterations per rate but seeds and LIME randomness not separated.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S30 — SIM-Shapley: A Stable and Computationally Efficient Approach to Shapley Value Approximation
- Record: `U487`; source: `Assignment 2/work/extract/ft_001.jsonl:29`.
- Inspection depth: Title and contents only; no substantive results checked.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2505.08198
- Finding: Primary full-text outline supports SIM-Shapley variance and consistency analysis. Exact 85% speed-up and each QA item not independently checked.
- Existing Q1–Q10: `[1, 1, 0.5, 1, 1, 1, 1, 1, 'NA', 1]`. Existing rationale: Q3 datasets named, sizes in appendix; Q10 repeated runs measure estimator variance.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S31 — Statistical Significance of Feature Importance Rankings
- Record: `U491`; source: `Assignment 2/work/extract/ft_001.jsonl:33`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2401.15800
- Finding: Primary abstract supports hypothesis-testing guarantees for top-K SHAP/LIME rankings. Exact FWER experiments and Q3/Q7 ratings not fully verified.
- Existing Q1–Q10: `[1, 1, 0.5, 1, 1, 1, 0.5, 1, 'NA', 1]`. Existing rationale: Q3 dataset details in appendix; Q7 brief; Q10 repeated replicates with error-rate control.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S32 — Explainable AI for Forensic Analysis: A Comparative Study of SHAP and LIME in Intrusion Detection Models
- Record: `U311`; source: `Assignment 2/work/extract/ft_005.jsonl:7`.
- Inspection depth: No primary paper inspection.
- Coverage: Primary paper not inspected; unverified.
- Source attempted/checked: https://doi.org/10.3390/app15137329
- Finding: Publisher/DOI returned 429. XGBoost 97.8% accuracy and explanation-stability judgment remain unverified.
- Existing Q1–Q10: `[1, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]`. Existing rationale: Q2/Q5 stability procedure brief; Q6 descriptive; Q9 legal defensibility discussed; Q10 not reported.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S33 — Which LIME should I trust? Concepts, Challenges, and Solutions
- Record: `C024`; source: `Assignment 2/work/extract/ft_003.jsonl:2`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/html/2503.24365
- Finding: Primary abstract supports LIME survey, taxonomy and public companion website. Remaining conclusion details and review-quality scores not fully rechecked.
- Existing Q1–Q10: `[1, 0.5, 'NA', 'NA', 'NA', 1, 1, 1, 'NA', 'NA']`. Existing rationale: Review: dataset/baseline/metric/randomness items N/A; Q2 selection criteria described but no counts.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S34 — SHAP Stability in Credit Risk Management: A Case Study in Credit Card Default Model
- Record: `U126`; source: `Assignment 2/work/extract/ft_001.jsonl:7`.
- Inspection depth: Abstract/introduction-level excerpts only; detailed extraction unverified.
- Coverage: Primary excerpts checked; partial verification only.
- Source attempted/checked: https://arxiv.org/pdf/2508.01851
- Finding: Primary PDF abstract supports dependence of SHAP consistency on feature importance. Exact KS/W numbers and complete QA require further checking.
- Existing Q1–Q10: `[1, 1, 1, 0.5, 1, 1, 1, 0.5, 0.5, 1]`. Existing rationale: Q4 no comparison explainer or model; Q7 explicit limitations; Q8 data public, code not linked; Q9 regulatory (ECOA/FCRA) framing; Q10 100 seeds.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

### S35 — Bayesian-AIME: Quantifying Uncertainty and Enhancing Stability in Approximate Inverse Model Explanations
- Record: `U042`; source: `Assignment 2/work/extract/ft_004.jsonl:1`.
- Inspection depth: No primary paper inspection.
- Coverage: Primary paper not inspected; unverified; author software checked.
- Source attempted/checked: https://github.com/ntakafumi/aime
- Finding: Paper DOI route inaccessible. Author repository ntakafumi/aime explicitly provides BayesianAIME; reproducibility none-found is outdated/incomplete as of audit. Paper results and dataset sizes remain unverified.
- Existing Q1–Q10: `[1, 1, 0.5, 1, 1, 1, 0.5, 0, 'NA', 1]`. Existing rationale: Q3 dataset sizes not checked; Q7 method limitations only; Q8 none found; Q10 50 bootstrap resamples.
- Score disposition: retain pending a complete evidence appraisal; the existing numeric score is not independently certified by this audit.

## Remaining work

Perform full-paper checks of every still-unverified extraction field and each quality item, write item-specific evidence notes, and reconcile the two human screening columns separately. Access failures in this audit are not evidence a paper does not exist or is permanently unavailable. Full-text author/publisher access may resolve them.