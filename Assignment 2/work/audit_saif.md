# AI source audit: S36–S79

Prepared 6 November 2026 by the AI agent named `saif`. This is not work performed by the human Saif Shahzad and is not independent human screening. Human screening cells and canonical data were not edited.

## Scope and limits

All 44 assigned records were inspected for auditability. 20 had targeted full-text passages checked (10 local extracted texts, 10 remote primary full texts); 8 had abstract-only corroboration; 16 had no accessible full text after primary-link attempts. Targeted checks are not exhaustive re-reading or an independent rescore of all 440 quality items. No study can be labelled fully human-verified from this audit. Access failures do not establish nonexistence or inaccuracy. Metadata, paper claims and actual code execution are different evidence levels.

The existing master records and original JSONL extracts were compared; original record locations are listed below. QA vectors shown are existing values, not a new endorsement. Original AI-generated text is never treated as corroborating source evidence.

## Actionable findings

1. **S36:** separate bootstrap performance tests from Wilcoxon explanation tests on 100 instances.
2. **S72:** correct the claim that repetitions are not detailed; 20 runs/20 seeds are explicit. Score change requires adjudication.
3. **S73:** record the explicit repository URL from footnote 9; the repository itself was not tested.
4. **S67:** replace the erroneous composite-stability note with the actual pairwise top-10 overlap protocol, preserving the score pending review.
5. **S78:** add the authors’ limitations and the source’s internal numeric inconsistencies. Keep disputed source numbers visible rather than silently repairing the source.
6. **S43:** explicitly discuss its tree-only split-frequency screen. Its target is global ranking flips across retrained models, not instance-level explanation change under prediction-preserving input perturbations. The proposed research gap should be narrowed to that difference. The screen’s validation uses precision/recall, not AUROC. Do not claim tree-only diagnostics do not exist.
7. **S49 and S75:** distinguish cross-method disagreement from within-method variation across models; reconcile borderline inclusion with EC3.
8. **S77:** retain the out-of-sample condition; the paper reports an in-sample counterexample.

## Per-study coverage

### S36: Evaluating XAI techniques under class imbalance using CPRD data

- Status: **Targeted full text**.
- Source: Publisher full text §§2, 3.1, 3.2.1. [Primary source](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1682919/full).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:14 (uid `U124`)`.
- Finding: Inclusion supported: CPRD tabular data and LIME/SHAP consistency. Main imbalance finding supported. Confirmed evaluation error: performance uses 100 bootstrap samples; explanation Wilcoxon tests use 100 test instances. Q10 rationale mentions bootstraps rather than explicit seed control; do not equate repetitions with control.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 0.5, 1, 1]`. Existing notes: Q8 restricted data, no code; Q9 anonymised data, licence noted; Q10 100 bootstraps.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S37: Integrating Shapley Value and Least Core Attribution for Robust Explainable AI in Rent Prediction

- Status: **Full text unavailable; source claims unverified**.
- Source: Publisher URL attempted; 429 access failure. [Primary source](https://doi.org/10.3390/buildings15173133).
- Original record: `Assignment 2/work/extract/ft_005.jsonl:10 (uid `U340`)`.
- Finding: Full-text inclusion, consistency metric, findings and QA remain unverified. No conclusion that the paper is false follows from access failure.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 0.5, 0.5, 0.5, 0, "NA", 0.5]`. Existing notes: Q5 consistency metric brief; Q6 one dataset; Q8 none; Q10 repetitions unclear.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S38: Developing a Hybrid Model to Enhance the Robustness of Interpretability for Landslide Susceptibility Assessment

- Status: **Full text unavailable; source claims unverified**.
- Source: Publisher URL attempted; 429 access failure. [Primary source](https://doi.org/10.3390/ijgi14070277).
- Original record: `Assignment 2/work/extract/ft_005.jsonl:2 (uid `U148`)`.
- Finding: The quoted CV values, repeated-fold design and Q1–Q10 cannot be checked without the source. Retain unverified status.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, "NA", 1]`. Existing notes: Q5 CV only; Q6 single study area; Q7 brief; Q8 no code; Q10 10 folds.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S39: On the Stability of Local Interpretable Model-Agnostic Explanations for Quantum Classifiers

- Status: **Targeted full text**.
- Source: Publisher HTML §§4, 5.2, 5.5–5.6. [Primary source](https://link.springer.com/article/10.1007/s10994-026-07071-5).
- Original record: `Assignment 2/work/extract/ft_006.jsonl:1 (uid `U502`)`.
- Finding: Tabular datasets and repeated LIME stability evaluation supported; §5.2 reports 23,400 calls over configurations. Quantum ideal/noisy environment finding is supported at a broad level. Current Q4/Q7 are judgments, not independently fully rescored.
- Existing QA Q1–Q10: `[1, 1, 1, 0.5, 1, 1, 0.5, 1, "NA", 1]`. Existing notes: Q4 no competing explainer; Q7 brief; Q10 repeated LIME calls.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S40: Calibrated and Explainable Gradient Boosting for Road Traffic Crash Severity Prediction: SHAP Audit and Cross-Jurisdiction Transfer Evaluation

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; 429 access failure. [Primary source](https://doi.org/10.3390/app16125876).
- Original record: `Assignment 2/work/extract/ft_005.jsonl:5 (uid `U183`)`.
- Finding: Perturbation failure percentage and code release are unverified. Especially check Q8=1 and Q10=1: preregistered audit criteria alone do not establish randomness control.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 1, "NA", 1]`. Existing notes: Q7 negative results reported; Q8 full release; Q10 audit criteria pre-specified.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S41: Class Imbalance Strategies, Calibration, and Explanation Stability in Tree-Based Intrusion Detection for IoT Networks

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; inaccessible. [Primary source](https://doi.org/10.65069/jessd21202623).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:26 (uid `U356`)`.
- Finding: Friedman result, Spearman figure, ten-seed protocol and all quality items remain source-unverified.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 0.5, "NA", 1]`. Existing notes: Q8 data public, no code; Q10 ten seeds.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S42: The Explainability–Reliability Gap in Fraud Detection: Evidence from SHAP and Permutation Importance Under Distribution Shift

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; inaccessible. [Primary source](https://doi.org/10.3390/fintech5030077).
- Original record: `Assignment 2/work/extract/ft_005.jsonl:4 (uid `U123`)`.
- Finding: Check whether the paper measures within-explainer stability separately from SHAP–permutation disagreement; current finding mixes them. No exclusion proposed without source.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 1]`. Existing notes: Q7 brief; Q8 on request; Q9 financial data, synthetic suite; Q10 repeated permutations and retraining.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S43: The Attribution Impossibility: No Feature Ranking Is Faithful, Stable, and Complete Under Collinearity

- Status: **Targeted full text**.
- Source: arXiv HTML v1 §§9.2, 10.16, Table 8. [Primary source](https://arxiv.org/html/2605.21492v1).
- Original record: `Assignment 2/work/extract/ft_002.jsonl:2 (uid `U521`)`.
- Finding: 68% of 77 datasets headline supported. Important omission: split-frequency diagnostic computed from one trained tree ensemble is validated against ranking flips under retraining. This is relevant prior tree-only diagnostic work. Different target from local perturbation instability; precision/recall are not AUROC. Diagnostic map and gap need explicit qualification. Underlying theoretical claims not formally validated in this audit.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 1, "NA", 1]`. Existing notes: Q10 seeds/trials and CIs reported.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S44: TreeSHAP-Importance-Weighted Feature-Group Fusion for Tabular Regression: Centered Additive Decomposition, Rade-Macher Complexity Control, and Attribution Stability

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; 429 access failure. [Primary source](https://doi.org/10.3390/math14132419).
- Original record: `Assignment 2/work/extract/ft_005.jsonl:9 (uid `U327`)`.
- Finding: Lipschitz guarantee, 20 seeds and SWFF-R result unverified. Distinguish a mathematical bound from an empirical stability metric before rescoring Q5.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 0.5, 1, 0.5, 0.5, "NA", 1]`. Existing notes: Q5 theoretical bound; Q7 brief; Q8 no code; Q10 20 seeds.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S45: Assessing Alignment and Stability of Feature Importance Explanations via Weight of Evidence

- Status: **Targeted full text**.
- Source: arXiv HTML v2 §§5.1, 5.4, 6. [Primary source](https://arxiv.org/html/2609.00090v2).
- Original record: `Assignment 2/work/extract/ft_002.jsonl:4 (uid `U528`)`.
- Finding: Titanic SHAP WoE counts (23 positive infinity, 27 negative infinity) and 80% CV range supported. Results depend on hypothesis, dataset and implementation; do not generalize SHAP determinism. arXiv metadata now notes workshop acceptance; publication status update needs venue verification and a documented update date.
- Existing QA Q1–Q10: `[1, 0.5, 0.5, 0.5, 1, 0.5, 1, 0, "NA", 1]`. Existing notes: Q2/Q3 details deferred; Q4 limited baselines; Q6 small experiments; Q8 none.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S46: Measuring Explainer Stability via Attribution Separability

- Status: **Targeted full text**.
- Source: arXiv HTML v1 §5 and §5.1. [Primary source](https://arxiv.org/html/2608.02697v1).
- Original record: `Assignment 2/work/extract/ft_002.jsonl:3 (uid `U524`)`.
- Finding: 50 runs on 50 instances, near-determinism on Churn/Diabetes, and promise to release code upon acceptance are supported. A v2 dated 31 Oct 2026 exists after original 24–25 Oct search: do not silently change version. Current original-version code score remains defensible.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 0, "NA", 1]`. Existing notes: Q8 code not yet released; Q10 50 runs per instance.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S47: UbiQTree: Uncertainty quantification in XAI with tree ensembles

- Status: **Targeted full text**.
- Source: arXiv HTML v2 summary, introduction, reproductive details. [Primary source](https://arxiv.org/html/2508.09639v2).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:27 (uid `U409`)`.
- Finding: Source explicitly links complete code and reports fixed random_state=42 in experimental details; supports Q8 and part of Q10. Uncertainty decomposition is supported. Q4/Q7/Q9 not fully reappraised; do not treat their provisional scores as verified.
- Existing QA Q1–Q10: `[1, 1, 1, 0.5, 1, 1, 0.5, 1, 0.5, 1]`. Existing notes: Q4 no competing UQ baseline reported in sections read; Q9 public health data, ethics brief; Q10 sampling-based.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S48: Reliability-calibrated explainable machine learning for scientific interpretation under spatial, resampling, and predictor-correlation uncertainty

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; inaccessible. [Primary source](https://doi.org/10.1088/2632-2153/ae8a13).
- Original record: `Assignment 2/work/extract/ft_006.jsonl:2 (uid `U288`)`.
- Finding: RPI recovery, spatial-transfer finding, bootstrap counts and supplementary material cannot be checked.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 0.5, "NA", 1]`. Existing notes: Q8 supplement only; Q10 bootstrap replicates.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S49: Evaluation of Rashomon Sets for the Determination of Stable and Plausible Model Explanations

- Status: **Targeted full text**.
- Source: Publisher proceedings PDF §§3–5, Table 1 (pp.147–149). [Primary source](https://doi.org/10.14428/esann/2026.es2026-250).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:35 (uid `U510`)`.
- Finding: Three tabular datasets, exact Shapley profiles and 100 trained models are explicit. ARI .26/.092/.07 compares clusters obtained from two importance-profile methods. Extraction should distinguish low cross-method cluster agreement from across-model stability; EC3 needs careful adjudication because Rashomon variability is present but headline ARIs are cross-method.
- Existing QA Q1–Q10: `[1, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5, "NA", 1]`. Existing notes: Q2 brief (6-page format); Q4 no external baseline; Q6 small experiments; Q8 no code; Q10 repeated training runs.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S50: An Empirical Evaluation of Explainability Drift Under Post-Training Quantization of Machine Learning Models

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; inaccessible. [Primary source](https://doi.org/10.7759/s44389-026-00100-3).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:13 (uid `U221`)`.
- Finding: INT4/global/local drift and single-model design unverified. Keep prediction-performance/explanation distinction source-conditional.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 0.5, "NA", 0.5]`. Existing notes: Q7 explicit limitations; Q8 data only; Q10 single trained model per family, no seeds reported.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S51: Reliability auditing of explanations for machine-learning-based intrusion detection systems

- Status: **Targeted full text**.
- Source: Publisher HTML §§3.9, 6.1.3.2, 6.2–6.3. [Primary source](https://link.springer.com/article/10.1007/s11416-026-00664-7).
- Original record: `Assignment 2/work/extract/ft_006.jsonl:5 (uid `U236`)`.
- Finding: Main comparison supported: logistic regression has strongest retraining explanation stability with weaker detection, while ensemble tradeoffs vary by dataset. Do not infer universal accuracy–stability tradeoff or a tested instance-level prediction-change relation.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 0.5, 0.5, "NA", 0.5]`. Existing notes: Q7 brief; Q8 data only; Q10 three seeds only.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S52: Uncertainty Gating for Cost-Aware Explainable Artificial Intelligence

- Status: **Targeted full text**.
- Source: arXiv HTML v1 abstract, §5.2 and code footnote. [Primary source](https://arxiv.org/html/2603.29915v1).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:22 (uid `U195`)`.
- Finding: Four tabular datasets and negative uncertainty–stability relation supported. §5.2 specifies fixed-seed shared test subset and perturbation grids. Code URL stated. No all-QA endorsement; dataset-noise dependence and surrogate uncertainty for some models limit generalization.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 1, "NA", 1]`. Existing notes: Q8 code public; Q10 perturbation types and budgets defined.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S53: Stability of SHAP-Based Feature Importance Ranking under Class Imbalance, Feature Correlation, and Feature Dimensionality

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; inaccessible. [Primary source](https://doi.org/10.32665/statkom.v5i1.6455).
- Original record: `Assignment 2/work/extract/ft_006.jsonl:10 (uid `U226`)`.
- Finding: Correlation, imbalance effects and repeated-fit protocol unverified. Check ethics evidence and code availability when full text is obtained.
- Existing QA Q1–Q10: `[1, 1, 1, 0.5, 1, 1, 0.5, 0.5, 0.5, 1]`. Existing notes: Q4 two models only; Q8 no code; Q9 anonymised health data; Q10 repeated fits with fixed seeds.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S54: The Stability Collapse in Local Explainability: Benchmarking H-LIME on High-Dimensional Data

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; inaccessible. [Primary source](https://doi.org/10.1109/access.2026.3709805).
- Original record: `Assignment 2/work/extract/ft_004.jsonl:3 (uid `U499`)`.
- Finding: Correlations .87/.35 and H-LIME improvement unverified. Existing Q8 note itself says not verified: preserve uncertainty, not a claim of source availability.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1]`. Existing notes: Q6 inference from SD intervals; Q8 not verified; Q9 student data, ethics not seen; Q10 repeated runs.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S55: The Effect of Data Imbalance on the Interpretation Stability of LIME-Based Explainable AI on Nutritional Status Prediction Models

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; inaccessible. [Primary source](https://doi.org/10.66472/iji.v1i2.438).
- Original record: `Assignment 2/work/extract/ft_006.jsonl:11 (uid `U224`)`.
- Finding: Stability .80 and absence of baseline/repetitions remain unverified. Existing low score is not proof of correct assessment.
- Existing QA Q1–Q10: `[1, 0.5, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0]`. Existing notes: Q4 no baseline; Q6 weak; Q8 none; Q9 child health data, ethics not discussed; Q10 no repetitions.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S56: Stable and Faithful Explanations for Knowledge Tracing

- Status: **Abstract only; full text unverified**.
- Source: arXiv abstract only. [Primary source](https://arxiv.org/abs/2609.28502).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:17 (uid `U326`)`.
- Finding: Tabular engineered features and TreeSHAP study supported at abstract level; rebuilt dataset/leakage correction is prominent. Full numerical stability, code and all-QA appraisal remain unverified. Preserve exact version before changing extraction.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1]`. Existing notes: Q9 ethics statements present (not applicable); Q10 seeds and folds reported.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S57: A Multi-World Synthetic Benchmark for Evaluating RSCE in Clinical Machine Learning

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; inaccessible. [Primary source](https://doi.org/10.1109/access.2026.3666756).
- Original record: `Assignment 2/work/extract/ft_004.jsonl:2 (uid `U045`)`.
- Finding: RSCE link to degradation and bootstrap claims unverified. Q9 explicitly says ethics detail not seen: no completed ethics verification.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1]`. Existing notes: Q9 clinical data use without ethics detail in sections read; Q10 configurations and bootstrap CIs.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S58: Multi-Algorithm Optimization of Explanation Stability in Ensemble Learning for Human Capital Attrition Risk Prediction

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; inaccessible. [Primary source](https://doi.org/10.3390/math14183283).
- Original record: `Assignment 2/work/extract/ft_005.jsonl:11 (uid `U355`)`.
- Finding: Variance/stability distinction and optimizer comparison unverified. Q8 code not seen remains uncertain.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 1]`. Existing notes: Q8 data public, code not seen; Q9 HR data context; Q10 bootstrap refits.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S59: Beyond Gradient-Based Attacks: Adversarial Robustness and Explainability Stability in Cybersecurity Classifiers

- Status: **Abstract only; full text unverified**.
- Source: arXiv abstract only. [Primary source](https://arxiv.org/abs/2607.01679).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:9 (uid `U163`)`.
- Finding: ZOO/XGBoost degeneracy, Square Attack vulnerability and RI/ESI values directly supported in abstract. Full settings and QA remain unverified.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 0.5, 1, "NA", 0.5]`. Existing notes: Q7 attack realism discussed; Q8 Zenodo; Q9 no personal data; Q10 repeated runs not clearly reported.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S60: Empirical Analysis of Adversarial Robustness and Explainability Drift in Cybersecurity Classifiers

- Status: **Abstract only; full text unverified**.
- Source: arXiv abstract and metadata only. [Primary source](https://arxiv.org/abs/2602.06395).
- Original record: `Assignment 2/work/extract/ft_003.jsonl:1 (uid `C013`)`.
- Finding: MLP predecessor, FGSM/PGD, SHAP drift and up-to-9% RI gain supported at abstract level. Do not pool as independent replication of S59 without noting shared authors/study lineage. Conference acceptance self-reported; final proceedings not checked.
- Existing QA Q1–Q10: `[1, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5, "NA", 0.5]`. Existing notes: Q2 architecture details brief; Q4 no explainer comparison; Q6 descriptive plots; Q8 data public, no code; Q10 fixed seed, no repetitions.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S61: Explainable AI for Institutional Fraud Decisions: A Cross-Sector Empirical Study Using Public Healthcare and Financial Transaction Data

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; inaccessible. [Primary source](https://doi.org/10.32996/jcsts.2025.8.1.7).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:15 (uid `U233`)`.
- Finding: Cross-sector stability finding, publication-date discrepancy in DOI versus year, and QA remain unverified.
- Existing QA Q1–Q10: `[1, 0.5, 1, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5]`. Existing notes: Q2 limited detail; Q4 no explainer comparison; Q5 single metric; Q6 descriptive; Q9 governance framing; Q10 fixed seed only.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S62: Beyond Detection Accuracy: Measuring Explanation Cost, Stability, and Utility for Resource-Aware IoT Intrusion Detection

- Status: **Abstract only; full text unverified**.
- Source: arXiv abstract only. [Primary source](https://arxiv.org/abs/2608.10349).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:12 (uid `U336`)`.
- Finding: Prediction-preserving perturbations, listed classifiers and TreeSHAP times supported. Abstract does not establish all QA scores or exact local perturbation protocol.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 0.5, 1, "NA", 1]`. Existing notes: Q8 Zenodo archive; Q10 prediction-preserving filter and deterministic partitioning.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S63: Interpretable Machine Learning for Football Performance Analysis: Evidence of Limited Transferability from Elite Leagues to University Competition

- Status: **Abstract only; full text unverified**.
- Source: arXiv abstract only. [Primary source](https://arxiv.org/abs/2605.10796).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:23 (uid `U248`)`.
- Finding: Elite-to-university shift and qualitative SHAP instability supported. Quoted seed-wise numeric results, restricted-data and privacy claims require full text.
- Existing QA Q1–Q10: `[1, 1, 1, 0.5, 1, 1, 0.5, 0.5, 1, 1]`. Existing notes: Q4 two models, no explainer baseline beyond CIS; Q7 limited; Q8 partial data; Q9 privacy reason for withholding data; Q10 seeds listed.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S64: Improving LIME Stability via Density-Awareness: Evaluation and Comparison of AKDE-LIME

- Status: **Full text unavailable; source claims unverified**.
- Source: Europe PMC URL and DOI attempted; inaccessible. [Primary source](https://doi.org/10.1080/08839514.2026.2640686).
- Original record: `Assignment 2/work/extract/ft_006.jsonl:6 (uid `U141`)`.
- Finding: AKDE-LIME superiority, statistics and metric orientation unverified. No full-text corroboration obtained.
- Existing QA Q1–Q10: `[1, 1, 0.5, 1, 1, 0.5, 1, 0.5, "NA", 1]`. Existing notes: Q3 single dataset; Q6 one dataset; Q8 data only; Q10 noise grid and repeated instances.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S65: Measuring the Fragility of Trust: Devising Credibility Index via Explanation Stability (CIES) for Business Decision Support Systems

- Status: **Abstract only; full text unverified**.
- Source: arXiv abstract only. [Primary source](https://arxiv.org/abs/2603.05024).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:8 (uid `U105`)`.
- Finding: CIES purpose and p<.01 across 24 configurations supported. Inferential details and Q1–Q10 not checked.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]`. Existing notes: Q7 robustness checks but limited threats discussion; Q8 no code; Q9 AI Act/GDPR context only; Q10 noise draws described, seeds not reported.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S66: Stabilising Explainability Fragility in Cybersecurity AI: The Impact and Mitigation of Multicollinearity in Public Benchmark Datasets

- Status: **Abstract only; full text unverified**.
- Source: arXiv abstract only. [Primary source](https://arxiv.org/abs/2605.22529).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:11 (uid `U077`)`.
- Finding: Collinearity mechanism, UNSW-NB15, bootstrap Kendall ranking and SHARP/CAA methods supported at abstract level. Reproducibility and QA remain unverified.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 0.5, 0, "NA", 1]`. Existing notes: Q7 limited threats discussion; Q8 no code/data release beyond public benchmark; Q10 bootstrap repetitions.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S67: A multidimensional trustworthiness evaluation framework for AI-enabled clinical decision support: a HER2 breast cancer case study

- Status: **Targeted full text**.
- Source: Publisher HTML Experimental methodology → Explanation stability; Table 7. [Primary source](https://link.springer.com/article/10.1186/s43093-026-01017-y).
- Original record: `Assignment 2/work/extract/ft_006.jsonl:4 (uid `U210`)`.
- Finding: Headline ROC-AUC and stability supported. Confirmed note error: stability is separate mean pairwise top-10 LIME feature-set overlap, not just an unspecified composite. It compares 100 evaluation-instance explanations per repetition; do not describe as same-instance repeated-call stability. Keep Q5 score pending methodological adjudication.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 0.5, 0.5, 1, 0.5, 0.5, 1]`. Existing notes: Q5 composite stability score; Q6 single case study; Q9 clinical data, ethics brief; Q10 30 runs.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S68: RoSHAP: A Distributional Framework and Robust Metric for Stable Feature Attribution

- Status: **Abstract only; full text unverified**.
- Source: arXiv abstract only. [Primary source](https://arxiv.org/abs/2605.15154).
- Original record: `Assignment 2/work/extract/ft_002.jsonl:1 (uid `U519`)`.
- Finding: Bootstrap/KDE attribution distributions and RoSHAP ranking supported. Exact experiments, reproducibility and all QA unverified.
- Existing QA Q1–Q10: `[1, 1, 0.5, 0.5, 1, 1, 0.5, 1, "NA", 1]`. Existing notes: Q3 dataset details partly in appendix; Q4 compared with standard SHAP only; Q10 bootstrap replicates.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S69: Explainable Credit Underwriting on FICO HELOC: A Framework for Counterfactual Recourse under Profit, Fairness, and Stability Constraints

- Status: **Full text unavailable; source claims unverified**.
- Source: DOI attempted; inaccessible. [Primary source](https://doi.org/10.69987/aimlr.2026.70107).
- Original record: `Assignment 2/work/extract/ft_001.jsonl:16 (uid `U133`)`.
- Finding: Spearman/cosine numbers, five replicas, fairness analysis and uncertain peer-review status require source check.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 0.5, 1, 0.5, 1, 0.5]`. Existing notes: Q6 few replicas; Q8 no code link; Q9 fairness analysed; Q10 B=5 only.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S70: Can We Trust Your Explanations? Sanity Checks for Interpreters in Android Malware Analysis

- Status: **Targeted full text**.
- Source: Local work/ft_txt/N101.txt pp.4–9,13–16. [Primary source](https://arxiv.org/abs/2008.05895).
- Original record: `Assignment 2/work/extract/ft_007.jsonl:2 (uid `N101`)`.
- Finding: Tabular malware features and explicit within-method/model stability satisfy broad inclusion. Main LIME≥SHAP ranking and ~.85 supported p.9. Model constructions and parameters are given; absence of full seed control is not absence of repetition. Q8 remains unverified.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 0.5, "NA", 0.5]`. Existing notes: Q8 not verified; Q10 repetitions not fully described in sections read.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S71: Shapley variable importance cloud for machine learning models

- Status: **Targeted full text**.
- Source: Local work/ft_txt/N339.txt pp.3–8,10–11. [Primary source](https://arxiv.org/abs/2212.08370).
- Original record: `Assignment 2/work/extract/ft_007.jsonl:4 (uid `N339`)`.
- Finding: Rashomon-model SHAP importance variability and uncertainty intervals supported. p.7 explicitly names CIRB approval 2021/2122 and consent waiver; existing Q9=.5 with ethics-approval note deserves reconsideration, not automatic upgrade. No code release located.
- Existing QA Q1–Q10: `[1, 1, 1, 0.5, 0.5, 0.5, 1, 0, 0.5, 0.5]`. Existing notes: Q4 compared with single model only; Q5 interval-based; Q6 one dataset; Q8 none found; Q9 ethics approval context; Q10 sampling of models described, seeds not.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S72: Using Decision Tree as Local Interpretable Model in Autoencoder-based LIME

- Status: **Targeted full text**.
- Source: Local work/ft_txt/N029.txt pp.3–5,7. [Primary source](https://arxiv.org/abs/2204.03321).
- Original record: `Assignment 2/work/extract/ft_007.jsonl:1 (uid `N029`)`.
- Finding: Inclusion supported on four tabular datasets. Confirmed Q10 note error: p.4 specifies 20 runs with 20 different seeds; p.5 T=20. Stability formulas and code link exist. Recommend revising Q10 evidence and considering 1 instead of .5, with score recalculation only after adjudication.
- Existing QA Q1–Q10: `[1, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1, "NA", 0.5]`. Existing notes: Q2/Q3 brief; Q5 stability score loosely defined; Q6 small experiments; Q10 repetitions not detailed.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S73: The Distributional Uncertainty of the SHAP Score in Explainable Machine Learning

- Status: **Targeted full text**.
- Source: Local work/ft_txt/N708.txt pp.5–7. [Primary source](https://arxiv.org/abs/2401.12731).
- Original record: `Assignment 2/work/extract/ft_007.jsonl:10 (uid `N708`)`.
- Finding: Binary/product-distribution uncertainty and SHAP interval findings supported. p.6 explicitly samples five times. Confirmed reproducibility omission: p.5 footnote 9 gives fuzzyshapley code URL. Link presence verified in paper; repository accessibility and completeness not verified.
- Existing QA Q1–Q10: `[1, 1, 1, 0.5, 1, 0.5, 1, 0.5, "NA", 0.5]`. Existing notes: Q4 no explainer comparison; Q6 small case study; Q8 not verified; Q10 5 repetitions.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S74: Probabilistic Shapley Value Modeling and Inference

- Status: **Targeted full text**.
- Source: Local work/ft_txt/N693.txt §§4.1–4.3, pp.10–15. [Primary source](https://arxiv.org/abs/2402.04211).
- Original record: `Assignment 2/work/extract/ft_007.jsonl:9 (uid `N693`)`.
- Finding: PSI synthetic attribution-distribution and eight-real-dataset results supported; credible-interval illustration is in pp.10–11. It estimates attribution uncertainty rather than measuring local perturbation stability directly, so keep this distinction. Source has real tabular data plus images; not EC2. Q8 remains unverified.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 0.5, 1, 1, 0.5, "NA", 1]`. Existing notes: Q5 uncertainty rather than stability; Q8 not verified; Q10 standard errors reported.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S75: Hypothesis Class Determines Explanation: Why Accurate Models Disagree on Feature Attribution

- Status: **Targeted full text**.
- Source: Local work/ft_txt/N300.txt pp.6–9. [Primary source](https://arxiv.org/abs/2603.15821).
- Original record: `Assignment 2/work/extract/ft_007.jsonl:3 (uid `N300`)`.
- Finding: Within-/cross-class .1.000/.369 and dimensionality .287 figures supported. Outcome is disagreement between prediction-equivalent model classes, not local input perturbation. Since EC3 excludes cross-method agreement, explicitly distinguish cross-model SHAP variability. Diagnostic relevance supported; no external code release confirmed.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 0.5, "NA", 1]`. Existing notes: Q8 code snippets only; Q10 multiple seeds controlled.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S76: Enhancing LIME using Neural Decision Trees

- Status: **Targeted full text**.
- Source: Local work/ft_txt/N557.txt pp.4–5,11–13, Table 2. [Primary source](https://arxiv.org/abs/2603.20919).
- Original record: `Assignment 2/work/extract/ft_007.jsonl:7 (uid `N557`)`.
- Finding: Repeated-call stability and neighbor regularity are distinct metrics. Table2 supports a fidelity–stability tradeoff, not universal improvement: NDT-LIME Digits stability .816 versus .980 linear/.961 tree. Code URL given p.12. Existing finding is broadly fair but should explicitly avoid conflating neighbor regularity with repeated-call stability.
- Existing QA Q1–Q10: `[1, 1, 0.5, 1, 1, 1, 0.5, 1, "NA", 1]`. Existing notes: Q3 dataset details in tables; Q7 brief; Q10 repeated perturbation runs.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S77: How Understanding Forecast Uncertainty Resolves the Explainability Problem in Machine Learning Models

- Status: **Targeted full text**.
- Source: Local work/ft_txt/N616.txt pp.12–15,20–21,25–28. [Primary source](https://arxiv.org/abs/2602.00179).
- Original record: `Assignment 2/work/extract/ft_007.jsonl:8 (uid `N616`)`.
- Finding: OOS low-uncertainty/low-instability relation supported. p.21 explicitly shows in-sample instability can reverse that relation for wave-like generator; synthesis should retain OOS condition. Models include XGBoost, sigmoid NN and piecewise-linear regression tree. Synthetic-only evidence limits generality; code not found.
- Existing QA Q1–Q10: `[1, 1, 0.5, 1, 1, 0.5, 1, 0, "NA", 1]`. Existing notes: Q3 synthetic only; Q6 no real data; Q8 none; Q10 simulation design specified.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S78: Explainable Machine Learning for ICU Mortality Prediction: Comparing SHAP, LIME and Anchors on MIMIC-IV

- Status: **Targeted full text**.
- Source: Local work/ft_txt/N480.txt pp.1,5–6,9–11. [Primary source](https://doi.org/10.26438/ijcse.v14i5.7396).
- Original record: `Assignment 2/work/extract/ft_007.jsonl:6 (uid `N480`)`.
- Finding: Main TreeSHAP .276/.312/4ms finding matches abstract. Source itself contains conflicting Anchors AOPC values (Table6 .239 vs Table10/prose .148), and inconsistent CV best/worst ordering. Authors discuss three limitations pp.10–11, omitted from extraction. Paper claims code repository without identifiable URL; do not claim code available. Fixed seeds reported; repeated model runs not established.
- Existing QA Q1–Q10: `[1, 0.5, 1, 1, 1, 0.5, 0.5, 0, 0.5, 0.5]`. Existing notes: Q2 brief; Q6 single dataset; Q8 none; Q9 data-use agreement; Q10 fixed seeds only.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

### S79: DEM Resolution Drives AI Variability: Tradeoffs Between Performance and Explainability in Landslide Susceptibility Prediction

- Status: **Targeted full text**.
- Source: Local work/ft_txt/N383.txt pp.7–8,13–18,37. [Primary source](https://doi.org/10.1109/jstars.2026.3736041).
- Original record: `Assignment 2/work/extract/ft_007.jsonl:5 (uid `N383`)`.
- Finding: Gaussian-perturbation stability, ten seeds, performance–reliability mismatch and single-event/spatial-validation limitations supported. Code not located; appendix implementation details are present. No all-QA endorsement merely from long methods.
- Existing QA Q1–Q10: `[1, 1, 1, 1, 1, 1, 1, 0.5, "NA", 1]`. Existing notes: Q8 not verified; Q10 repeated seeds.
- QA audit limit: no full independent item-by-item rescore. Only specific evidence findings above can support a revision.

## Safe integration

Suggested field-level corrections are in `work/audit_saif/proposed_patches.json`. They are proposals, not applied changes. Apply factual corrections to original JSONL extraction data and its shipped mirror, then rebuild dependent master records/tables; preserve human workbook inputs. Keep the human contribution and dual-screening status pending until the named students actually complete them.

Versions matter: S46 v2 was released 31 October after the original 24–25 October search. This audit checked v1 for original extraction and reports newer metadata separately. Other online sources may have changed since extraction; preserve source/version dates when integrating changes.
