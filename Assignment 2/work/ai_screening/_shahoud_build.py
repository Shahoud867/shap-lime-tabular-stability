import json,re,html
from pathlib import Path
p=Path('Assignment 2/work/ai_screening')
records=json.loads((p/'records.json').read_text())
# Explicit per-record judgments made from scrubbed titles and stored abstracts only.
include={
'N420':'SHAP contributions are fused across bootstrap replicas into belief/plausibility intervals on three tabular datasets; explanation uncertainty is a direct outcome.',
'N421':'County-month tabular data and TreeSHAP are described; 500 Bayesian bootstrap iterations generate confidence intervals for SHAP importance as well as class probabilities.',
'N594':'SHAP-based manufacturing drivers are assessed under decision-threshold shifts and removal of operator identity, with stable top-ranked features reported; inspect stability procedure in full text.',
'N638':'Polymer descriptor models are evaluated for reproducibility of feature importance across multiple runs and consistency of SHAP values; potentially eligible stability evidence.',
'N687':'Experiments explicitly report that hierarchical shrinkage stabilizes random-forest SHAP values and decision boundaries; obtain full text to check the stability measure.',
'N708':'SHAP scores are varied over unknown population distributions, with tight score ranges and real-data experiments; direct attribution uncertainty study.',
'U001':'Repeated-sampling stability of SHAP/LIME top-k features is assessed with Jaccard similarity and Spearman correlation across classifiers; confirm tabular datasets in full text.',
'U008':'Symptom dataset is tabular; SHAP importance across RF/SVM/MLP is separately summarized with an Explainability Stability Score. Confirm this is model-variation stability rather than cross-explainer agreement.',
'U010':'Tabular network-traffic datasets and SHAP stability under input perturbations are explicit; RANK-ECI uses Kendall tau.',
'U019':'SHAP/LIME ranking stability is quantified under anonymization using rank correlations and hypothesis tests on real datasets.',
'U031':'Adult/COMPAS tabular tasks, repeated SHAP/LIME explanations and Jensen–Shannon consistency are explicit; attribution variability is evaluated separately from predictive accuracy.',
'U050':'Review of tabular interpretability explicitly compares SHAP/LIME robustness and stability tradeoffs; check whether its coverage satisfies the review eligibility rule.',
'U123':'SHAP feature rankings are assessed across biased/shifted fraud datasets separately from permutation-based reliance, providing direct distribution-shift stability evidence.',
'U126':'100 seeded XGBoost fits on tabular credit accounts are compared through SHAP rankings and Kendall W.',
'U163':'Four tabular security datasets and TreeSHAP attribution drift under attacks are evaluated using a separate Explainability Stability Index.',
'U175':'Benchmark of symbolic/tabular regression equations separately compares explainer robustness, explicitly identifying SHAP robustness; inspect evaluation metric in full text.',
'U192':'Gene-expression feature tables use outer-test SHAP and perturbation, with stability-weighted SHAP and feature nonselection across 25 models reported; assess direct attribution-stability measure in full text.',
'U224':'Tabular nutritional-status RF models under original/SMOTE balance are compared using LIME and an interpretation stability score of 0.80.',
'U226':'Simulated/claims tabular datasets, repeated XGBoost/LightGBM fits and SHAP Sequential Rank Agreement directly assess ranking stability.',
'U277':'Standard tabular datasets and SHAP attributions are tested under weight/input perturbations, with loss-landscape curvature related to explanation stability.',
'U302':'CIC-DDoS2019 feature tables use SHAP Feature Stability Score across folds, separately related to prediction robustness.',
'U481':'MIMIC-III background resampling explicitly assesses fluctuation of SHAP values and variable rankings as sample size changes.',
'U491':'Sampling-based top-K feature stability is the focus, with empirical SHAP/LIME validation and error guarantees; check datasets in full text.',
'U579':'Advertising classifiers use Shapley feature explanations; repeated-sampling instability and Monte Carlo sample requirements are directly studied.'}
ex={
'N005':('EC1','SHAP global/dependence/interaction interpretation is reported, while cross-model spatial consistency concerns predictions; no explanation-variation outcome is described.'),
'N061':('EC1','Results assess classification accuracy and cross-validation; LIME/SHAP identify features but no explanation stability outcome is described.'),
'N109':('EC1','Bootstrap intervals quantify prediction uncertainty; SHAP supplies feature weights, without uncertainty or stability of those weights.'),
'N153':('EC1','Stable performance follows class merging; SHAP identifies clinical drivers without a separate attribution-stability analysis.'),
'N167':('EC2','Title and supplied abstract explicitly concern NLP spam-message classification, not tabular model explanations.'),
'N199':('EC1','SHAP provides clinical insight and Bayesian uncertainty provides diagnosis confidence; no explanation-uncertainty or stability result is described.'),
'N221':('EC1','Posture classification accuracy/F1 and SHAP feature identification are reported, without explanation-stability measurement.'),
'N257':('EC1','Fraud model accuracy/F1/AUC and SHAP transparency are reported; explanation stability is not assessed in the supplied abstract.'),
'N282':('EC1','Split conformal intervals quantify quality-prediction uncertainty; SHAP supplies feature-interaction interpretations without attribution variability.'),
'N289':('EC3','Analyte importance is compared across SHAP, permutation and impurity importance; no within-explainer perturbation/repeated-run stability outcome is described.'),
'N292':('EC1','Uncertainty concerns crack-growth predictions and calibration; SHAP identifies feature effects, without attribution stability/uncertainty.'),
'N297':('EC1','Sensitivity analysis perturbs model hyperparameters and reports prediction metrics; SHAP gives one worked feature-contribution example.'),
'N306':('EC1','Study evaluates classifier performance and denoising; SHAP interpretation is proposed as future work rather than evaluated.'),
'N321':('EC1','Prediction reliability/uncertainty is evaluated and SHAP interprets material drivers; no explanation stability measurement is reported.'),
'N323':('EC1','Ablations and accuracy/F1 evaluate fault-diagnosis reliability; SHAP dependence plots interpret features without attribution stability.'),
'N327':('EC3','Framework uses intrinsic severity/uncertainty heads on PFN embeddings, not SHAP/LIME-family explanations.'),
'N331':('EC1','SHAP quantifies optical saturation and terrain contributions; stable physical contribution below biomass thresholds is not explanation perturbation stability.'),
'N335':('EC1','School-district classifiers are compared for accuracy and SHAP plots interpret features; no explanation-stability outcome is reported.'),
'N349':('EC1','Global soil-reflectance prediction/generalization is evaluated; stable baseline refers to soil spectra, with SHAP only interpreting covariates.'),
'N355':('EC1','GAM discrimination/calibration and MCDA patient rankings are assessed; SHAP confirms risk drivers without an attribution-stability outcome.'),
'N369':('EC5','Title explicitly concerns structural stability of circular/square tunnels in soils, not explanation stability.'),
'N375':('EC1','KDE intervals quantify lateral-pressure prediction uncertainty; SHAP interprets geotechnical inputs without attribution uncertainty.'),
'N376':('EC1','Stability selection concerns microstructure predictors; SHAP accompanies boosting but separate SHAP explanation stability is not described.'),
'N378':('EC1','Bootstrap uncertainty concerns concrete-strength prediction intervals; SHAP explains material factors without explanation uncertainty.'),
'N388':('EC3','Explanation uses decoder Jacobians, latent traversals and fuzzy rules; reported agreement concerns those methods rather than SHAP/LIME.'),
'N401':('EC3','Interpretability comparison corroborates SHAP with random-forest Gini ranking; no perturbation/repeated-run explanation stability is described.'),
'N402':('EC1','Confidence values refer to diabetes prediction and clinical risk stratification; SHAP identifies risk factors without attribution uncertainty.'),
'N408':('EC1','Chromosome instability and clinical survival are biological outcomes; SHAP is used for biomarker association, not explanation stability.'),
'N410':('EC1','Mondrian-forest uncertainty is correlated with prediction errors; Shapley values explain predictions without a separate attribution-uncertainty test.'),
'N413':('EC1','Repeated splits assess insurance prediction error and conformal coverage; SHAP identifies cost drivers, without explanation variability.'),
'N414':('EC1','Blind-well stability concerns model generalization and physical consistency; SHAP confirms physically meaningful predictors rather than attribution stability.'),
'N425':('EC1','Bayesian uncertainty and fairness concern clinical predictions; SHAP feature effects are reported without explanation-stability analysis.'),
'N426':('EC1','Seasonal prediction performance and turbidity uncertainty are assessed; SHAP river-discharge influence is not an explanation-stability measure.'),
'N432':('EC1','Conformal intervals quantify blood-glucose prediction uncertainty; SHAP interprets models but no attribution variation is assessed.'),
'N433':('EC1','SHAP locates a predictive income-rank discontinuity; robustness concerns social-dynamics discretization rather than explanation stability.'),
'N436':('EC1','Conformal calibration and test metrics assess life-expectancy predictions; SHAP explains policy drivers without measured attribution variability.'),
'N467':('EC1','PET/radiomic model transfer and adaptation are evaluated for prediction robustness; feature attribution is descriptive, without explanation stability.'),
'N471':('EC1','NGBoost supplies prediction uncertainty and SHAP interprets machining features; no explanation uncertainty or stability is separately assessed.'),
'N497':('EC1','Runoff decomposition/forecast robustness is studied; XGBoost-SHAP selects/interprets predictors without explanation-stability measurement.'),
'N514':('EC1','Fraud prediction metrics, cost and runtime are compared; SHAP provides transparency without attribution stability assessment.'),
'N519':('EC1','Bootstrap confidence intervals quantify extraction yield/optimal conditions; SHAP explains factor importance without attribution uncertainty.'),
'N522':('EC1','Bibliometric review maps RF/GEE publications and research trends; SHAP is only an emerging topic, not a reviewed explanation-stability outcome.'),
'N525':('EC1','Monte Carlo uncertainty quantifies predictive reliability as sample size increases; SHAP importance is not separately tested for stability.'),
'N534':('EC1','Perturbation tests evaluate loan prediction discrimination/calibration; post-hoc attribution is mentioned without explanation stability evidence.'),
'N537':('EC1','Ensemble CRPS and forecast uncertainty are evaluated; global/local SHAP explains forecasts without attribution stability/uncertainty.'),
'N545':('EC1','Consistency/CV measures manufacturing load robustness, and SHAP/PDP informs design; no explanation-stability measurement is reported.'),
'N547':('EC1','Resampling changes prediction performance/calibration; SHAP/LIME provide clinical interpretation without explanation consistency measurement.'),
'N572':('EC1','Uncertainty intervals and errors assess compressive-strength predictions; SHAP interprets influential factors, without explanation uncertainty.'),
'N596':('EC5','Stability is the production-line flow-rate response to replenishment/barrel changes; SHAP explains a physical process-stability target.'),
'N598':('EC1','Monte Carlo/Sobol uncertainty concerns gasification outputs and sensitivity convergence; SHAP explains factors without attribution variability.'),
'N602':('EC1','HuntGPT evaluates technical answer accuracy and response readability; SHAP/LIME support a dashboard but explanation stability is not measured.'),
'N605':('EC1','Bayesian GLM intervals quantify shear-wave prediction uncertainty; SHAP interprets XGBoost parameters without attribution uncertainty.'),
'N606':('EC1','Conformal intervals quantify maize-yield uncertainty; SHAP identifies drivers without explanation stability or uncertainty assessment.'),
'N610':('EC1','FET prediction discrimination and clinical utility are evaluated; SHAP identifies predictors without attribution-stability measurement.'),
'N620':('EC1','Noise tests assess prediction stability of network QoS models; SHAP validates physical alignment, not explanation stability.'),
'N643':('EC1','Sensitivity analyses concern diet/cognition associations; SHAP identifies clinical predictors without attribution stability.'),
'N650':('EC3','SHAP comparisons describe LR-versus-RF mechanisms; stable radiomics selection and cross-model visual differences are not separately measured explanation stability.'),
'N658':('EC1','Stability refers to stock-market entropy/fractal dynamics; SHAP diagnoses information content without explanation variability.'),
'N662':('EC5','Title explicitly studies pseudo-static slope stability, a geotechnical target rather than explanation stability.'),
'N668':('EC1','Model selection and popularity prediction metrics are evaluated; SHAP explains recency/loudness without attribution stability.'),
'N672':('EC3','Interpretability uses permutation importance and TabNet attention, not SHAP/LIME; atmospheric stability is a physical input.'),
'N681':('EC1','Wind-power predictive accuracy and site-specific SHAP drivers are reported without explanation-stability measurement.'),
'N691':('EC1','SHAP molecular heatmaps recover known chemical structure effects; no attribution stability/uncertainty evaluation is reported.'),
'N715':('EC3','Shapley values price data records and optimize broker revenue; these are data valuation, not SHAP/LIME feature explanations.'),
'N718':('EC2','SHAP uncertainty is evaluated exclusively on MNIST and neuroimaging data, with no tabular study described.'),
'N719':('EC3','Title concerns Bayesian Shapley valuation of inputs under supply uncertainty, not SHAP/LIME machine-learning feature explanations.'),
'U036':('EC3','Spearman 0.70 is agreement between LIME and SHAP, not stability of either explainer under repeats or perturbations.'),
'U037':('EC2','Fake-news explanations are token-level on text corpora with semantic/structural text perturbations, not tabular data.'),
'U038':('EC3','Explanations are intrinsic graph/causal-attention outputs; no SHAP/LIME-family evaluation is described.'),
'U047':('EC3','Metrics explicitly assess rule-based textual explanations with Levenshtein similarity rather than SHAP/LIME attributions.'),
'U052':('EC1','Stability refers to depressive-symptom model performance; SHAP supplies predictor interpretation without attribution variation.'),
'U062':('EC1','Longitudinal PD accuracy and early prediction are evaluated; attribution aids transparency without measured explanation stability.'),
'U080':('EC3','Consensus aggregation/heatmaps compare SHAP, permutation, Gini and gain rankings across techniques rather than repeat/perturbation stability.'),
'U087':('EC1','Grouped cross-validation studies concrete-prediction robustness across sources; SHAP identifies contributors without attribution variability.'),
'U104':('EC1','Cross-year crop-yield errors and drought robustness are evaluated; SHAP interprets climate drivers without explanation stability.'),
'U171':('EC1','Radiomic classifier accuracy/AUC comparisons and descriptive SHAP biomarkers are reported, without explanation-stability measurement.'),
'U182':('EC1','External validation establishes mortality-prediction robustness; SHAP ranks prognostic variables without attribution stability.'),
'U191':('EC1','Log-anomaly performance/model stability is improved by preprocessing; SHAP supplies interpretation without an explanation-stability outcome.'),
'U197':('EC1','Multicenter osteoporosis prediction and calibration are evaluated; SHAP identifies top contributors without attribution stability.'),
'U204':('EC1','Stability is predictive performance across cost models; SHAP explains cost-transmission mechanisms without attribution variability.'),
'U205':('EC2','LIME stability is studied on text classifiers and semantic text perturbations only.'),
'U217':('EC3','Study generates local explanations with ExIFFI, outside the SHAP/LIME family.'),
'U219':('EC1','Player-value ablations assess prediction/ranking utility; SHAP explains market drivers without explanation-stability measurement.'),
'U235':('EC1','Study compares diabetes prediction, interpretability and computational costs; no separate SHAP/LIME stability result is reported.'),
'U263':('EC1','Stable predictors concern feature selection and clinical prediction; SHAP interprets a nomogram without attribution stability.'),
'U265':('EC1','Stability concerns protein classifier performance and bootstrap confidence; SHAP supplies biological interpretations without attribution variability.'),
'U286':('EC1','Multi-seed evaluation concerns accepted malware accuracy/latency and routing; SHAP guides refinement without explanation-stability assessment.'),
'U317':('EC1','Wine prediction/transfer performance is compared; SHAP identifies chemical determinants without attribution variability.'),
'U321':('EC1','Drift concerns equipment configuration; SHAP interprets prediction effects without measuring stability of explanations.'),
'U329':('EC1','Wildfire prediction ROC/PR-AUC and SHAP spatial drivers are reported; no attribution-stability outcome is assessed.'),
'U347':('EC1','Rainfall models report accuracy/AUC/classification kappa; SHAP gives variable importance without explanation-stability assessment.'),
'U357':('EC1','Economic stability is the monetary-policy target; SHAP interprets stock forecasts without explanation-stability analysis.'),
'U361':('EC1','Exact fragment Shapley computation removes sampling, but the abstract reports chemical interpretability/optimization rather than a separately evaluated stability outcome.'),
'U367':('EC1','SMOTE/RF stroke accuracy and SHAP/LIME clinical drivers are assessed without explanation-stability measurement.'),
'U376':('EC1','Sensitivity analysis tests dropout prediction performance with/without CTGAN; SHAP supports intervention signals without attribution stability.'),
'U385':('EC1','Robustness refers to pollutant-removal prediction and MOF design; SHAP interprets factors without explanation stability.'),
'U448':('EC5','Lime is CaO catalyst and stability concerns hydration/carbonation during storage, not machine-learning explanations.'),
'U463':('EC2','Title explicitly concerns EEG emotion recognition, outside the tabular-data scope.'),
'U520':('EC3','Attribution values score distributed-training participants/data contributions rather than input-feature SHAP/LIME explanations.'),
'U522':('EC2','Free-text explanations and annotator imitation concern sentence-pair NLP tasks, not tabular SHAP/LIME.'),
'U529':('EC2','Vision-transformer LRP/image attribution study is exclusively non-tabular in the supplied abstract.'),
'U535':('EC3','Gale-Shapley stable matching and strategic Nash equilibria are outside SHAP/LIME feature attribution.'),
'U549':('EC3','Study concerns Gale-Shapley stable marriage algorithms, not machine-learning explanations.'),
'U550':('EC2','Explanation stability concerns fine-tuned language models, not tabular models.'),
'U553':('EC3','Two-sided matching-market stability uses Gale-Shapley, not SHAP/LIME feature explanations.'),
'U564':('EC3','Title concerns cooperative-game core stability of the Shapley value, not machine-learning feature explanations.')}
ex['U431']=('EC1','Unstable Pb prediction routes geochemical source analysis; SHAP bounds pollutant sources but no explanation-stability outcome is described.')
uncertain={
'U004':'Abstract explicitly labels its quantitative results mock; verify whether this is an eligible full methodological report with genuine explanation-stability evidence.',
'U033':'SHAP stability is an aim/regularizer but no separate stability measure/result is specified; full text needed to distinguish measured outcome from an asserted benefit.',
'U048':'Abstract names explanation stability as a concern but reports only predictive verification/calibration; SHAP/LIME method and separate stability evidence are not established.',
'U054':'Abstract includes feature stability analysis and SHAP for classical/quantum models, but does not distinguish SHAP attribution stability from feature-selection or prediction robustness.',
'U444':'Robustness/stability measures vary inputs and patient subpopulations, but abstract does not identify SHAP/LIME and reports only prediction gains from regularization.',
'U498':'Abstract directly studies relative explanation stability on real data, but does not identify which of seven explanation methods or data modalities are evaluated; inspect full text.',
'U500':'MVFS-SHAP measures stability of selected feature subsets; unclear whether SHAP attribution rankings themselves are separately evaluated rather than only selection-pipeline stability.',
'U552':'Explanation stability is measured across multiple modalities/methods, but abstract does not identify tabular data or a SHAP/LIME-family explainer.',
'N239':'Truncated abstract describes adversarial attacks on cloud detection and LIME; cannot establish whether explanation changes or only prediction evasion are measured.'}
outputs=[]
for r in records:
 ident=r['id']; a=re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',r.get('abstract') or ''))).strip(); basis='Stored title and abstract' if a else 'Title only'
 if ident in include:
  decision='Include';criterion='Potentially eligible';reason=include[ident];confidence='medium' if ident in {'N594','N638','N687','U001','U008','U050','U175','U192','U491'} else 'high'
 elif ident in ex:
  decision='Exclude';criterion,reason=ex[ident];confidence='high' if ident not in {'N349','N376','N650','U361','N719'} else 'medium'
 elif ident in uncertain:
  decision='Uncertain';criterion='Insufficient evidence';reason=uncertain[ident];confidence='low'
 elif not a:
  decision='Uncertain';criterion='Insufficient evidence';reason=f'Title identifies {r["title"]}; no abstract is stored, so tabular scope, SHAP/LIME involvement and a separately assessed explanation-stability outcome cannot all be established.';confidence='low'
 elif a.endswith('...') or a.endswith('…'):
  decision='Uncertain';criterion='Insufficient evidence';reason=f'Stored abstract is truncated. It introduces {r["title"]}, but the available excerpt cannot establish a separate SHAP/LIME explanation-stability outcome; retrieve complete abstract.';confidence='low'
 else:
  raise ValueError('Unreviewed full abstract: '+ident)
 outputs.append({'id':ident,'decision':decision,'criterion':criterion,'rationale':reason,'evidence_basis':basis,'confidence':confidence})
assert len(outputs)==197 and len({r['id'] for r in outputs})==197
assert {x['id'] for x in records}=={x['id'] for x in outputs}
(p/'shahoud_decisions.json').write_text(json.dumps(outputs,indent=2,ensure_ascii=False))
from collections import Counter
print(Counter(r['decision'] for r in outputs));print(Counter(r['criterion'] for r in outputs));print('Missing',sum(not (r.get('abstract') or '').strip() for r in records));print('Truncated',sum((r.get('abstract') or '').strip().endswith(('...','…')) for r in records))
