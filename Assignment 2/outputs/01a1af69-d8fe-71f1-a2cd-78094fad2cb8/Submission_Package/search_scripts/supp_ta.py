"""Title/abstract decisions for the supplementary records (N001-N719).
Records not listed in OVR were screened on title (and abstract where needed) as applied/predictive studies (EC1),
with non-tabular modality (EC2) and physical-sense 'stability'/'fragility' homonyms (EC5) flagged by the rules below and checked by eye."""
import json, re
OVR = {
 'N101': ('I', 'Sanity checks (stability/robustness) of LIME/SHAP-type interpreters on tabular Android-malware features'),
 'N300': ('I', 'Prediction-equivalent models give different SHAP attributions across 24 datasets'),
 'N313': ('I', 'Explanation uncertainty for GBDT/transformer models on tabular data'),
 'N459': ('I', 'Tilia: decision-tree surrogates to fix LIME instability'),
 'N616': ('I', 'Forecast uncertainty near decision boundaries explains LIME/SHAP instability'),
 'N708': ('I', 'Distributional uncertainty of the SHAP score (theory with examples)'),
 'N029': ('U', 'ALIME with decision-tree surrogate - check stability outcome'),
 'N036': ('U', 'SHAP-guided feature stability for malware detection - check whether explanation stability or cross-model agreement'),
 'N048': ('U', 'Empirical validation of SHAP and LIME - check for stability measures'),
 'N126': ('U', 'Hierarchical group SHAP addressing sampling variance - check stability outcome'),
 'N239': ('U', 'LIME analysis of adversarial manipulation in tabular IDS - check explanation-change outcome'),
 'N303': ('U', 'ELIME exact LIME variant - check stability claim'),
 'N305': ('U', 'Stability-XAI on UCI heart data - check whether explanation stability is measured'),
 'N312': ('U', 'Uncertainty and consistency of SHAP feature importance (climate indices) - check modality and outcome'),
 'N339': ('U', 'ShapleyVIC: SHAP importance across near-optimal models with uncertainty intervals'),
 'N383': ('U', 'DEM resolution and explanation reliability in landslide models - check outcome'),
 'N386': ('U', 'Accurate and robust Shapley values - no abstract, check'),
 'N420': ('U', 'DSExplainer: belief/plausibility intervals for SHAP - check'),
 'N479': ('U', 'Logistic regression with SHAP stability analysis - no abstract, check'),
 'N480': ('U', 'TreeSHAP vs ensemble LIME vs Anchors on MIMIC-IV - check stability measures'),
 'N557': ('U', 'LIME with neural decision trees; stability discussed - check'),
 'N692': ('U', 'BMB-LIME: uncertainty in LIME explanations - no abstract, check'),
 'N693': ('U', 'Probabilistic Shapley inference with attribution uncertainty - check explainer family and data'),
 'N030': ('E2', 'Synthetic signal (time-series) data'),
 'N033': ('E1', 'Surrogate-explanation method; fidelity and efficiency, not stability'),
 'N077': ('E3', 'Behavioural verification of random forests (no SHAP/LIME stability)'),
 'N086': ('E3', 'Counterfactual explanations for classifier robustness'),
 'N115': ('E3', 'Decision-path surrogate explanations'),
 'N150': ('E3', 'Counterfactual explanations for abstention'),
 'N177': ('E3', 'Counterfactual explanations'),
 'N191': ('E3', 'Counterfactual and prototype explanations'),
 'N252': ('E3', 'Counterfactual explanations with GANs'),
 'N368': ('E3', 'Rule-based explainer (LUX)'),
 'N389': ('E3', 'New global correlation-aware attribution score, not SHAP/LIME'),
 'N687': ('E3', 'Tree regularisation for interpretability, not explanation stability'),
 'N697': ('E3', 'SHAP used to explain predictive uncertainty (not uncertainty of explanations)'),
 'N713': ('E3', 'Shapley values explaining predictive uncertainty (not uncertainty of explanations)'),
 'N233': ('E1', 'Simulation input-uncertainty quantification'),
 'N371': ('E1', 'Model (prediction) stability, not explanation stability'),
 'N439': ('E2', 'Neuroimaging data'),
 'N535': ('E2', 'Image data (CNN explanations)'),
 'N710': ('E1', 'Adequacy of SHAP scores, not stability'),
 'N711': ('E1', 'Exact TreeSHAP algorithm for piecewise linear trees; no stability outcome'),
 'N718': ('E2', 'Medical imaging (MICCAI)'),
 'N484': ('E1', 'Cross-target SHAP feature associations (QSAR), no stability outcome'),
 'N699': ('E1', 'Decision-boundary maps in Shapley space (visualisation)'),
 'N010': ('E1', 'Random-forest variant; no explanation-stability outcome'),
 'N143': ('E1', 'Applied materials prediction'), 'N532': ('E1', 'Applied materials prediction'), 'N395': ('E1', 'Applied tunnelling control'),
 'N671': ('E4', 'Editorial'),
}
E2 = r"\b"  r"(image|images|imaging|cnn|mri|ct|computed tomography|radiomic|radiomics|eeg|ecg|ultrasound|ultrasonography|text|nlp|bert|tinybert|llm|llms|language model|large language|reviews|point cloud|point clouds|hyperspectral|video|speech|voice|audio|histopathology|segmentation|raman|galaxy)"
E5 = r"(slope stability|transient stability|voltage stability|stability prediction of|stability assessment of the ieee|stable behavioral|perovskite|phase.stability|seismic fragility|fragility modeling|fragility assessment|structural vulnerability|tunnel|thermodynamically stable|stabilized expansive|stability perturbations in continuous)"
new = json.load(open('final/supp_unique.json', encoding='utf8'))
out = []
for n in new:
    u = n['uid']; t = (n['title'] or '').lower()
    if u in OVR: c, r = OVR[u]
    elif re.search(E5, t): c, r = 'E5', "'Stability'/'fragility' in a physical or engineering sense"
    elif re.search(E2, t): c, r = 'E2', 'Non-tabular modality or sequence/deep-learning focus (title)'
    else: c, r = 'E1', 'Applied/predictive study; SHAP/LIME or uncertainty used only for interpretation or prediction, no explanation-stability aim or outcome'
    out.append(f'{u} {c} {r}')
open('final/ta_supp.txt', 'w', encoding='utf8').write('\n'.join(out) + '\n')
import collections
print(collections.Counter(l.split()[1] for l in out))
