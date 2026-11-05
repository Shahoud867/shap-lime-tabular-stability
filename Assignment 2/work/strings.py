I = ['SHAP','"Shapley additive"','"Shapley values"','TreeSHAP','KernelSHAP','LIME','"local interpretable model-agnostic"','"feature attribution*"','"post-hoc explanation*"','"local explanation*"']
O = ['"explanation stability"','"stability of explanations"','"stability of model explanations"','"stability of LIME"','"LIME stability"','"stability of SHAP"','"SHAP stability"','"attribution stability"','"explanation instability"','"unstable explanations"','"stable explanations"','"explanation robustness"','"robustness of explanations"','"robust explanations"','"robustness of interpretability"','"attribution robustness"','"explanation consistency"','"explanation fragility"','"stability indices"']
P = ['tabular','"structured data"','"tree ensemble*"','"tree-based"','"random forest*"','XGBoost','"gradient boosting"','LightGBM','"logistic regression"','"credit scoring"','"credit risk"','fraud']
def generic():
    j = lambda t: "(" + " OR ".join(t) + ")"
    return f"{j(I)} AND {j(O)} AND {j(P)}"
def ieee():
    f = lambda terms: "(" + " OR ".join(f'"All Metadata":{t}' for t in terms) + ")"
    return f"{f(I)} AND {f(O)} AND {f(P)}"
