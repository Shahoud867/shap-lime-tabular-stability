"""Supplementary search (realigned RQ3: mechanisms and pre-perturbation predictors of explanation instability)."""
from strings import I
# word-level stability concept (broader than the main 19 phrases, because the mechanism concept already narrows the set)
O2 = ['stability', 'stable', 'unstable', 'instability', 'robustness', 'robust', 'fragility', 'fragile', 'consistency', 'uncertainty']
# mechanism and diagnostic concept, with truncation (IEEE)
M = ['"split threshold*"', '"split point*"', '"decision threshold*"', '"decision boundar*"', 'discretiz*', 'discretis*', '"bin edge*"', 'binning',
     '"piecewise constant"', '"tree structure"', '"decision path*"', '"fragility score"', '"instability score"', '"stability score"', '"reliability score"',
     '"uncertainty estimat*"', '"uncertainty quantification"', '"epistemic uncertainty"', '"explanation uncertainty"', '"predict* instability"',
     '"weak learner*"', 'smoothness', 'lipschitz*', '"credible interval*"', '"uncertainty interval*"', '"sampling variance"', '"sampling noise"', '"sampling randomness"']
# the same concept spelled out for sources without truncation (arXiv, Semantic Scholar)
M_PLAIN = ['"split threshold"', '"split thresholds"', '"split point"', '"split points"', '"decision threshold"', '"decision boundary"', '"decision boundaries"',
           'discretization', 'discretisation', 'discretized', '"bin edges"', 'binning', '"piecewise constant"', '"tree structure"', '"decision path"',
           '"decision paths"', '"fragility score"', '"instability score"', '"stability score"', '"reliability score"', '"uncertainty estimation"',
           '"uncertainty estimates"', '"uncertainty quantification"', '"epistemic uncertainty"', '"explanation uncertainty"', '"predict instability"',
           '"predicting instability"', '"weak learner"', '"weak learners"', 'smoothness', 'Lipschitz', '"credible intervals"',
           '"uncertainty intervals"', '"sampling variance"', '"sampling noise"', '"sampling randomness"']
T = ['tabular', '"structured data"', '"tree ensemble*"', '"tree-based"', '"random forest*"', 'XGBoost', '"gradient boost*"', 'LightGBM', '"decision tree*"', '"logistic regression"']
T_PLAIN = ['tabular', '"structured data"', '"tree ensemble"', '"tree ensembles"', '"tree-based"', '"random forest"', '"random forests"', 'XGBoost', '"gradient boosting"',
           '"gradient boosted"', 'LightGBM', '"decision tree"', '"decision trees"', '"logistic regression"']
I_PLAIN = [t.replace('*', '') for t in I]

def ieee():
    f = lambda terms: "(" + " OR ".join(f'"All Metadata":{t}' for t in terms) + ")"
    return f"{f(I)} AND {f(O2)} AND {f(M)} AND {f(T)}"
def arxiv():
    f = lambda terms: "(" + " OR ".join(f"abs:{t}" for t in terms) + ")"
    return f"{f(I_PLAIN)} AND {f(O2)} AND {f(M_PLAIN)} AND {f(T_PLAIN)} AND submittedDate:[201601010000 TO 202612312359]"
def s2():
    f = lambda terms: "(" + " | ".join(terms) + ")"
    return " + ".join(f(x) for x in (I_PLAIN, O2, M_PLAIN, T_PLAIN))
