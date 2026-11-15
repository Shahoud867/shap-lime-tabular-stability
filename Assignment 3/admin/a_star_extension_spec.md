# A* Extension Specification: Proposition P2 ($F_{\mathrm{jump}}$), Scale-Free Calibration ($F_{\mathrm{cal}}$), and SOTA Baselines

**Author:** `lit-strategist`  
**Date:** 2026-11-15  
**Target Consumers:** `@architect-data` (`crossings.py`, `fragility.py`, `baselines.py`, `download_data.py`), `@experimenter`, `@paper-writer`  
**Status:** Specification locked for Phase 2 implementation.

---

## 1. Purpose & Relation to Phase 1 (`v1.0-assignment3-baseline`)

In Phase 1, unweighted split-crossing fragility $F_{\mathrm{tree}}(x; s)$ established strong within-dataset explanation instability correlation and outperformed scalar margin and tree-variance baselines on individual benchmarks, but raw threshold transfer in Leave-One-Dataset-Out (`LODO`) gating collapsed (`r7_lodo_gate.json`) because unnormalized split sums scale with tree count $T$, tree depth, and dataset dimensionality $d$. Additionally, elevating the work from a course paper to a Tier-1 A* submission (NeurIPS / ICML / ICLR / AAAI / KDD / FAccT) requires:

1. **Theoretical Strengthening (Proposition P2):** Weighting every internal split crossing probability by the fitted leaf-value jump $\Delta_{t,v} = |V(v_L) - V(v_R)|$ normalized by tree prediction spread $\hat{\sigma}_{\mathrm{pred}}(t)$, yielding a formal attribution-shift sensitivity bound ($F_{\mathrm{jump}}$).
2. **Scale-Free Calibration ($F_{\mathrm{cal}}$):** Mapping raw $F_{\mathrm{jump}}(x; s)$ against the training distribution via empirical quantile ranks (and robust MAD z-scores) so a single cross-dataset gating threshold transfers cleanly across heterogeneous ensembles in LODO evaluation.
3. **Head-to-Head SOTA A\* Baselines:** Implementing faithful baselines for **S-LIME** (`zhou2021`), **GPEC** (`hill2024`), **ST-SHAP** (`kelodjou2024`), **Epistemic UQ-Gating** (`mikriukov2026`), and **AXOM** (`pala2024`).

---

## 2. Proposition P2: Leaf-Value-Jump Sensitivity Bound & $F_{\mathrm{jump}}(x; s)$

### 2.1 Bottom-Up Node Value & Jump Extraction (Tree Ensembles)

Let a trained tree ensemble $f$ consist of $T$ trees $\{t = 1, \dots, T\}$, where each tree $t$ has internal split nodes $\mathcal{V}_t$ and leaf nodes $\mathcal{L}_t$. Each internal node $v \in \mathcal{V}_t$ routes an input $x \in \mathbb{R}^d$ to its left child $v_L$ if $x_{j(v)} \prec_t \theta_v$ (with $\prec_t \equiv \le$ for scikit-learn `RandomForestClassifier` and $\prec_t \equiv <$ for `XGBoost`), and to its right child $v_R$ otherwise. Let $\mathcal{P} \subseteq \{1, \dots, d\}$ denote the set of perturbable features.

For every node $u \in \mathcal{V}_t \cup \mathcal{L}_t$, let $c_u > 0$ denote its training cover weight (`weighted_n_node_samples[u]` in scikit-learn; `Cover` in XGBoost `trees_to_dataframe()`). Define the **tree-conditional expected output** $V(u)$ recursively bottom-up in $O(|\mathcal{V}_t| + |\mathcal{L}_t|)$ time:

- **Base case ($u \in \mathcal{L}_t$):**
  - **Random Forest (`RF`):** $V(u)$ is the positive-class proportion at leaf $u$:
    $$V(u) = \frac{\texttt{tree.value[u, 0, 1]}}{\texttt{tree.value[u, 0, 0]} + \texttt{tree.value[u, 0, 1]}}.$$
    *(Implementation note for scikit-learn 1.8: explicitly normalize by row sum because `sklearn >= 1.4` stores normalized class proportions in `tree_.value` rather than raw counts; dividing by the sum handles both conventions safely.)*
  - **XGBoost (`XGB`):** $V(u)$ is the leaf raw log-odds margin weight (stored in the `Gain` column of `booster.trees_to_dataframe()` where `Feature == "Leaf"`).
- **Recursive step ($u \in \mathcal{V}_t$):**
  $$V(u) = \frac{c_{v_L}}{c_{v_L} + c_{v_R}} V(v_L) + \frac{c_{v_R}}{c_{v_L} + c_{v_R}} V(v_R), \quad \text{where } (v_L, v_R) \text{ are the children of } u.$$
  *(For XGBoost, `trees_to_dataframe()` stores split-loss reduction—not node prediction—in `Gain` on internal rows; therefore `V(u)` MUST be computed bottom-up from the leaf rows via post-order traversal using `Yes`/`No` child IDs and `Cover` weights.)*

For each internal node $v \in \mathcal{V}_t$, define:
1. **Mean Cover-Weighted Jump:**
   $$\Delta_{t,v} = \left| V(v_L) - V(v_R) \right|.$$
2. **Worst-Case Leaf-Range Oscillation (used in Proposition P2a exact bound):**
   $$\bar{\Delta}_{t,v} = \max_{\ell \in \mathcal{L}(v_L),\, \ell' \in \mathcal{L}(v_R)} |V(\ell) - V(\ell')| = \max\!\big(|M_{v_L}^{\max} - M_{v_R}^{\min}|,\; |M_{v_R}^{\max} - M_{v_L}^{\min}|\big),$$
   computable in the same $O(|\mathcal{V}_t|+|\mathcal{L}_t|)$ bottom-up pass by tracking $(M_u^{\min}, M_u^{\max}) = (\min_{c} M_c^{\min}, \max_{c} M_c^{\max})$ at each node. Note $\Delta_{t,v} \le \bar{\Delta}_{t,v}$ always, with equality when child subtrees are homogeneous or depth $D = 1$.

### 2.2 Normalization Constant $\hat{\sigma}_{\mathrm{pred}}(t)$

To make jump weights dimensionless across classifiers and scale families, define the tree output scale $\hat{\sigma}_{\mathrm{pred}}(t)$ on the training set $X_{\mathrm{train}}$ (floored at $\epsilon = 10^{-6}$):

- **Random Forest (`RF`):** $\hat{\sigma}_{\mathrm{pred}}(t) = \max\!\left(\mathrm{SD}\!\big(f_t(X_{\mathrm{train}})\big),\, 10^{-6}\right)$, where $f_t(x) \in [0,1]$ is tree $t$'s positive-class probability.
- **XGBoost (`XGB`):** Because boosting trees are additive contributions $F(x) = b_0 + \sum_{t=1}^T f_t(x)$ whose late-round step magnitudes $f_t$ shrink toward zero as residuals vanish, dividing each tree's $|\Delta_{t,v}|$ by its own vanishing $\mathrm{SD}(f_t(X_{\mathrm{train}}))$ would artificially inflate late negligible trees to equal weight with early foundational trees. Therefore:
  - **Primary (`sigma_pred_mode="tree"` vs `"ensemble"` check):** Implement per-tree $\hat{\sigma}_{\mathrm{pred}}(t) = \max(\mathrm{SD}(f_t(X_{\mathrm{train}})), 10^{-6})$ for RF, and compare `"ensemble"` normalization $\hat{\sigma}_{\mathrm{pred}}(t) = \max(\mathrm{SD}(F(X_{\mathrm{train}})) / T,\, 10^{-6})$ (or raw-margin SD $\mathrm{SD}(F(X_{\mathrm{train}}))$) against per-tree normalization on XGBoost so additive tree magnitudes are preserved accurately. Record both variants in `fragility.py` (`sigma_mode: Literal["tree", "ensemble"]`) and verify empirically on a validation dataset before locking the default.

### 2.3 Closed-Form $F_{\mathrm{jump}}(x; s)$

Given perturbation scale $s > 0$ and training feature standard deviations $\{\sigma_j\}_{j=1}^d$ (with zero-SD features excluded/floored), the **Jump-Weighted Boundary Fragility** at instance $x \in \mathbb{R}^d$ is:

$$F_{\mathrm{jump}}(x; s) \;=\; \frac{1}{T} \sum_{t=1}^{T} \sum_{v \in \mathcal{V}_t :\, j(v) \in \mathcal{P}} \frac{|\Delta_{t,v}|}{\hat{\sigma}_{\mathrm{pred}}(t)} \, \Phi\!\left(-\frac{|x_{j(v)} - \theta_v|}{s\,\sigma_{j(v)}}\right),$$

where $\Phi$ is the standard normal CDF. Under unconstrained isotropic-marginal Gaussian perturbations $X' \sim \mathcal{N}(x, \Sigma_s)$ with $\Sigma_s = \mathrm{diag}(s^2\sigma_j^2 \mathbb{I}[j \in \mathcal{P}])$, each event $v \in C_t(x, X') := \{v \in \mathcal{V}_t : \mathbb{I}[x_{j(v)} \prec_t \theta_v] \neq \mathbb{I}[X'_{j(v)} \prec_t \theta_v]\}$ occurs with exact marginal probability $\mathbb{P}[v \in C_t(x, X')] = \Phi\!\left(-\frac{|x_{j(v)}-\theta_v|}{s\sigma_{j(v)}}\right)$, so $F_{\mathrm{jump}}(x; s)$ equals the exact expected normalized split-jump sum across crossed tree splits.

*(Note on $\tau$-filtered prediction preservation: When perturbations are filtered to $\{X' : |\hat{p}(X') - \hat{p}(x)| \le \tau\}$, $\Phi(-\frac{|x_{j(v)}-\theta_v|}{s\sigma_{j(v)}})$ upper-bounds the conditional crossing rate of high-leverage splits inside the $\tau$-band, while remaining zero-cost and closed-form.)*

### 2.4 Formal Statement & Proof of Proposition P2

**Proposition P2 (Leaf-Value-Jump Attribution Sensitivity Bound).**  
*Let $f = \frac{1}{T}\sum_{t=1}^T f_t$ be a fitted tree ensemble (probability average for RF, or raw margin sum $\sum_t f_t$ with $L_\sigma$-Lipschitz probability link $\sigma(z) = (1+e^{-z})^{-1}$, $L_\sigma = \frac{1}{4}$, for XGBoost), and let $\phi_f^{\mathrm{path}}(x) \in \mathbb{R}^d$ denote its Path-Dependent TreeSHAP attribution vector (`lundberg2020`). Let $X' \sim \mathcal{N}(x, \Sigma_s)$ be a Gaussian perturbation on $\mathcal{P}$.*

1. **(P2a — Exact Worst-Case Oscillation Bound):**
   $$\mathbb{E}_{X'}\!\left[\,\|\phi_f^{\mathrm{path}}(X') - \phi_f^{\mathrm{path}}(x)\|_\infty\,\right] \;\le\; \frac{2\,L_f}{T} \sum_{t=1}^{T} \sum_{v \in \mathcal{V}_t :\, j(v) \in \mathcal{P}} \bar{\Delta}_{t,v}\,\Phi\!\left(-\frac{|x_{j(v)}-\theta_v|}{s\,\sigma_{j(v)}}\right),$$
   *and consequently $\mathbb{E}_{X'}[\|\phi_f^{\mathrm{path}}(X') - \phi_f^{\mathrm{path}}(x)\|_1] \le d \cdot \text{RHS}$, where $L_f = 1$ for RF probabilities and raw-margin GBDT Shapley values (or $L_f = T/4$ for probability-rescaled GBDTs).*

2. **(P2b — Expected Conditional-Jump Estimator & Monotonicity):**  
   *Replacing worst-case extrema $\bar{\Delta}_{t,v}$ with the normalized cover-weighted conditional jump $\Delta_{t,v}/\hat{\sigma}_{\mathrm{pred}}(t)$ yields $F_{\mathrm{jump}}(x; s)$, which:*
   - *equals the exact expected normalized attribution-perturbation jump on any depth-1 ensemble (and whenever crossed nodes have locally uniform child subtrees, i.e., $\Delta_{t,v} = \bar{\Delta}_{t,v}$);*
   - *recovers Proposition P1 consistency: if $F_{\mathrm{jump}}(x; s) = 0$ on an ensemble with non-degenerate splits ($\Delta_{t,v} > 0$), then $\mathbb{P}[C_t(x, X') \neq \emptyset] = 0$ and $\phi_f^{\mathrm{path}}(X') = \phi_f^{\mathrm{path}}(x)$ a.s. (up to $s=0$ degeneracy);*
   - *is strictly monotonically non-decreasing in perturbation scale $s > 0$ and invariant to adding redundant zero-jump splits ($\Delta_{t,v} = 0$).*

**Proof of (P2a).**  
Fix a single tree $t$ and perturbation realization $x'$. By linearity of TreeSHAP across additive trees, $\phi_{f,j}^{\mathrm{path}}(x) = \frac{1}{T}\sum_{t=1}^T \phi_{t,j}^{\mathrm{path}}(x)$ (with $1/T$ omitted before $L_f$-rescaling for raw GBDT margins). For any coalition $A \subseteq [d]$, let $g_t(x_A) = \mathbb{E}_{\mathrm{tree}}[f_t(X) \mid X_A = x_A]$ be the path-dependent conditional expectation evaluated recursively from root to leaves (`lundberg2020`, Algorithm 1). Define at each subtree rooted at $u \in \mathcal{V}_t \cup \mathcal{L}_t$ the discrepancy $E_u(x, x', A) := |g_{t,u}(x'_A) - g_{t,u}(x_A)|$, and let $C_t(x, x')$ be the set of crossed split nodes in tree $t$.

We prove by structural induction on $u$ that for every $A \subseteq [d]$:
$$E_u(x, x', A) \;\le\; \sum_{v \in C_t(x, x') \cap \mathcal{V}(u)} \bar{\Delta}_{t,v},$$
where $\mathcal{V}(u)$ denotes internal nodes in the subtree rooted at $u$.
- *Base case ($u \in \mathcal{L}_t$):* Leaf outputs are constant in $x$, so $E_u = 0$.
- *Inductive step ($u \in \mathcal{V}_t$):*
  - If $j(u) \notin A$, path-dependent TreeSHAP mixes the left and right child subtrees with fixed cover weights $w_L = \frac{c_{u_L}}{c_{u_L}+c_{u_R}}$ and $w_R = 1 - w_L$. By the triangle inequality and inductive hypothesis:
    $$E_u \le w_L E_{u_L} + w_R E_{u_R} \le \max(E_{u_L}, E_{u_R}) \le \sum_{v \in C_t(x, x') \cap \mathcal{V}(u)} \bar{\Delta}_{t,v}.$$
  - If $j(u) \in A$ and $u \notin C_t(x, x')$, both $x$ and $x'$ route to the same child $c \in \{u_L, u_R\}$, so $E_u = E_c \le \sum_{v \in C_t(x, x') \cap \mathcal{V}(u)} \bar{\Delta}_{t,v}$.
  - If $j(u) \in A$ and $u \in C_t(x, x')$, one input routes into $\mathcal{L}(u_L)$ and the other into $\mathcal{L}(u_R)$. Because $g_{t,u_L}(\cdot)$ is a convex combination of leaf values in $\mathcal{L}(u_L)$ and $g_{t,u_R}(\cdot)$ is a convex combination of leaf values in $\mathcal{L}(u_R)$,
    $$E_u = |g_{t,u_L}(\cdot) - g_{t,u_R}(\cdot)| \le \max_{\ell \in \mathcal{L}(u_L),\, \ell' \in \mathcal{L}(u_R)} |V(\ell) - V(\ell')| = \bar{\Delta}_{t,u} \le \sum_{v \in C_t(x, x') \cap \mathcal{V}(u)} \bar{\Delta}_{t,v}.$$

Applying the induction at $u = \mathrm{root}(t)$ gives $\max_{A \subseteq [d]} |g_t(x'_A) - g_t(x_A)| \le \sum_{v \in C_t(x, x')} \bar{\Delta}_{t,v}$. Now, for every feature $j \in [d]$, the Shapley formula uses non-negative coalition weights $w_{|S|} = \frac{|S|!(d-|S|-1)!}{d!}$ satisfying $\sum_{S \subseteq [d]\setminus\{j\}} w_{|S|} = 1$:
$$|\phi_{t,j}^{\mathrm{path}}(x') - \phi_{t,j}^{\mathrm{path}}(x)| \le \sum_{S \subseteq [d]\setminus\{j\}} w_{|S|} \left( |g_t(x'_{S\cup\{j\}}) - g_t(x_{S\cup\{j\}})| + |g_t(x'_S) - g_t(x_S)| \right) \le 2 \sum_{v \in C_t(x, x')} \bar{\Delta}_{t,v}.$$
Taking the maximum over $j \in [d]$, averaging over $t=1,\dots,T$, and taking $\mathbb{E}_{X' \sim \mathcal{N}(x, \Sigma_s)}[\cdot]$ yields $\mathbb{E}_{X'}[\mathbb{I}[v \in C_t(x, X')]] = \Phi\!\left(-\frac{|x_{j(v)}-\theta_v|}{s\sigma_{j(v)}}\right)$ for $j(v) \in \mathcal{P}$ (and $0$ for $j(v) \notin \mathcal{P}$), establishing (P2a). $\blacksquare$

**Why $F_{\mathrm{jump}}$ uses $\Delta_{t,v}$ rather than $\bar{\Delta}_{t,v}$ in practice:**  
In deep trees, $\bar{\Delta}_{t,v}$ can be inflated by a single sparse outlier leaf deep inside $v_L$ or $v_R$ that $x$ has near-zero probability of ever reaching, whereas $\Delta_{t,v} = |V(v_L) - V(v_R)|$ measures the cover-weighted conditional expectation jump directly induced at split $v$. Both can be exposed via `compute_f_jump(..., jump_mode="mean" | "range")` for ablation.

---

## 3. Scale-Free Calibration ($F_{\mathrm{cal}}$) for Cross-Dataset LODO Gating

### 3.1 Motivation (Why Raw Cutoffs Fail)
In `LODO` evaluation across datasets $\mathcal{D}_1, \dots, \mathcal{D}_M$, ensembles trained on different schemas vary substantially in split count $|\mathcal{V}_t|$, depth, and feature count $d$. An uncalibrated raw cutoff $\tau_{\mathrm{raw}}$ tuned on $\mathcal{D}_1$ will either flag $0\%$ or $100\%$ of instances on $\mathcal{D}_2$. Calibration must be **strictly label-free and test-set leak-free** (fit solely on $X_{\mathrm{train}}$ and the fitted model $f$).

### 3.2 Primary Calibrator: Empirical Training Quantile Rank ($F_{\mathrm{cal}}^{\mathrm{quant}}$)
Given fitted model $f$ and reference distribution $X_{\mathrm{ref}} \subseteq X_{\mathrm{train}}$ (full $X_{\mathrm{train}}$ or stratified sample of $n_{\mathrm{ref}} = \min(|X_{\mathrm{train}}|, 2000)$ rows):
1. Precompute $r_i = F_{\mathrm{jump}}(x_{\mathrm{ref}}^{(i)}; s)$ for $i = 1, \dots, n_{\mathrm{ref}}$.
2. For any query sample $x$ (validation or test), compute its Laplace-smoothed empirical CDF rank:
   $$F_{\mathrm{cal}}^{\mathrm{quant}}(x; s) \;=\; \frac{1 + \sum_{i=1}^{n_{\mathrm{ref}}} \mathbb{I}\!\big[r_i \le F_{\mathrm{jump}}(x; s)\big]}{n_{\mathrm{ref}} + 1} \;\in\; (0, 1).$$

**Properties:**
- Invariant to any strictly increasing dataset-specific scale transformation of $F_{\mathrm{jump}}$.
- On in-distribution test data, $F_{\mathrm{cal}}^{\mathrm{quant}}(X_{\mathrm{test}}; s) \approx \mathrm{Uniform}(0, 1)$, so a gating threshold $\theta_{\mathrm{gate}} \in (0, 1)$ (e.g., $\theta_{\mathrm{gate}} = 0.70$, or tuned on the $(M-1)$ training datasets in LODO CV to hit a target precision/false-deferral constraint) guarantees predictable coverage ($\approx 1 - \theta_{\mathrm{gate}}$ deferral rate) on unseen datasets.

### 3.3 Secondary Calibrator: Robust MAD / IQR Z-Score ($F_{\mathrm{cal}}^{\mathrm{rz}}$)
To preserve extreme-tail separation when needed:
$$F_{\mathrm{cal}}^{\mathrm{rz}}(x; s) \;=\; \frac{F_{\mathrm{jump}}(x; s) - m_{\mathrm{ref}}}{\tilde{s}_{\mathrm{ref}}},$$
where $m_{\mathrm{ref}} = \mathrm{median}(r_{1:n_{\mathrm{ref}}})$ and $\tilde{s}_{\mathrm{ref}} = \max\!\big(1.4826 \cdot \mathrm{median}|r_i - m_{\mathrm{ref}}|,\; \mathrm{IQR}(r_{1:n_{\mathrm{ref}}})/1.349,\; 10^{-6}\big)$ (with sigmoid mapping $\sigma(F_{\mathrm{cal}}^{\mathrm{rz}})$ optional when a $[0,1]$ score is needed). Use $F_{\mathrm{cal}}^{\mathrm{quant}}$ as the primary LODO gating score and $F_{\mathrm{cal}}^{\mathrm{rz}}$ as a parametric sensitivity companion.

---

## 4. Exact Specifications for SOTA A* Baselines (`src/baselines.py`)

All five baselines already have verified BibTeX citations in `Assignment 3/paper/references.bib`. Implement them strictly faithful to the published algorithms below.

### 4.1 S-LIME (`zhou2021`, KDD 2021 — Stabilized-LIME)

**Purpose in Benchmark:** Adaptive sample-size stabilization for LIME via CLT hypothesis testing (competing against fixed-budget LIME vs $F_{\mathrm{cal}}$-gated fallback / diagnostic).

**Exact Algorithm (Zhou et al., 2021, Algorithm 3):**
- **Inputs:** Black-box probability function $f(X) \in [0, 1]$, target instance $x \in \mathbb{R}^d$ (in interpretable/standardized space used by LIME), initial perturbation size $n_0 = 1000$, max sample budget $n_{\max} = 10000$, significance level $\alpha = 0.05$ ($Z_\alpha = \Phi^{-1}(1 - \alpha) \approx 1.6449$ one-sided), top-$k$ features $k = 4$, exponential kernel proximity weights $\omega(z) = \exp(-D(x, z)^2 / \sigma_w^2)$.
- **Procedure:**
  1. Set $n = n_0$, generate $n$ LIME perturbations $\mathcal{D}$ around $x$, evaluate black-box $y_i = f(x_i)$ and weights $\omega_i$, standardize weighted covariates to zero mean and unit norm, and run LARS step-by-step up to $k$ active features.
  2. At each LARS entry step ($< k$ active features), let $\hat{c}_1 \ge \hat{c}_2$ be the top two residual correlations $\hat{c}_a = \frac{1}{n}\sum_{t=1}^n r_t x_{t, a}$ among remaining candidates, and form the empirical $2 \times 2$ covariance matrix $\hat{\Sigma}$ of $(r_t x_{t,1}, r_t x_{t,2})$ across the $n$ perturbations:
     $$\mathrm{SE}_{\Delta} = \sqrt{\frac{2\,(\hat{\sigma}_{11}^2 + \hat{\sigma}_{22}^2 - 2\hat{\sigma}_{12}^2)}{n}}, \qquad t = \hat{c}_1 - \hat{c}_2 - Z_\alpha \, \mathrm{SE}_\Delta.$$
  3. If $t \ge 0$, accept the step and continue LARS.
  4. If $t < 0$, compute the standardized observed test $z$-score $Z_{p_n} = \frac{\hat{c}_1 - \hat{c}_2}{\mathrm{SE}_{\Delta}}$ (so $t = 0 \iff Z_{p_n} = Z_\alpha$) and update required sample size:
     $$n' = \min\!\left(n_{\max},\; \max\!\left(n + 100,\; \left\lceil n \left(\frac{Z_\alpha}{\max(Z_{p_n}, 10^{-3})}\right)^2 \right\rceil\right)\right).$$
     If $n' \ge n_{\max}$, set $n = n_{\max}$, draw remaining perturbations up to $n_{\max}$, and execute standard LARS to return top-$k$ features without further testing; otherwise reuse existing samples, augment to $n'$, and restart LARS.
- **Outputs & Fair Comparison:** Return top-$k$ weights $\hat{\beta} \in \mathbb{R}^d$, final sample count $n_{\mathrm{used}} \in [n_0, n_{\max}]$, and wall-clock runtime. Note: S-LIME stabilizes *Monte Carlo variance* of LIME surrogates, NOT structural decision-boundary fragility under covariate perturbation $X' \sim \mathcal{N}(x, \Sigma_s)$—which our experiments can demonstrate directly!

### 4.2 GPEC (`hill2024`, AISTATS 2024, `arXiv:2210.02419` — Gaussian Process Explanation Uncertainty)

**Purpose in Benchmark:** Strongest SOTA **uncertainty estimator for local explanations** capturing decision-boundary complexity via Weighted Exponential Geodesic (WEG) GP regression.

**Exact Algorithm (Hill et al., 2024, Sections 3 & 4.3):**
- **Setup:** Given black-box classifier $F: \mathbb{R}^d \to [0,1]$, explainer $\mathcal{E}(x) \in \mathbb{R}^d$ (here TreeSHAP or LIME), training/calibration samples $\{x_m\}_{m=1}^M$ ($M \le 500$ for $O(M^3)$ GP tractability), and their explanations $e_m = \mathcal{E}(x_m)$.
- **Decision Boundary Sampling ($\mathcal{M}$):** Sample $K$ boundary points $\{m_1, \dots, m_K\} \subset \mathcal{M} = \{z : F(z) = t\}$ ($K \in [100, 300]$) by finding opposite-class pairs $(x^{(+)}, x^{(-)})$ on opposite sides of threshold $t$ (in standardized space) and running 1D bisection along the segment $(1-\gamma)x^{(-)} + \gamma x^{(+)}$ until $|F(m_k) - t| < 10^{-3}$.
- **Geodesic Distance Approximation ($d_{\mathrm{geo}}$):** Construct a symmetric $k$-NN graph ($k_{\mathrm{nn}} = 10$) over $\{m_1, \dots, m_K\}$ weighted by Euclidean edge lengths $\|m_a - m_b\|_2$, and compute all-pairs shortest-path geodesic distances $d_{\mathrm{geo}}(m_i, m_j)$ (Dijkstra / Floyd-Warshall; fall back to penalized Euclidean if disconnected).
- **WEG Kernel (Eqs. 8–9):** With hyperparameters $\lambda > 0$ (geodesic scale, default median-heuristic $1/\mathrm{median}(d_{\mathrm{geo}})$) and $\rho > 0$ (boundary proximity weight, default $1/\mathrm{median}(\|x - m\|_2^2)$):
  $$Z(x) = \frac{1}{K}\sum_{i=1}^K \exp(-\rho \|x - m_i\|_2^2),$$
  $$k_{\mathrm{WEG}}(x, x') = \frac{1}{Z(x)\,Z(x')\,K^2}\sum_{i=1}^K \sum_{j=1}^K \exp\!\big(-\lambda\,d_{\mathrm{geo}}(m_i, m_j)\big)\,\exp\!\big(-\rho(\|x - m_i\|_2^2 + \|x' - m_j\|_2^2)\big).$$
- **GP Posterior Uncertainty Score:** Fit multi-output (independent per-feature sharing kernel matrix $K_{\mathrm{WEG}}$) GP regression $e_m = \mathcal{E}(x_m) + \eta_m$, $\eta_m \sim \mathcal{N}(0, \sigma_\eta^2)$ (use tiny jitter $\sigma_\eta^2 = 10^{-3}$ for deterministic TreeSHAP, or bootstrap variance for stochastic LIME). For test query $x_*$, compute posterior variance per feature $\sigma_{\mathrm{post}, j}^2(x_*) = k_{\mathrm{WEG}}(x_*, x_*) - k_*^\top (K_{\mathrm{WEG}} + \sigma_\eta^2 I)^{-1} k_*$ (plus diagonal observation/noise scale if modeling feature-specific variance $\tau_{m,j}^{-1}$ via empirical feature scale $\mathrm{Var}(e_{\cdot,j})$). Aggregate uncertainty as total 95% CI width / scalar SD:
  $$U_{\mathrm{GPEC}}(x_*) = \sum_{j=1}^d 2 \cdot 1.96 \, \sqrt{\hat{\sigma}_j^2 \, \sigma_{\mathrm{post}}^2(x_*)}$$
  *(or $\ell_2$-norm across features; rank-equivalent for AUROC/Spearman when kernel matrix is shared, and strictly comparable when scaling by per-feature target variance $\hat{\sigma}_j^2 = \mathrm{Var}(e_{\cdot,j})$)*.
- **Key Contrast vs $F_{\mathrm{jump}}$:** GPEC requires $O(M^3 + K^2 M)$ boundary sampling + geodesic graph + GP inversion and does not inspect internal tree structure; $F_{\mathrm{jump}}$ runs analytically in $O(|\mathcal{V}_{\mathrm{trees}}|)$ with zero boundary search or GP fitting.

### 4.3 ST-SHAP (`kelodjou2024`, AAAI 2024 — Layer-Wise Stable Kernel SHAP)

**Purpose in Benchmark:** Deterministic / low-variance coalition sampling baseline for Kernel SHAP.

**Exact Algorithm (Kelodjou et al., 2024, Section 4):**
- Given $d$ features and total coalition budget $B$ (e.g., match default Kernel SHAP `nsamples = 2*d + 2048` or budget sweep $B \in \{2d,\, 2d + 2\binom{d}{2},\, 500\}$):
- Define symmetric Layer $\ell \in \{1, \dots, \lfloor d/2 \rfloor\}$ as all binary coalition masks $z \in \{0,1\}^d$ with $|z| = \ell$ or $|z| = d - \ell$ (size $2\binom{d}{\ell}$ when $\ell < d/2$, and $\binom{d}{\ell}$ when $d$ is even and $\ell = d/2$).
- **Sampling Rule:**
  1. Iteratively saturate complete layers $\ell = 1, 2, \dots$ in order as long as remaining budget $B_{\mathrm{rem}} \ge |\text{Layer }\ell|$.
  2. Once $B_{\mathrm{rem}} < |\text{Layer }\ell^*|$ (or $B_{\mathrm{rem}} = 0$), draw the remaining $B_{\mathrm{rem}}$ coalitions uniformly **without replacement** strictly from $\text{Layer }\ell^*$ alone (never spilling random sampling across mixed higher layers).
  3. Solve the standard weighted least-squares Kernel SHAP regression with exact Shapley kernel weights $\pi_z = \frac{d-1}{\binom{d}{|z|}|z|(d-|z|)}$ (combining exact-layer analytical weights with sampled $\ell^*$ weights).
- **Deterministic Regime:** At Layer-1 budget $B = 2d$, ST-SHAP has zero stochastic sampling variance (seed stability $= 1.0$), though its *input-perturbation* stability under $X' \sim \mathcal{N}(x, \Sigma_s)$ still depends on local tree splits—providing a clean distinction between *algorithmic* stability and *input-perturbation* stability!

### 4.4 Epistemic Uncertainty Gating (`mikriukov2026`, `arXiv:2603.29915` — Cost-Aware XAI Gating)

**Purpose in Benchmark:** Direct SOTA competitor for **predicting explanation instability and gating/deferring samples** before computing expensive explanations.

**Exact Specification (Mikriukov et al., 2026, Sections 3.1, 4.2, 5.4):**
- **For Random Forest (`RF`):** Native inter-tree epistemic uncertainty:
  $$\mathcal{U}_{\mathrm{epi}}^{\mathrm{RF}}(x) \;=\; \frac{1}{T} \sum_{t=1}^{T} \big(\hat{p}_t(x) - \bar{p}(x)\big)^2, \qquad \bar{p}(x) = \frac{1}{T}\sum_{t=1}^T \hat{p}_t(x),$$
  matching `baseline_tree_variance(rf_model, X)` in `src/fragility.py`.
- **For Gradient Boosted Trees (`XGB`):** Because sequential GBDTs lack native i.i.d. posterior ensemble variance, Mikriukov et al. (2026, §3.1 & §4.2) explicitly train a **lightweight Random Forest surrogate** ($T_{\mathrm{surr}} = 100$, $\text{max\_depth} = 15$) on the training set $(X_{\mathrm{train}}, \hat{y}_{\mathrm{XGB}}(X_{\mathrm{train}}))$ (or training labels $y_{\mathrm{train}}$) and take its inter-tree prediction variance $\mathcal{U}_{\mathrm{epi}}^{\mathrm{XGB-RF}}(x)$ as the epistemic gating score. *(Optionally record a 20-seed bootstrap XGBoost ensemble variance $\mathcal{U}_{\mathrm{epi}}^{\mathrm{XGB-Boot}}(x)$ as a high-cost ablation.)*
- **Gating / Deferral Evaluation:** Defer the top-$\nu$ fraction ($\nu \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$) with highest $\mathcal{U}_{\mathrm{epi}}(x)$ (comparing deferral precision-recall for detecting unstable explanations and stability-coverage AUROC directly against $F_{\mathrm{jump}}$ and $F_{\mathrm{cal}}^{\mathrm{quant}}$).

### 4.5 AXOM (`pala2024`, `arXiv:2402.19025` — Averaging on eXplanations Of the Majority)

**Purpose in Benchmark:** SOTA tree-ensemble TreeSHAP stabilization method against which we compare (and potentially combine with) $F_{\mathrm{jump}}$-guided gating.

**Exact Algorithm (Pala & García-Cuesta, 2024, Algorithm 1):**
- **Scope:** Primarily defined for bagging ensembles (`RandomForestClassifier`).
- **Procedure for test sample $x$:**
  1. Compute ensemble predicted binary class $l_e = \mathbb{I}\!\big[\hat{p}_{\mathrm{RF}}(x) \ge t\big]$ (using the evaluation threshold $t$, e.g. $0.5$ or Bayes cost threshold).
  2. For each constituent tree $w_m \in \text{RF.estimators\_}$ ($m=1,\dots,T$), compute individual tree class prediction $l_{w_m} = \mathbb{I}\!\big[\hat{p}_{w_m}(x) \ge t\big]$.
  3. Form the majority-agreeing subset $\mathcal{W}^+(x) = \{w_m : l_{w_m} = l_e\}$. (Edge-case guard: if $\mathcal{W}^+(x) = \emptyset$, e.g. under extreme skewed threshold $t$ with tie probabilities, fall back to all trees.)
  4. Compute per-tree TreeSHAP values $\phi_{w_m}(x)$ for each $w_m \in \mathcal{W}^+(x)$ (using `shap.TreeExplainer(w_m)` with the same interventional/path mode as the full RF baseline) and return:
     $$\phi_{\mathrm{AXOM}}(x) \;=\; \frac{1}{|\mathcal{W}^+(x)|} \sum_{w_m \in \mathcal{W}^+(x)} \phi_{w_m}(x).$$
- *(Note on XGBoost: Pala & García-Cuesta 2024 only formulate AXOM for Random Forests because individual boosting trees predict log-odds residuals rather than standalone class labels. Keep AXOM strictly on RF as primary faithful baseline; do not conflate with an unvalidated boosting variant unless explicitly labeled exploratory.)*

---

## 5. BibTeX & Dataset Attribution Guardrail (`Assignment 3/paper/references.bib`)

- **Current State:** `references.bib` contains **40 verified entries** (exact ceiling of the 25–40 Assignment 3 cap; camera-ready A* cap is unlimited). All five SOTA baseline citations and the OpenML platform citation for the 3 Phase 2 external benchmarks (`1461` `bank`, `1590` `adult`, `43682` `heart`) are verified and present:
  - `zhou2021` (S-LIME, KDD 2021, DOI `10.1145/3447548.3467274`)
  - `hill2024` (GPEC, AISTATS 2024, PMLR 238:55–63)
  - `kelodjou2024` (ST-SHAP, AAAI 2024, DOI `10.1609/aaai.v38i12.29208`)
  - `mikriukov2026` (Epistemic UQ-Gating, `arXiv:2603.29915`)
  - `pala2024` (AXOM, `arXiv:2402.19025`)
  - `vanschoren2014` (OpenML platform, ACM SIGKDD Explorations 15(2):49–60, DOI `10.1145/2641190.2641198`)
- **Dataset Citation Format:** Keep `references.bib` at 40 entries for Assignment 3 compliance and cite OpenML IDs (`1461`, `1590`, `43682`) via `\cite{vanschoren2014}` in text/tables; for post-submission A\* camera-ready, individual dataset papers (`moro2014` Bank Marketing, Kohavi `1996` Adult) may be added if desired.
