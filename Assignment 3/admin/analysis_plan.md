# Pre-Registered Statistical & Experimental Analysis Plan (v1.0)

**Status:** LOCKED prior to main experimental runs (`02_replicate.py`, `03_extend.py`, `04_robustness.py`)  
**Freeze Date:** 2026-11-15  
**Owner:** `lit-strategist` (reviewed against `Assignment 3/plan.md` §§5.2, 6.1–6.10)  
**Reproducibility Note:** Record the git commit hash of this file in the paper's reproducibility statement. Any post-freeze modification must be appended to `Assignment 3/admin/deviation_log.md` with justification.

---

## 1. Datasets, Splits, and Instance Sampling

| Dataset | Role | $N \times D$ (Pos. Rate) | Loan Exposure $A_i$ | Perturbable Continuous Features |
|---|---|---|---|---|
| **HMEQ** | Base-paper sample 1 & Case study | $5{,}960 \times 12$ ($19.95\%$) | `LOAN` | `LOAN`, `MORTDUE`, `VALUE`, `YOJ`, `CLAGE`, `DEBTINC` |
| **SGCS** | Base-paper sample 2 | $1{,}000 \times 20$ ($30.00\%$) | `amount` | `duration`, `amount`, `age` |
| **Taiwan Default** | Independent main ("own") sample | $30{,}000 \times 23$ ($22.12\%$) | `LIMIT_BAL` | `LIMIT_BAL`, `BILL_AMT1`–`BILL_AMT6`, `PAY_AMT1`–`PAY_AMT6` |

- **Preprocessing (train-fit only, zero test leakage):** One-hot encode categoricals; impute missing numerics with training-fold median and categoricals with training-fold mode; standardise continuous numerics using training-fold mean and SD.
- **Part 1 (Replication) splits:** Stratified fixed test set of $N = 300$ instances per dataset (preserving natural default rate); remainder is the training pool.
- **Part 2 (Extension) splits:** Stratified 70/30 train/test split on HMEQ and Taiwan Default; for SGCS ($N=1{,}000$), reuse the same $N = 300$ stratified test set ($70/30$ split) so SGCS test size supports $n = 200$.
- **Part 2 Instance Sampling Rule ($n = 200$ per dataset):** From each dataset's test set, draw $n = 200$ instances **stratified 50/50 by predicted class $\hat{y}(x)$** (100 predicted defaults / adverse-action candidates and 100 predicted non-defaults, or maximum available minority predictions topped up from majority predictions if fewer than 100 exist, with inverse-probability population weights recorded for reporting population-level instability rates).
- **A-Priori Power Justification (G\*Power 3.1):** Exact test — Correlation: Bivariate normal model, two-tailed, $\rho = 0.20$, $\alpha = 0.05$, power $1-\beta = 0.80 \implies n_{\min} = 193$ (approx. $194$). Setting $n = 200$ per dataset meets this threshold independently on every dataset. *(Team action: attach G\*Power screenshot as `Assignment 3/admin/power_analysis.png`.)*

---

## 2. Frozen Hyperparameters, Thresholds, and Operational Definitions

| Parameter / Symbol | Frozen Value | Sensitivity / Robustness Range | Justification |
|---|---|---|---|
| Significance level $\alpha$ | $0.05$ | — | Standard confirmatory threshold |
| Operating threshold $t$ | Dataset training default prior $\hat{\pi}_1$ | $t = 0.50$ (`R8`) | Aligns classification boundary with base-rate calibration |
| Prediction-change tolerance $\tau$ | $0.05$ ($|p(x') - p(x)| \le \tau$ and $\hat{y}(x')=\hat{y}(x)$) | $\tau \in \{0.02, 0.10\}$ (`R2`) | Standard small-probability-shift band; non-preserving draws retained as H3 contrast |
| Perturbation scale $s$ | $s \in \{0.01, 0.05\}$ ($\delta_j \sim \mathcal{N}(0, (s\cdot\mathrm{SD}_j)^2)$) | $s = 0.10$ (`R3`) | Realistic 1%–5% measurement/reporting noise on continuous features only |
| Clipping bounds | Training $[0.5\text{th}, 99.5\text{th}]$ percentiles | Unclipped for MC check of $F(x)$ | Prevents out-of-distribution tail artifacts (cf. Tolay 2026) |
| Perturbation draws per $(x, s)$ | $R = 25$ | — | Bounds worst-case binomial SE of $\bar{U}_i$ at $\le 0.10$ |
| Noise-floor repeats | $K = 10$ independent explainer seeds on unperturbed $x$ | — | $q_i = 95\text{th}$ percentile of $\{D_k(\phi^{(a)}(x_i), \phi^{(b)}(x_i))\}_{a<b}$ ($45$ pairs) |
| Top-$k$ reason-code depth | $k = 4$ (by $|\phi|$) | $k \in \{3, 5\}$ (`R4`) | Matches Regulation B Comment 9(b)(2)-1 four-principal-reasons convention |
| Instability label cut-off | $Y_i = \mathbb{I}[\bar{U}_i \ge 0.20]$ | $\{0.10, 0.30\}$ (`R4`/sensitivity) | Flags instances where $\ge 20\%$ of valid draws exceed the noise floor |
| Weak-indicator AUROC bound | $\text{AUROC upper } 95\%\text{ CI} < 0.70$ | LR+LinearSHAP internal benchmark | Conventional "acceptable discrimination" lower bound (Hosmer et al., 2013) |
| Part 1 default rates $\Pi$ | $\{0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30\}$ | — | Matches Ballegeer et al. (2025) Algorithm 2 |
| Part 1 resampled pool size | $\lfloor \#\text{defaults}_{\text{train}} / 0.30 \rfloor$ | — | Matches Ballegeer et al. (2025) §3.8 |
| Part 1 refits $J$ & SRA depth $d$ | $J = 10$ (MVP; $J = 25$ if pilot runtime fits overnight); $d = 10$ | $J = 10$ vs. $25$ (`R9`) | Replicates Ballegeer et al. (2025) §3.6–3.8 |
| Bootstrap resamples | $B = 5{,}000$ (cluster bootstrap over instances, fixed seed) | — | Satisfies brief p. 7 (5,000-resample indirect-effect CIs) |

---

## 3. Exact Mathematical Definitions

1. **Explanation Change Metrics ($\Delta E$):**
   - **Primary ($D_k$):** $D_k(x, x') = 1 - \frac{|T_k(\phi(x)) \cap T_k(\phi(x'))|}{|T_k(\phi(x)) \cup T_k(\phi(x'))|}$, where $T_k$ is the top-$k$ feature set ranked by $|\phi_j|$.
   - **Secondary (`R4` & measurement quality):** $D_\rho = 1 - \rho_S(\phi(x), \phi(x'))$ (Spearman rank distance); $D_1 = \frac{\|\phi(x') - \phi(x)\|_1}{\|\phi(x)\|_1 + \epsilon}$ (normalized $L_1$).
   - **Reason-Code Change ($RC_{ir}$):** $RC_{ir} = \mathbb{I}[D_k(x_i, x'_{ir}) > q_i]$, where $q_i$ is the 95th percentile of the noise-floor distribution (for deterministic exact explainers, $q_i = 0$ so $RC_{ir} = \mathbb{I}[T_k(\phi(x_i)) \neq T_k(\phi(x'_{ir}))]$).
   - **Instance Instability Rate & Binary Label:** $\bar{U}_i = \frac{1}{|V_i|}\sum_{r \in V_i} U_{ir}$ over prediction-preserving draws $V_i$, with $U_{ir} = \mathbb{I}[D_k(x_i, x'_{ir}) > q_i]$ and $Y_i = \mathbb{I}[\bar{U}_i \ge 0.20]$.

2. **Crossing Counts ($C_{\text{tree}}, C_{\text{bin}}$):**
   - $C_{\text{tree}}(x, x') = \sum_{t=1}^T \sum_{n \in N_t} \mathbb{I}[\text{side}_n(x_{j(n)}) \neq \text{side}_n(x'_{j(n)})]$, respecting `scikit-learn` (`x <= theta`) and `XGBoost` (`x < split`) branching conventions.
   - $C_{\text{bin}}(x, x') = \sum_{j} \sum_{e \in E_j} \mathbb{I}[\min(x_j, x'_j) \le e < \max(x_j, x'_j)]$ (edges crossed between $x_j$ and $x'_j$ for LIME discretiser bin edges $E_j$).

3. **Ex-Ante Fragility Scores ($F_{\text{tree}}, F_{\text{bin}}$) — Two Pre-Specified Variants Only (No Weight Tuning):**
   - **Unweighted Expected Crossing Count:**
     - $F_{\text{tree}}^{\text{unw}}(x; s) = \frac{1}{T}\sum_{t=1}^T \sum_{n \in N_t} \Phi\!\left(-\frac{|x_{j(n)} - \theta_n|}{\sigma_{j(n)}}\right)$ for perturbable continuous $j(n)$ ($\sigma_j = s\cdot\mathrm{SD}_j$, $0$ otherwise).
     - $F_{\text{bin}}^{\text{unw}}(x; s) = \sum_j \sum_{e \in E_j} \Phi\!\left(-\frac{|x_j - e|}{\sigma_j}\right)$.
   - **Cover-Weighted Variant:** Each node $n$ in $F_{\text{tree}}^{\text{cov}}(x; s)$ is weighted by its normalized training node cover $w_n = \text{cover}_n / \sum_{m \in N_t}\text{cover}_m$ (or tree-normalized leaf/node sample share). For $F_{\text{bin}}$, bin edges inherit uniform weights (no node cover exists).

4. **Perturbation-Free Baselines for H5:**
   - **Decision Margin Baseline:** $B_{\text{margin}}(x) = -|p(x) - t|$.
   - **Tree-Variance Epistemic Uncertainty Baseline (Mikriukov et al., 2026 style):** $B_{\text{var}}(x) = \mathrm{Var}_{t=1}^T(p_t(x))$ directly from RF trees (for XGB, optional RF surrogate variance on training predictions; primary H5 comparison against $B_{\text{margin}}$ across all models and $B_{\text{var}}$ on RF).
   - **Post-hoc Reference (not perturbation-free):** Mean $|\Delta p|$ across draws.

---

## 4. Pre-Registered Hypotheses & Statistical Testing Protocol

| ID | Pre-Registered Statement & Direction | Unit | Exact Statistical Test | Effect Size & CI | Multiplicity Control | Confirmatory Decision Rule |
|---|---|---|---|---|---|---|
| **H1a** *(Rep., Dir.)* | Explanation instability (`CoV`, `SRA`) increases as training default rate $\pi$ decreases. | Instance $\times$ rate | Jonckheere–Terpstra trend test across ordered $\pi$; Spearman $\rho(\pi, \delta)$ | Spearman $\rho$; median ratio $\pi=1\%$ vs. $30\%$ | Holm–Bonferroni across (dataset $\times$ model $\times$ explainer) | Supported if significant negative $\pi$-trend holds in majority of cells with same sign as base paper. |
| **H1b** *(Rep., Dir.)* | IDCS models (`cslogit`, `csboost`) are less stable than CE counterparts (`logit`, `boost`). | Instance (paired at $\pi$) | Paired Wilcoxon signed-rank (one-sided, IDCS $>$ CE); two-sample KS (base paper match) | Rank-biserial $r$; KS $D$ | Holm–Bonferroni across (dataset $\times$ model $\times$ $\pi$ $\times$ explainer) | Supported if IDCS is significantly less stable in majority of cells. |
| **E1** *(Est.)* | *Estimation objective:* Share of base-style across-refit attribution variance due to explainer stochasticity. | Instance (at $\pi\in\{1\%,30\%\}$, $K=5$) | Random-effects variance decomposition / ICC(1) | Explainer variance share ($95\%$ bootstrap CI) | None (descriptive) | Report point estimate + $95\%$ CI. |
| **H2** *(Ext., Dir.)* | Under prediction-preserving perturbations, $\Delta E$ exceeds the explainer's noise floor (one-sided). | Instance (paired) | Stochastic explainers (LIME, PermSHAP): one-sided paired Wilcoxon ($\overline{\Delta E}_i > q_i$ or paired median $>0$). Exact explainers (LinearSHAP, TreeSHAP): share $\bar{U}_i > 0$ with exact Clopper–Pearson $95\%$ CI. | Rank-biserial $r$; proportion $>0$ ($95\%$ CI) | Holm–Bonferroni across (dataset $\times$ model $\times$ explainer $\times s$) | Supported per cell if Holm-adjusted $p < 0.05$ (or Clopper–Pearson lower bound $> 0$ for exact). |
| **H3** *(Ext., Dir.)* | $|\Delta p|$ is a weak indicator of reason-code change $RC$ in tree models (AUROC upper $95\%$ CI $< 0.70$) and stronger in LR + LinearSHAP. | Draw (clustered by instance) | AUROC of $|\Delta p|$ predicting $RC_{ir}$; $5{,}000$-resample cluster bootstrap $95\%$ CI on **both** full perturbation set (primary) and $\tau$-filtered set. | AUROC ($95\%$ cluster bootstrap CI) | None (3 datasets $\times$ 2 tree models) | Supported for tree models if upper $95\%$ CI $< 0.70$ on the **full** set (avoiding range-restriction bias), and LR AUROC $>$ tree AUROCs. |
| **H4a** *(Ext., Med.)* | Crossing count $C$ mediates the effect of perturbation scale $s$ on $\Delta E$ (RF/XGB + TreeSHAP; discretised LIME). | Draw (clustered by instance) | Clustered linear mediation: $C_{ir} = a_0 + a s_r + u_i$; $\Delta E_{ir} = b_0 + c' s_r + b C_{ir} + v_i$; $5{,}000$-resample **cluster** bootstrap CI for $a \cdot b$. | Indirect effect $a\cdot b$ ($95\%$ CI), proportion mediated | Holm–Bonferroni across (dataset $\times$ model–explainer pair) | Supported if $95\%$ cluster bootstrap CI for $a\cdot b$ excludes $0$ in expected positive direction. |
| **H4b** *(Ext., Dissoc.)* | Double dissociation: TreeSHAP $RC$ tracks $C_{\text{tree}}$ ($\beta_{\text{tree}}>0$) not $C_{\text{bin}}$ ($\beta_{\text{bin}}\approx 0$); LIME $RC$ tracks $C_{\text{bin}}$ ($\beta_{\text{bin}}>0$) not $C_{\text{tree}}$. | Draw (clustered) | Joint logistic regression $RC_{ir} \sim C_{\text{tree},ir} + C_{\text{bin},ir}$ with cluster-robust SEs (HC/sandwich over instances). | Odds ratios ($95\%$ CI), standardized $\beta$ | Holm–Bonferroni across target paths | Supported if predicted crossover coefficient pattern holds (focal $\beta > 0, p_{\text{Holm}}<0.05$; non-focal negligible/smaller). |
| **H4c** *(Ext., Interv.)* | Disabling LIME discretisation removes bin-edge instability (for draws conditioned on $C_{\text{bin}}>0$). | Instance (paired) | Paired Wilcoxon comparing $\bar{U}_i$ (discretised ON vs. OFF) | Rank-biserial $r$ | None | Supported if instability drops significantly when discretisation is disabled. |
| **H5** *(Ext., Dir.)* | $F(x)$ discriminates unstable instances ($Y_i=1$) better than perturbation-free baselines ($B_{\text{margin}}$, $B_{\text{var}}$). | Instance | Paired AUROC difference (DeLong test & paired $5{,}000$ bootstrap) + Spearman $\rho(F, \bar{U}_i)$. | $\Delta\text{AUROC}$ ($95\%$ CI), $\rho(F, \bar{U}_i)$ | Holm–Bonferroni across (baselines $\times$ datasets $\times$ models) | Supported if $F(x)$ significantly outperforms both baselines in majority of dataset–model cells. |
| **H6** *(Expl.)* | *Exploratory bridge:* Adding $C$/$F(x)$ to base factors (dataset, model, explainer, IDCS, $\pi$) raises explained variance in $\bar{U}_i$; IDCS models have higher mean $F(x)$. | Instance | Nested OLS $M_0 \to M_3$ with HC3 SEs; $F$-test & $\Delta R^2$ ($5{,}000$ bootstrap CI). | $\Delta R^2$, $\Delta\text{AUROC}$ | Exploratory (report unadjusted & Holm) | Reported transparently as exploratory. |

---

## 5. Measurement Quality & Robustness Summary

- **Measurement Quality Analogues (reported in Table 7 of paper instead of inapplicable survey SEM $\alpha$/CR/AVE/HTMT):**
  1. *Test–retest reliability:* ICC(1) across $K=10$ repeated explainer calls on unperturbed $x_i$ (Shrout & Fleiss, 1979).
  2. *Convergent validity:* Pairwise Spearman correlations among $D_k$, $D_\rho$, $D_1$ (expected moderate-to-high positive correlation).
  3. *Discriminant validity (HTMT logic):* Monotrait correlations among $\{D_k, D_\rho, D_1\}$ must exceed heterotrait correlations between $\Delta E$ and $|\Delta p|$.
  4. *Common-method-bias analogue:* Independent random seeds for baseline $\phi(x)$ vs. perturbed $\phi(x')$, verifying shared-seed artifacts do not drive $\Delta E$.
- **Robustness Variants (`R1`–`R9`):** `R1` nested models $M_0\to M_3$ + $F(x)$ component ablation (table + forest plot); `R2` $\tau \in \{0.02, 0.05, 0.10\}$; `R3` $s \in \{0.01, 0.05, 0.10\}$ plot; `R4` $k \in \{3, 4, 5\}$ and $D_k / D_\rho / D_1$; `R5` LIME `num_samples` $1{,}000$ vs. $5{,}000$ + discretisation on/off; `R6` exact TreeSHAP vs. `PermutationExplainer`; `R7` defaulters vs. non-defaulters, per-dataset, and leave-one-dataset-out $F$-gate cut-off; `R8` $t = \text{prior}$ vs. $0.50$; `R9` $J=10$ vs. $25$ / $r_i$ margin sensitivity.
