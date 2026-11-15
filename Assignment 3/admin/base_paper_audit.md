# Base Paper Audit: Ballegeer, Bogaert & Benoit (2025)

**Audit Date:** 2026-11-15  
**Audited By:** `lit-strategist`  
**Primary Sources Checked:**
- Crossref API (`https://api.crossref.org/works/10.1016/j.ejor.2025.05.039`)
- Accepted Manuscript HTML (`https://arxiv.org/html/2509.01409v1`, arXiv:2509.01409v1 [cs.LG], 01 Sep 2025)
- Vanderschueren et al. (2022b) open-source cost-matrix reference (`toonvds/CostSensitiveLearning`, `preprocessing/preprocessing.py`)

---

## 1. Bibliographic & Quartile Verification

| Field | Verified Value | Source |
|---|---|---|
| **Authors** | Matteo Ballegeer, Matthias Bogaert, Dries F. Benoit | Crossref API |
| **Title** | Evaluating the stability of model explanations in instance-dependent cost-sensitive credit scoring | Crossref API |
| **Journal** | *European Journal of Operational Research* (EJOR), Elsevier B.V. | Crossref API |
| **ISSN** | 0377-2217 | Web search / SCImago |
| **Year / Issue / Pages** | 2025 (Nov), Vol. 326, No. 3, pp. 630–640 | Crossref API |
| **DOI** | `10.1016/j.ejor.2025.05.039` | Crossref API |
| **Accepted MS** | `arXiv:2509.01409` (`https://arxiv.org/abs/2509.01409`) | arXiv |
| **SJR Quartile** | **Q1** across Management Science & Operations Research, Information Systems & Management, Industrial & Manufacturing Engineering, Modeling & Simulation, and Computer Science (miscellaneous) | Web search (SCImago-indexed summaries) |
| **SJR Score** | ~2.239 (2024) / ~2.373 (2025) | Web search |

> [!IMPORTANT]
> **Manual Team Action Required for Submission (Checklist Item #1):**  
> SCImago (`https://www.scimagojr.com/journalsearch.php?q=0377-2217&tip=iss`) blocks automated HTTP clients (`HTTP 403`). A team member must open that URL in a desktop browser, verify the current Q1 badge and SJR metric directly on SCImago, and save a full screenshot as `Assignment 3/admin/base_paper_quartile.png`. Additionally, download the published EJOR PDF and supplementary material via the HEC Digital Library if per-dataset stability numbers are needed.

---

## 2. Datasets in the Base Paper (Table 2, arXiv:2509.01409v1 §3.1)

| Dataset | Abbr. | $N$ | $D$ | Default Rate (% Pos) | Used in Our Study? |
|---|---|---:|---:|---:|---|
| Kaggle Give Me Some Credit (Credit Fusion, 2011) | GMSC | 112,915 | 10 | 6.74% | Optional (excluded from main scope per `plan.md` §4.2) |
| VUB Credit Scoring (Petrides et al., 2022) | VUB | 18,667 | 20 | 16.95% | Excluded (no public URL in base paper) |
| Home Equity (Baesens et al., 2016) | **HMEQ** | 5,960 | 12 | 19.95% | **Yes (Base-paper sample 1)** |
| South-German Credit Scoring (UCI, 2019) | **SGCS** | 1,000 | 20 | 30.00% | **Yes (Base-paper sample 2)** |
| Taiwan Default of Credit Card Clients (Yeh & Lien, 2009) | **Taiwan** | 30,000 | 23 | 22.12% | **Yes (Independent main sample; not in base paper)** |

---

## 3. Models & Instance-Dependent Cost Matrix (§3.2–3.4, Table 3)

- **Models evaluated in base paper:** `boost` (XGBoost), `csboost`, `logit` (Logistic Regression), `cslogit`, `forest` (Random Forest), `csforest`, `net` (MLP), `csnet`.
- **Our Part 1 scope (`plan.md` §4.3):** `logit`, `cslogit`, `boost`, `csboost`, and `forest` (traditional only). `csforest`, `net`, and `csnet` are omitted as out of scope and logged in the paper's Deviation Table.
- **Cost Matrix (Table 3, Eq. 1):**
  - $C_i(0 \mid 0) = C_i(1 \mid 1) = 0$.
  - $C_i(0 \mid 1) = A_i \cdot \mathrm{LGD}$, with $\mathrm{LGD} = 0.75$ and loan amount $A_i$ (`LOAN` in HMEQ, `amount` in SGCS, `LIMIT_BAL` in Taiwan).
  - $C_i(1 \mid 0) = r_i + C_{\text{alt}}$, where $C_{\text{alt}} = -\bar{r}\cdot\pi_0 + \bar{A}\cdot\mathrm{LGD}\cdot\pi_1$.
  - **Crucial detail:** $\pi_0$ and $\pi_1$ in $C_{\text{alt}}$ are fixed to the **original dataset priors** across all resampled default rates $\pi \in \Pi$, and $\bar{A}, \bar{r}$ are computed on the training set only.
  - **Verification of $r_i$ (not printed explicitly in arXiv §3.3):** Ballegeer et al. state they follow Bahnsen et al. (2014) and build on Vanderschueren et al. (2022a/2022b). In Vanderschueren et al.'s public implementation (`toonvds/CostSensitiveLearning/preprocessing/preprocessing.py`), the Bahnsen credit-scoring cost function uses:
    - $a(A_i, \text{int}_r, n) = A_i \cdot \frac{\text{int}_r (1+\text{int}_r)^n}{(1+\text{int}_r)^n - 1}$, $\text{pv}(a, \text{int}_{cf}, n) = \frac{a}{\text{int}_{cf}}\left(1 - (1+\text{int}_{cf})^{-n}\right)$, $r_i = \text{pv} - A_i$,
    - with $\text{int}_r = 0.0479/12$, $\text{int}_{cf} = 0.0294/12$, $n = 24$ months (yielding $r_i \approx 0.01918\, A_i$), and $C_i(1\mid 0) = \max(0, r_i + C_{\text{alt}})$.
- **Cost metrics:**
  - $\mathrm{AEC}(y_i, s(\mathbf{x}_i), A_i) = y_i(1-s(\mathbf{x}_i))A_i\mathrm{LGD} + (1-y_i)s(\mathbf{x}_i)(r_i + C_{\text{alt}})$ (Eq. 4).
  - $\mathrm{relAEC} = 1 - \frac{\mathrm{AEC}(y, s(\mathbf{x}), A)}{\mathrm{AEC}(y, \pi_1, A)}$ (Eq. 5).
  - $\mathrm{Savings}_f(D) = 1 - C_f(D) / \min\{C(0\mid D), C(1\mid D)\}$ (Eqs. 9–11).

---

## 4. Stability Metrics & Table 4 SRA Verification (§3.6, Table 4)

- **Coefficient of Variation (Eq. 6):** $\delta_i^{\mathrm{CoV}} = \frac{1}{P}\sum_{p=1}^P \frac{\sigma_{ip}}{\mu_{ip}}$.
- **Sequential Rank Agreement (SRA, Ekstrøm et al., 2019; Table 4):**
  - Feature importance magnitudes are ranked from $1$ (most important) to $P$ (least important).
  - Per-feature rank variance across $L$ lists: $\widehat{A_L}(p) = \frac{1}{L-1}\sum_{i=1}^L (R_i(p) - \overline{R_L}(p))^2$ (Eq. 7).
  - **Typo identified in Eq. 8 of arXiv:2509.01409v1:** Eq. 8 prints $\delta_d^{\mathrm{SRA}} = \frac{1}{|S(d)|}\sum_{p\in S(d)} \frac{(L-1)\widehat{A_L}(p)}{(L-1)|S(d)|}$, which divides by $|S(d)|$ twice.
  - **Table 4(a)–(c) exact verification:**
    - Features $\{A,B,C,D,E\}$ across $L=3$ lists have $\Phi_1 = (.20,.84,.55,.93,.12)$, $\Phi_2 = (.21,.22,.33,.65,.10)$, $\Phi_3 = (.36,.28,.89,.77,.11)$.
    - Ranks: $A:(4,4,3)\Rightarrow \widehat{A_L}=1/3\approx .33$; $B:(2,3,4)\Rightarrow 1$; $C:(3,2,1)\Rightarrow 1$; $D:(1,1,2)\Rightarrow 1/3\approx .33$; $E:(5,5,5)\Rightarrow 0$.
    - Using the standard Ekstrøm et al. formula $\delta_d^{\mathrm{SRA}} = \frac{1}{|S(d)|}\sum_{p\in S(d)} \widehat{A_L}(p)$:
      - Depth 1: $S(1)=\{C,D\} \Rightarrow (1 + 1/3)/2 = 2/3 = \mathbf{0.67}$ (matches Table 4c `.67`).
      - Depth 2: $S(2)=\{B,C,D\} \Rightarrow (1 + 1 + 1/3)/3 = 7/9 \approx 0.778$ (Table 4c prints `.77`).
      - Depth 3: $S(3)=\{A,B,C,D\} \Rightarrow (1/3 + 1 + 1 + 1/3)/4 = 2/3 = \mathbf{0.67}$ (matches Table 4c `.67`).
      - Depth 4: $S(4)=\{A,B,C,D\} \Rightarrow \mathbf{0.67}$ (matches Table 4c `.67`).
      - Depth 5: $S(5)=\{A,B,C,D,E\} \Rightarrow (1/3 + 1 + 1 + 1/3 + 0)/5 = 8/15 = \mathbf{0.533}$ (matches Table 4c `.53`).
  - **Implementation rule:** Use $\delta_d^{\mathrm{SRA}} = \frac{1}{|S(d)|}\sum_{p\in S(d)} \widehat{A_L}(p)$ at depth $d=10$, unit-tested directly against Table 4(c).

---

## 5. Hyperparameter Grid (Table 5, arXiv:2509.01409v1 §3.7)

| Model | Parameter | Candidate Values |
|---|---|---|
| `(cs)boost` | Learning Rate | `0.3` |
| | `min_child_weight` | `{0, 50}` |
| | `max_depth` | `{1, 3, 5, 7}` |
| | `colsample_bytree` | `{0.8, 1}` |
| | `gamma` | `{0, 5, 10}` |
| `(cs)logit` | `penalty` | `{L1, L2}` |
| | `C` (inverse reg. strength) | `{0, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2, 1e-1}` |
| `(cs)forest` | `max_depth` | `{None, 3, 5, 10}` |
| | `n_estimators` | `{20, 50, 100}` |
| `(cs)net` *(out of scope)* | Learning Rate / Neurons | `0.005` / `{32, 64, 128}` |

*(Note: For `logit` in `scikit-learn`, `C=0` is invalid; map `C=0` to `penalty=None` / unregularized, whereas in custom `cslogit` $\lambda = 0$.)*

---

## 6. Baseline Model Performance (Table 6, Averaged Across 4 Datasets)

| Model | AUC | AP | Brier | relAEC | Savings |
|---|---|---|---|---|---|
| `boost` | **0.837** (0.074) | 0.556 (0.184) | 0.102 (0.050) | 0.288 (0.170) | 0.475 (0.180) |
| `csboost` | 0.803 (0.079) | 0.507 (0.175) | 0.202 (0.074) | **0.554** (0.079) | **0.491** (0.121) |
| `logit` | 0.783 (0.027) | 0.475 (0.133) | 0.117 (0.041) | 0.175 (0.083) | 0.343 (0.074) |
| `cslogit` | 0.750 (0.034) | 0.371 (0.120) | 0.259 (0.061) | 0.513 (0.051) | 0.431 (0.080) |
| `forest` | 0.821 (0.045) | 0.526 (0.131) | 0.115 (0.042) | 0.167 (0.044) | 0.449 (0.091) |
| `csforest` | 0.825 (0.054) | 0.531 (0.167) | 0.117 (0.042) | 0.141 (0.036) | 0.438 (0.113) |
| `net` | 0.834 (0.063) | **0.570** (0.193) | **0.100** (0.046) | 0.306 (0.183) | 0.469 (0.169) |
| `csnet` | 0.794 (0.034) | 0.433 (0.127) | 0.258 (0.066) | 0.504 (0.046) | 0.425 (0.087) |
| `null model` | 0.500 | 0.184 | 0.143 | 0.000 | 0.000 |

---

## 7. Algorithm 2 & Known Unspecified Details (§3.8)

- **Stratified test set:** $N = 300$ instances preserving the original default rate.
- **Default-rate grid:** $\Pi = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]$.
- **Resampled training pool size:** Fixed at $\lfloor \text{\#defaults}_{\text{train}} / 0.3 \rfloor$ across all $\pi \in \Pi$ (removes sample-size confounding).
- **Tuning:** Inner 5-fold CV grid search executed **only in iteration $j=1$** (`tuning_round = True`), then reused for $j=2\dots J$ ($J=25$ in base paper; $J=10$ MVP or $J=25$ if runtime permits).
- **Explainers:** SHAP `PermutationExplainer` and LIME (`LimeTabularExplainer`, ridge regression, exponential Euclidean kernel), ranked by $|\phi|$.
- **Known gaps/unknowns to log in the paper's Deviation Table:** missing-value imputation rule (not stated; we use train-fold median/mode), random seeds, SHAP background sample size, LIME `num_samples` & discretisation flag, whether `CoV` denominator uses signed or absolute mean (use $|\mu_{ip}|$ or document exact convention), and per-dataset figures (visual-only in Figs. 4–7 of arXiv v1).
