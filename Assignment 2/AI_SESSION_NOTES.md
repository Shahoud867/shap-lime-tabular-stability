# Assignment 2 — AI session notes

Record of the work and decisions from the Claude Code session on this project, kept here so it carries over if you switch tools. Last updated 6 Nov 2026.

## Project overview

- **Course:** Methods in Business Research (MG 3010), FAST-NUCES Islamabad.
- **Assignment 2:** "Literature Search, Screening and Evidence Review," due **8 Nov 2026, 8:00 PM**.
- **Team:** Shahoud Shahid (23i-2515), Saif Shahzad (23i-2634).
- **Topic:** systematic review of the stability of SHAP and LIME explanations for tabular ML models (logistic regression, random forest, XGBoost), continuing the problem from Assignment 1.
- **Three RQs:**
  - RQ1 — how stability is measured, and under which perturbation/resampling regimes.
  - RQ2 — whether stable/accurate predictions indicate stable explanations, and when that link breaks.
  - RQ3 — mechanisms behind instability (split thresholds, bin edges, collinearity) and whether a cheap fragility score can predict it without running perturbations.

## Where things are

- Project root: `Assignment 2/`.
- Report source: `Deliverables/report/main.tex`, built with `pdflatex` + `bibtex` (3 pdflatex passes + 1 bibtex + 1 final pass).
- **All counts and tables in the report are generated — never hand-edit `counts.tex`, `evidence_lists.tex` or `tab_*.tex`.** They're produced by `Deliverables/search_scripts/build_tex.py` (a mirrored copy also lives at `work/build_tex.py` — keep both in sync if you edit one) from the review data in `work/final/` (`coding.json`, `search_log.json`) via `data_model.py`.
- 79 included studies (S01–S79), from 2,272 total records identified across five databases + citation searching + a supplementary RQ3 search.

## Verification done (25 Oct 2026)

- Regenerated all 9 data-driven `.tex` files from `work/final/` — byte-identical to what's shipped in `Deliverables/report/`.
- Checked 37 PRISMA arithmetic identities (record counts, dedup, EC1–EC5 exclusion breakdowns, stage totals) — all passed.
- Independently recomputed quality scores, publication-type/year tallies and RQ tallies from the master records — all matched the report's prose.
- Clean LaTeX build: no errors, no undefined citations/references. `main.pdf` is 59 pages.

## Changes made

- Replaced hard-coded synthesis numbers and `\cite{...}` lists with generator macros, so a re-coding of a study can't leave the prose stale. Added a parallel `\ev*` family of cite-wrapper macros (alongside the existing `\map*` counts and `\list*` ID-text macros) in `build_tex.py`'s `evidence_lists.tex` generation.
- Reworded two sentences that cited only a subset of studies but read as if the list were exhaustive (pre-2022 foundational studies; collinearity mechanism).
- Trimmed the thematic synthesis from 743 to 698 words to fit the brief's 500–700-word guideline (§7), keeping all four themes, the strength-of-evidence note, and the full three-part research gap statement intact.

## Still open — human work, due 8 Nov 2026

1. **Dual screening / Cohen's kappa not done.** The 15% subset (197 of 1,311 database records, seed 2515) is set up in the workbook with kappa formulas ready; both screening columns are still empty. The brief (§4.2) requires this.
2. **Contribution statement is a placeholder** in `main.tex` — currently reads "to be confirmed or edited by the team."
3. **AI-assisted decisions need human verification.** Screening, full-text and extraction/quality decisions were drafted with AI assistance and are disclosed and traceable to logged data files, which protects against an integrity issue — but the team needs to be able to defend individual decisions if asked. The brief (§8) says unexplained AI-copied work risks zero marks.

## Follow-on empirical study (discussed, not started)

The review's identified gap is a real empirical study: test, under prediction-preserving perturbations, whether the size of a prediction change indicates the size of the SHAP/LIME explanation change for logistic regression, random forest and XGBoost, beyond each explainer's own noise floor; explain instability via split-threshold/bin-edge crossing counts; and validate a tree-derived fragility score (AUROC) against observed instability without needing perturbation runs.

- **Compute:** no GPU needed. LR/RF/XGBoost and TreeSHAP/KernelSHAP/LIME are all CPU-bound at tabular-data scale; TreeSHAP is exact and fast, KernelSHAP/LIME cost scales with perturbation-sample count and parallelizes across CPU cores.
- **Timeline (AI-assisted, 2 machines in parallel):** ~3–4 weeks best case, ~5–7 weeks realistic. Parallel machines only speed up the explainer-experiment compute stage (hours). They don't speed up the iterative fragility-score design/validation loop (~1–2 weeks, inherently serial), or the time needed for the team to actually understand and verify results — those are the real bottleneck, not compute.
- **Publication outlook:** plausible but conditional. Realistic targets are XAI-focused workshops (e.g. the CCIS *Explainable Artificial Intelligence* series, NeurIPS/ICML XAI workshops, FAccT workshop track) or a regional/national conference — not a top-tier main track on this timeline. Risks: the novelty window may be closing (S52, S75, S77, S62 — 2026 arXiv preprints — are adjacent work; re-check before committing real effort), and the current review isn't publication-ready on its own (dual screening undone, ACM/Scopus/WoS excluded from coverage, AI-drafted decisions need independent verification) — it would become the related-work section of the empirical paper, not a standalone submission.
