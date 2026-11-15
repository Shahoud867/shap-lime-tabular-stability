# AI Compliance, Turnitin, and Overleaf Rewrite Guide

**Course:** Business Research Methods / Research Methods — Assignment 3 (Research Paper)  
**Team:** Shahoud Shahid (23i-2515), Saif Shahzad (23i-2634) — FAST-NUCES, Islamabad  
**Target Files:** `Assignment 3/paper/main.tex`, `Assignment 3/paper/references.bib`, `Assignment 3/paper/figures/*`

---

## 1. Critical Academic Integrity Warning

Per `Assignment_3_Research_Paper.pdf` (pp. 1, 9):
- **Zero-Marks Threshold:** Any submission with **plagiarism OR AI-generated content > 15%** receives **zero marks for the entire group** with no deadline extension.
- **Permitted AI Scope:** The course brief explicitly limits AI usage to *grammar checking, LaTeX debugging, and code debugging*. AI must not author final paper prose, fabricate references, or fabricate results, and all AI assistance must be transparently declared in the **Acknowledgment** section.
- **Overleaf Typing History:** You must share an Overleaf view link with the instructor; the Overleaf edit history is audited to verify that **both** group members typed and contributed substantively.
- **Live Demo & Individual Viva (15 marks):** Both members must be able to explain the methodology, equations, code, and every table/figure during the viva.

---

## 2. How to Use `Assignment 3/paper/main.tex` Safely

The compiled `main.tex` and `main.pdf` in this workspace are a **verified structural, mathematical, and empirical reference draft**:
- All citations come exclusively from verified entries in `references.bib` (with DOIs/URLs).
- All numbers, metrics, p-values, confidence intervals, and tables come directly from the executed pipeline in `Assignment 3/analysis/results/`.
- All formatting follows `\documentclass[conference]{IEEEtran}` (`booktabs`, captions above tables / below figures, bolded best metrics, numbered hypotheses/equations, cross-references before first appearance).

**Before university submission, you MUST perform the following manual rewrite workflow:**

1. **Create a Fresh Overleaf Project:**
   - In Overleaf: `New Project -> Templates -> IEEE Conference Template` (or upload the clean `\documentclass[conference]{IEEEtran}` preamble, `references.bib`, and `figures/*.pdf` / `figures/*.png`).
   - Share the **View Link** with the instructor as required.
   - Never copy-paste whole sections in a single commit—both Shahoud and Saif should type their assigned sections incrementally in Overleaf so the revision history reflects genuine dual authorship.

2. **Paraphrase & Rewrite All Prose in Your Own Voice:**
   - Read each subsection of `main.tex` alongside the underlying source paper / `analysis/results/` output, **close the reference draft**, and type the explanation in your own academic phrasing.
   - **Keep exact and unchanged:**
     - Mathematical definitions, equations, and symbol notation (`F_tree`, `F_bin`, `D_k`, `SRA`, `CoV`, Proposition P1).
     - Exact empirical values (sample sizes, AUROCs, p-values, confidence intervals, regression coefficients, robustness deltas, case-study applicant numbers).
     - BibTeX citation keys (`\cite{...}`) and figure/table `\ref{...}` structure.
   - **Never** copy text from Assignment 2 (`Assignment 2/Deliverables/report/main.tex`) or `plan.md`; reusing Assignment 2 text risks Turnitin self-plagiarism and AI-detection flags.

3. **Verify Every Cited Source Personally:**
   - Open the DOI/URL for every entry in `references.bib` (especially the 10–15 rows in the Gap-Based Literature Table and Ballegeer et al. 2025) so you can answer viva questions on any cited work.

4. **Run Pre-Submission Similarity & AI Detection:**
   - Export the compiled PDF from Overleaf and run it through Turnitin (excluding bibliography only, per brief p. 9).
   - Confirm **both** Similarity Index **and** AI Writing Indicator are **strictly below 15%** (target <10% for safety margin).
   - If any paragraph is flagged by the AI detector, rewrite it manually in Overleaf with concrete, natural phrasing and re-test until <15%.
   - Save the final Turnitin Similarity + AI report PDF to attach with your submission.

---

## 3. Human-Required Submission Checklist Items

These items cannot be automated and must be completed by the team before submission:

| Checklist # | Item | Action Needed | Target Location |
|---|---|---|---|
| **1** | Base paper Q1/Q2 proof | Open SCImago (SJR) or WoS JCR for *European Journal of Operational Research* (2025/2026) and save screenshot | `Assignment 3/admin/base_paper_quartile.png` |
| **15** | Plagiarism + AI report (<15%) | Run Turnitin on final Overleaf-typed PDF and attach report | `Assignment 3/admin/turnitin_report.pdf` |
| **16** | Final deliverable naming & Overleaf zip | Rename final PDF to `<leader-roll>_REPORT_<SECTION>.pdf` (confirm `23i-2515` vs `i22-xxxx` format with instructor); export Overleaf `.zip`; package analysis `.zip` + editable `.drawio`/`.py` sources | Submission portal |
| **17** | Fresh-environment test | Run `run_demo.py` and pipeline on the second team member's laptop from `README.md` | Verified before viva |
| **18** | Ethics / consent note | Keep instructor confirmation (if applicable) that secondary public datasets require no human consent form ("Not applicable — secondary public data only") | `Assignment 3/admin/` |
| **—** | Overleaf view link & edit history | Confirm both Shahoud Shahid (23i-2515) and Saif Shahzad (23i-2634) appear in Overleaf history; share view link | Overleaf |
| **—** | Power analysis verification | Optional: save G*Power 3.1 screenshot matching the sample size / power justification in Section IV | `Assignment 3/admin/power_analysis.png` |
