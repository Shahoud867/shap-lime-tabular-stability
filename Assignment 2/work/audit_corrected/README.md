# Corrected AI-audit source draft

Prepared 6 November 2026. This folder is a separate draft, not a final submission package. The original `Deliverables/` report, workbook and libraries have not been replaced.

## What changed

`applied_corrections.json` records 14 evidence-backed field corrections across 11 studies, including the original values, replacement values and sources. The corrected originating extraction records and rebuilt master live in `work/`. The regenerated report sources live in `report/`.

The evidence map additionally includes S43's split-count diagnostic. Diagnostic counts change from 14 to 15 and perturbation-free predictor counts from three to four. The synthesis now distinguishes ranking instability across retrained models from local explanation instability under prediction-preserving input changes. It does not claim that tree-derived diagnostics are unprecedented.

Study IDs, inclusion decisions and numeric quality scores are unchanged. The quality scores remain provisional because targeted source checks did not constitute a complete independent appraisal. Notes identify where the original rationale was contradicted by the source. Repositories were not executed or used to reproduce experimental results.

The contribution statement still requires the actual students' confirmation. The disclosure identifies the two named agents as AI agents and keeps human screening pending. No kappa or human-review result has been fabricated.

## What this folder does not contain

- A completed human screening workbook.
- A synchronized submission workbook or reference library: the original workbook and libraries still describe the original draft.
- A claim that every study or quality score has been verified.
- A final exported PDF. Compilation diagnostics, if available, do not by themselves export a replacement PDF.

Do not mix this draft's changed diagnostic counts with the original workbook. After human review, promote accepted changes consistently into the originating extraction records and mirrored data, rebuild the master and evidence map, update the affected workbook cells without overwriting human input, regenerate tables and libraries where affected, and rebuild/export the report.

## Validation completed

All 79 study IDs are retained, all numeric Q1–Q10 vectors match the original, and the revised diagnostic map includes S43. The source generator ran successfully. The desktop editor compiler confirmed successful compilation on retry after its first attempt timed out while downloading supporting files. This validates compilation, not final PDF layout or an exported submission PDF. The existing word-count helper reports 664 words for the revised synthesis; counts depend on treatment of citations and compound words. The original package passed its consistency checker before this audit, and all nine original generated TeX files were reproduced exactly in a temporary directory.

See `../../AI_AUDIT_STATUS.md` for overall status, `../audit_shahoud.md` for S01–S35 and `../audit_saif.md` for S36–S79.
