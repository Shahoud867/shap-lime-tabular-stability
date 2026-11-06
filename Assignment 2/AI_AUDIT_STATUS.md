# Assignment 2: AI audit status — 6 November 2026

This is a record of AI-assisted review, not a claim of independent human screening or a finished submission. Two AI agents named `shahoud` and `saif` were assigned S01–S35 and S36–S79 respectively. These names identify agents, not the students.

## Files to use

- `work/audit_shahoud.md`: source checks and unresolved questions for S01–S35.
- `work/audit_saif.md`: source checks and unresolved questions for S36–S79.
- `work/audit_manifest.json`: initial file hashes and structural checks, including score-note flags.
- `work/audit_corrected/README.md`: separate corrected source draft and its limitations.
- `backups/before_ai_audit_20261106_111544/`: workbook, report and master-data snapshots taken before this audit.

## Checks completed by the coordinating agent

- The existing `crosscheck.py` passed all of its checks across the workbook, report, counts and libraries before corrections.
- Regenerating the nine data-driven TeX files in a temporary directory produced byte-identical files. The current report can be traced back to the supplied data.
- Both human input columns F/G in `Dual_Screening!F7:G203` were empty at the start. Column E was hidden. An AI audit must not fill these as if the named students had screened independently.
- The report contains all 27 PRISMA checklist items, grouped where appropriate. Some statuses remain pending; the checklist does not need to be created from scratch.
- Twenty-two studies have at least one below-1 quality score without an explicit corresponding Q-number in `qa_notes`. This is a heuristic documentation flag, not proof that evidence is absent elsewhere or that the scores are wrong. Inspect each study's limitations and evidence before deciding whether a note or score needs changing.
- Only 11 full-text PDFs are stored in `work/ft_pdfs/`. Full-text URLs may supply the remaining papers, but the existence of a DOI or an old extraction does not itself verify their contents.

## Completion-guide corrections made

The guide now distinguishes the brief's required independent group screening from its optional kappa statistic, identifies the existing full PRISMA checklist, and explains the actual source-data flow. It also distinguishes disclosure of AI use from permission to use AI.

Extraction and quality data originate in `work/extract/ft_00*.jsonl`, then feed `master_included.json` through `build_master.py`. The evidence map comes from `coding.py` and is written to `coding.json`. Editing only `coding.json` cannot correct an extracted dataset size or quality score.

## What is still required from the students

1. Both students independently screen the same 197-record subset using titles and abstracts, then discuss and document disagreements. Agent reviews cannot be represented as this human work. Kappa may be reported once genuinely calculated; no target score should be imposed.
2. Read the papers and check the audit findings and all remaining unverified entries. Record what was actually verified and by whom.
3. Confirm what each student actually did before finalizing the contribution statement. Include the later Codex audit in a truthful AI-use disclosure if it is used.
4. Establish whether the instructor permits the broader AI assistance described in the draft; disclosure alone does not answer this.
5. After accepting corrections, synchronize the originating records, mirrored data, master, workbook, generated tables, libraries where affected, and compiled report. Preserve human input. Run the consistency checker again.

## Submission status

Both agent audits are complete. The S01–S35 audit performed targeted section checks for nine studies, abstract/introduction checks for sixteen, title/contents checks for three, and could not inspect primary paper text for seven. The S36–S79 audit checked targeted full-text passages for 20 studies, abstracts only for eight, and could not access full text for 16. These categories describe access and targeted checks, not complete validation of every extraction field or quality score.

A separate corrected source draft applies 14 field corrections across 11 studies (S05, S08, S10, S18, S35, S36, S67, S72, S73, S76 and S78). It also adds S43's single-model diagnostic to the evidence map (15 diagnostic studies, four perturbation-free predictors) and narrows the proposed gap to prediction-preserving input perturbations rather than claiming that tree-derived diagnostics do not exist. All 79 study IDs and every numeric quality score are preserved; questionable scores are explicitly provisional.

The original report and workbook remain drafts until the human work and audit findings are resolved. The original master, evidence-map JSON and PDF were not overwritten by the audit. The workbook was resaved by Microsoft Excel during the audit (its file packaging changed); a cell-by-cell comparison with the backup found zero value/formula changes and both human columns still empty. That external save has been preserved. Successful arithmetic checks do not verify the papers or establish academic-policy compliance. Do not submit the original PDF as if it already incorporates subsequent audit corrections.

The corrected source draft compiled successfully with the desktop editor compiler on retry. No replacement submission PDF was exported, and final visual layout review remains outstanding.

## Separate AI screening workbook

`outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8/AI_Screening_Review.xlsx` now contains both AI agents' preliminary decisions for the 197-record sample, reasons, stored source text and explicit review provenance. At the user's request, the workbook now uses Include/Exclude only. Records with insufficient evidence to exclude advance as Include for full-text assessment. Agent shahoud: 90 Include / 107 Exclude; agent saif: 92 Include / 105 Exclude. Their binary labels differ on two records. The original three-category agent JSON is retained as historical provenance, not the current workbook decisions. The workbook describes agreement as AI-label agreement, not human agreement or kappa. All rows say AI-assisted review and human review pending. The original human F/G columns remain empty. The stored evidence has 26 missing abstracts and 38 ending in possible truncation markers.

## Latest web-checked screening version

Use `outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8/AI_Screening_Review_Web_Checked.xlsx` as the latest screening supplement. The earlier AI workbook was found to contain its older decision labels again; the new filename distinguishes this verified version. The original submission workbook has not been overwritten.

Column D contains 197 binary AI screening dispositions: 69 Include (advance for full-text assessment) and 128 Exclude. Columns E/F retain each agent's preliminary binary decision, updated by that agent's assigned web checks. The web-check disposition takes precedence in column D; otherwise records are excluded only when both agents exclude. These are AI-assisted decisions, not completed independent human screening.

All 73 records with missing, shortened or ambiguous evidence received actual web searches. Evidence obtained: 30 full abstracts, seven targeted full-text checks, 30 metadata-only matches and six unsuccessful searches. Access-only failures are not grounds for exclusion. URLs, queries, evidence summaries and access notes are recorded in the source sheet and `work/ai_screening/web_review_combined.json`. No complete-paper or successful-retrieval claim is made for inaccessible sources. The saved new workbook was reopened for validation: all three decision columns contain Include/Exclude only, all 73 web dispositions match their logs, and no standalone third-category wording appears in any cell.
