# Manual completion guide — Assignment 2

Step-by-step instructions for the work that has to be done by Shahoud and Saif themselves, not by the AI assistant. Tier 1 is required for the 8 Nov 2026, 8:00 PM submission. Tier 2 is extra work only worth doing if this review is going to feed the follow-on empirical study or stand on its own as something publishable (see [AI_SESSION_NOTES.md](AI_SESSION_NOTES.md)).

**Before you start anything:** make a dated backup copy of `Deliverables/screening/MBR_A2_Screening_Workbook.xlsx` (e.g. `MBR_A2_Screening_Workbook_2026-11-06.xlsx` in a `backups/` folder). The workbook's `TA_Screening`, `FullText`, `Extraction` and `Quality` sheets are **generated** from `work/final/coding.json` and `search_log.json` by `Deliverables/search_scripts/build_workbook.py` — if anyone reruns that script after you've hand-corrected a cell or filled in `Dual_Screening`, your edits are wiped. Until this assignment ships, nobody should run `build_workbook.py` again. `build_tex.py` (which only reads the JSON, not the workbook) is safe to rerun.

---

## Tier 1 — required before submission

### Step 1: Independent dual screening + Cohen's kappa

**Why:** the brief (§4.2) requires a second, independent screener on a sample and a reported kappa, not just the AI-assisted single pass. This can't be delegated — the entire point is two humans deciding without seeing each other's answers.

**File:** `Deliverables/screening/MBR_A2_Screening_Workbook.xlsx` → sheet `Dual_Screening`, rows 7–203 (197 records, a `random.Random(2515).sample()` of the 1,311 database records). Columns: A=ID, B=Title, C=Year, D=DOI/arXiv, E=AI-assisted first pass (hidden), F=Shahoud, G=Saif, H=Agree? (auto), I=Notes on disagreement.

1. Open the workbook. Right-click the column headers and confirm column **E is hidden**. If it's visible, select it and hide it again — seeing the AI's answer first defeats the independence.
2. The sheet only has Title/Year/DOI, not the abstract. For each row, open the DOI or arXiv link in column D and read the abstract from the publisher page, Semantic Scholar, or arXiv — the same way the original screening would have.
3. Apply the same eligibility criteria used for the original title/abstract screening (reproduced in `main.tex`, §4, Table "Eligibility criteria"):
   - **EC1** stability asserted but not measured (e.g. folded into a composite score, no separate results) → exclude
   - **EC2** non-tabular data only (images, text, signals, graphs, code) → exclude
   - **EC3** explainer outside the SHAP/LIME family, or only cross-method agreement rather than stability → exclude
   - **EC4** not a full research report (abstract/poster/editorial) or not in English → exclude
   - **EC5** "stability" used in a physical/chemical/engineering sense (homonym) → exclude
   - Otherwise → include
4. **Shahoud fills column F, Saif fills column G.** Do this independently — don't discuss records with each other while screening, and don't look at the other person's column until both are completely done. Working in the same sitting but on separate copies of the sheet (then pasting one column in) avoids accidental glances.
5. When every one of the 197 rows has both F and G filled, open the sheet's top-left summary block: `n complete` (should read 197), `Observed agreement`, `Expected agreement`, and `Cohen's kappa` will stop showing "pending" and calculate automatically — the formulas are already in place.
6. For every row where column H says "No" (disagreement), discuss that specific record together, agree on a final decision, and write the reasoning in column I. This resolution log is what you'd point to if an examiner asks why a given record was included or excluded.
7. Interpret the kappa value using the standard Landis & Koch bands: >0.80 almost perfect, 0.61–0.80 substantial, 0.41–0.60 moderate, <0.40 poor. Report whatever you get honestly — if it's low, say so and note likely causes (ambiguous criteria, screener fatigue) rather than adjusting decisions to inflate it.
8. Report the kappa figure, n, and agreement % in the report. Check `main.tex` around the dual-screening/limitations text (§Limitations currently says "Dual screening and the kappa statistic are pending") and replace that sentence with the actual result once you have it. This is a hand-edit to prose, not to a generated `.tex` file, so it's fine to do directly.
9. Time estimate: title+abstract only, no full text — roughly 2–4 hours per person for 197 records if you keep a steady pace.

### Step 2: Verify the AI-drafted screening, extraction and quality decisions for your half

**Why:** the brief (§8) says unexplained AI-copied work risks zero marks. Regenerating the `.tex` files confirmed the report's *arithmetic* is internally consistent with the logged data — it does not confirm the underlying judgments (why a study was included, what its method was, what score it got) are actually correct. You need to be able to defend your half if asked.

**Files:** workbook sheets `FullText` (decision + criterion + page/section evidence for every full text sought), `Extraction` (problem, method, dataset, findings, limitations, reproducibility per included study), `Quality` (Q1–Q10 scores per included study). Split: **Shahoud verifies S01–S35, Saif verifies S36–S79** (per the draft contribution statement — confirm this split in Step 3).

1. In the `FullText` sheet, filter or sort by `Study ID` to isolate your assigned range.
2. For each study, open the actual source — `work/ft_pdfs/` has the PDFs, `work/ft_txt/` has extracted text, or use the link in the `Link` column — and check three things against the workbook row:
   - The **Decision** (include/exclude) actually follows from the stated **Criterion/reason**, and that reason is true of the paper.
   - The **Evidence (page/section)** cited actually contains what it's credited with supporting — don't take the page number on faith.
3. In the `Extraction` sheet for the same study, re-read the paper's abstract/method/results and check the Problem/Method/Dataset(s)/Evaluation/Key findings/Limitations/Reproducibility columns aren't paraphrased in a way that drifts from what the paper says. Pay particular attention to **Key findings** — this is the field most likely to be subtly wrong without being obviously wrong.
4. In the `Quality` sheet, re-score Q1–Q10 yourself from the paper (Yes=1, Partial=0.5, No=0, N/A) using the definitions in `main.tex` §"Data extraction and quality appraisal": Q1 clear aims, Q2 replicable method, Q3 datasets described, Q4 appropriate comparison, Q5 well-defined stability measure, Q6 evidence sufficient (scale/repetitions/statistics), Q7 limitations discussed, Q8 code/data available, Q9 ethics/privacy/regulation addressed where relevant, Q10 randomness controlled and reported. Compare your score to the logged one — recheck anything that surprises you, especially a score of 1 or 0 you wouldn't have given yourself.
5. If you find an error, **don't edit the generated `.tex` tables or `counts.tex` directly** — they'll just be overwritten next time someone regenerates them, and the report prose (means, tallies) would silently go stale. Instead:
   - Correct the value in the workbook cell (for your own record and for the between-you discussion), **and**
   - Find the same record in `work/final/coding.json` and correct it there too, since that JSON (not the workbook) is what `build_tex.py` actually reads to produce the report.
6. After any correction to `coding.json`, regenerate the `.tex` files so the report prose and tables match: run `build_tex.py` with the same arguments it was run with before (check `work/build_tex.py`, the mirrored copy, for the exact invocation), then rebuild the PDF (3× `pdflatex`, 1× `bibtex`, 1× `pdflatex` on `main.tex`). Confirm no new undefined-reference or citation warnings in `main.log`.
7. Time estimate: roughly 35–44 studies each; at ~10–15 minutes per study for a confidence check (not a from-scratch re-extraction), that's about half a day each.

### Step 3: Finalize the contribution statement

**Why:** `main.tex` §"Contribution statement and AI-use disclosure" (around line 177–178) currently opens with *"Proposed division, to be confirmed or edited by the team before submission."* That's a flag for exactly this task, not a finished statement.

1. Read the current draft text in `main.tex`. It currently proposes: Shahoud leads RQ1, owns IEEE Xplore/ScienceDirect/DBLP searches, confirms title-abstract decisions, verifies extraction/quality for S01–S35; Saif leads RQ2, owns arXiv/Semantic Scholar searches and citation searching, performs the independent dual screening, verifies S36–S79; both do RQ3, its supplementary search, synthesis and final edit.
2. Decide together whether that division matches what actually happened (adjust if, say, the dual-screening split in Step 1 or the verification split in Step 2 ended up different in practice).
3. Edit the sentence to remove "Proposed division, to be confirmed or edited by the team before submission" and state the agreed-upon, final division as fact.
4. Keep the AI-use disclosure sentence that follows — it's accurate (an AI assistant ran the documented searches, deduplicated records, drafted screening/extraction/quality decisions, and drafted the report; no invented sources or counts) — but it should now also truthfully reflect that the team verified the decisions, once Step 2 is actually done. Don't claim verification you haven't completed.
5. Both of you should read and agree to the final wording before it ships — this is the paragraph you're putting your names to for the AI-use policy.

### Step 4: Decide what to do about the retrieval gaps (optional but cheap)

**Why:** `main.tex` §Limitations already discloses these honestly, so this step is not required to submit — but closing a few of them cheaply strengthens the review.

1. Open `Deliverables/links/reports_not_retrieved.csv` (52 rows) — mostly paywalled IEEE conference papers and open-access papers behind bot checks.
2. If either of you has HEC Digital Library / institutional access, try pulling 2–3 of the highest-value ones (check against which studies are cited most or are closest to your RQs).
3. The 5 benchmark papers the documented search missed (`hwang2025`, `decker2025`, `lin2023`, `takefuji2025`, `hasnain2026` — see §Limitations point 2) were deliberately **not** added, because the search as documented didn't retrieve them and adding them by hand would misrepresent the search's actual yield. Leave this as-is; it's honestly disclosed and flagged for a future search update, which is the correct call for a reproducible review. Don't quietly slip them into the included set.

---

## Tier 2 — only if this is going toward real publication

Do these only if you're carrying the review into the follow-on empirical study as a related-work section, or submitting it somewhere on its own. None of this is needed for the 8 Nov grade.

### Step 5: Add the missing databases

ACM Digital Library, Scopus and Web of Science were never searched (OpenAlex was piloting-only). This is the single biggest credibility gap for an external reviewer.

1. Get institutional access to each (library proxy login).
2. Re-run the same query logic documented in `Deliverables/report/tab_searchlog.tex` / the `Search_Log` workbook sheet, adapted to each database's syntax.
3. Screen, dedupe against the existing 2,272-record pool, and fold any new includes into `work/final/coding.json`, then regenerate.

### Step 6: Independent re-extraction sample

Beyond the screening kappa, pick a 10–15% random sample of the 79 included studies and have both of you independently re-extract the Quality (Q1–Q10) scores without seeing each other's or the logged scores, then compute agreement the same way as Step 1. This is what a methodologically serious review reports for extraction reliability, not just screening reliability.

### Step 7: Fill out the PRISMA 2020 checklist explicitly

Most venues expect the checklist itself as a supplementary file, not just a report that happens to follow its structure. Go through the official 27-item PRISMA 2020 checklist and note, item by item, where in `main.tex` each is satisfied (the Appendix already has a partial version — check `main.tex` around line 228, "Eligibility criteria & Section 4" — extend this table to all 27 items).

### Step 8: Full citation audit

Go through every `\cite{}` in `main.tex`'s synthesis section (§7) and confirm the cited study actually supports the specific claim attached to it — not just that it's topically related. Watch especially for sentences that cite 3–5 studies together; confirm each one individually supports the claim, not just the majority.

### Step 9: Human prose pass

Reread §7 (Thematic synthesis) and §Limitations line by line in your own voice. AI-drafted academic prose tends toward hedge-everything phrasing that reads fine but isn't necessarily the judgment you'd actually stand behind. Rewrite anything that doesn't match what you'd say if asked to defend it out loud.

### Step 10: Decide on protocol registration

If you're taking this toward a standalone submission (not just related work for the empirical paper), consider registering a protocol on PROSPERO or OSF before doing Step 5's additional searches — reviewers increasingly expect this, and it has to happen before the extra searches, not after.
