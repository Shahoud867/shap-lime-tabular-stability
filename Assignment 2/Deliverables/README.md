# MG 3010 Assignment 2: Deliverables

Group: Shahoud Shahid (23i-2515), Saif Shahzad (23i-2634).

The main searches and citation searching were run on 24 October 2026. A supplementary search for the refined RQ3 (mechanisms of instability, and whether it can be predicted without perturbation runs) was run on 24–25 October 2026: DBLP on the 24th; IEEE Xplore, arXiv and Semantic Scholar on the 25th. ScienceDirect was tried and dropped. The supplementary round added 10 studies (S70–S79), bringing the total to 79 included studies. Study IDs S01–S69 are unchanged.

| Brief item | File |
|---|---|
| 1. Search strategy report | `report/main.pdf` (LaTeX source: `report/main.tex`) |
| 2. Reference library | `library/included_studies.bib`, `library/included_studies.ris` (79 included studies); `library/all_screened_records.ris` (all 1,381 title/abstract records with their decisions) |
| 3. Screening spreadsheet | `screening/MBR_A2_Screening_Workbook.xlsx` |
| 4. PRISMA flow diagram and checklist | `report/main.pdf`, Figure 1 and Appendix D; counts in workbook sheet `PRISMA_Counts` |
| 5. Extraction and quality table | `report/main.pdf`, Appendix C; workbook sheets `Extraction` and `Quality` |
| 6. Thematic synthesis and gap (500–700 words; currently 692) | `report/main.pdf`, Section 8 |
| 7. DOI / full-text links | `links/fulltext_links.md`, `links/included_fulltext_links.csv`, `links/reports_not_retrieved.csv` |
| 8. Contribution statement | `report/main.pdf`, Section 10 (draft for the team to confirm) |

## Still to do by the team

1. **Dual screening.** Each member fills in one column of the `Dual_Screening` sheet (197 records, a 15% random sample of the 1,311 database records, seed 2515) without looking at the other's column. The kappa cell then calculates itself. Discuss and settle disagreements in the notes column, then report the kappa in Section 6.
2. **Check the AI-assisted decisions.** Screening, extraction and quality scores were drafted with AI assistance (disclosed in Section 10). Check them against the papers, especially the borderline cases and the supplementary-search studies S70–S79.
3. **Retrieve the 52 unretrieved reports** if possible, through the HEC Digital Library (see `links/reports_not_retrieved.csv`); 10 of them come from the supplementary search. Assess any you obtain against IC1–IC6 and EC1–EC5, then rerun the build scripts so the counts update everywhere.
4. **Confirm the contribution statement** and the AI-use disclosure against course policy.

## Rebuilding

The counts and tables in the report are generated from the data files in `search_scripts/data/`. The scripts expect a working folder with `final/` (the data files) and `extract/` (the `fulltext_decisions/*.jsonl` files):

```
python coding.py <work dir>
python build_master.py <work dir>
python make_search_log.py <work dir>
python build_tex.py <work dir> ../report
python build_workbook.py <work dir> ../screening/MBR_A2_Screening_Workbook.xlsx
python build_library.py <work dir> ..
python crosscheck.py .. 
```

Then compile the report:

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Supplementary-search scripts: `supp_strings.py` (query strings), `supp_fetch.py` (IEEE, arXiv and Semantic Scholar retrieval), `supp_validate.py` (benchmark check), `supp_merge.py` (merging and de-duplication against the main search), `supp_ta.py` (title/abstract coding) and `supp_dl.py` (full-text download). The DBLP export (`data/supp_dblp_raw.txt`) was retrieved through a browser, because DBLP's bot check blocks scripted access.

**Process note.** The first round's temporary working files were deleted before the supplementary round. The working folder was rebuilt from the copies in `search_scripts/data/`, and the supplementary searches were re-run on 25 October; they returned the same records as on 24 October. The main-search log is restored from `data/search_log_v1.json`, because the raw main-search API exports were among the deleted files.
