# Archived search and decision data

The Python files and query files preserve the search workflow. `data/master_included.json` is the final 84-study master; `data/coding.json` contains the corresponding evidence map. The flat `data/` directory contains the identification, deduplication, TA, full-text and screening-resolution records. `data/fulltext_decisions/` contains the extraction/eligibility JSONL files, including the 45-record follow-up batch.

`meta_included.json`, `master_v1.json` and the older RQ map retain the 79-study baseline used to preserve identifiers. They are inputs to the merge, not competing final totals. The five verified additions and their complete metadata are in `manual_included_additions.json`; the builder appends them as S80–S84. `master_included.json` is authoritative for the final set.

## Rebuild counts, report tables and reference exports

Use Python 3.10 or newer:

```text
python3 search_scripts/rebuild.py --output /path/to/a/new/rebuild-folder
```

This reconstructs the expected `final/` and `extract/` data layout in the new folder, merges the master, regenerates coding, report tables/counts and the RIS/BibTeX/link exports. It does not repeat web searches, download inaccessible papers, rewrite narrative judgments or overwrite the submitted workbook. Compile the supplied LaTeX sources with Tectonic or a TeX installation supporting BibTeX to produce the PDF. Keep the report’s `included_studies.bib` aligned with the regenerated library copy.

`build_workbook.py` is the archived Python workbook generator, with final human decisions restored from `dual_screening.json`; the delivered workbook was assembled with the spreadsheet authoring tool and includes the additional Quality_Evidence view. The final CSV preserves that complete item-level view independently of the workbook generator.

The source exports were normalized during the original workflow. Some intermediate raw API responses and download caches are not retained. The package supports regeneration from the supplied processed decision data; it does not claim byte-for-byte replay of every historical search response. Live search results may change.

`ta_change_log.json`, `screening_resolutions.json`, `quality_score_updates.json` and the two `fulltext_*.json` files document the November reconciliation. The 197 human choices are preserved separately from the AI-assisted original and follow-up assessments. Primary Crossref metadata checks for the newly retrieved journal reports are archived as `crossref_*.json`.
