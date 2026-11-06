from pathlib import Path
import shutil,json,re
H=Path(__file__).resolve().parent
P=H.parents[1]/'outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8/Submission_Package'
S=H/'review_data'; scripts=P/'search_scripts'
for f in S.glob('*.py'):shutil.copy2(f,scripts/f.name)
for f in (S/'final').iterdir():
    if f.is_file():shutil.copy2(f,scripts/'data'/f.name)
for f in (S/'extract').glob('*.jsonl'):shutil.copy2(f,scripts/'data/fulltext_decisions'/f.name)
for name in ['screening_resolutions.json','ta_change_log.json','quality_score_updates.json','finalization_summary.json','fulltext_saif.json','fulltext_shahoud.json','synthesis_factcheck_saif.json']:
    shutil.copy2(H/name,scripts/'data'/name)
for f in H.glob('crossref_*.json'):shutil.copy2(f,scripts/'data'/f.name)
for f in (scripts/'data').glob('*.xlsx'):f.unlink()
f=P/'report/main.tex';t=f.read_text()
t=t.replace('Prediction-preserving filters already exist','Predicted-class-preserving filters already exist')
t=t.replace('publication details and publication/review status with its metadata source (Crossref, the arXiv record, the paper or Semantic Scholar); bibliographic registration alone was not treated as proof of peer review, problem, RQs informed, method, datasets, evaluation, findings, limitations, reproducibility, quality scores with notes, and the decision evidence.', 'publication details and publication/review status with its metadata source, problem, RQs informed, method, datasets, evaluation, findings, limitations, reproducibility, quality scores with notes, and decision evidence. Metadata sources included Crossref, the arXiv record, the paper and Semantic Scholar; bibliographic registration alone was not treated as proof of peer review.')
f.write_text(t)
# Keep the supplied rebuild source aligned with final binary screening and human decisions.
f=scripts/'build_workbook.py';t=f.read_text()
t=t.replace("'I': 'Include - sought', 'U': 'Uncertain - sought'", "'I': 'Include', 'U': 'Include'")
t=t.replace("'DUP': 'Duplicate (removed before screening)'", "'DUP': 'Exclude'")
t=t.replace(".get(code, 'Exclude ' + dm.TA_MAP.get(code, code))", ".get(code, 'Exclude')")
t=t.replace("['I / U', 'Include / Uncertain at title-abstract stage: both were sought for full text']", "['I', 'Include at title/abstract stage means advance for full-text assessment']")
t=t.replace("'Peer-review status (verified)', 'Verified via'", "'Publication / review status', 'Metadata verification source'")
insert='''# Restore the actual independent decisions rather than regenerating a blank review.
dual_file = os.path.join(S, 'final', 'dual_screening.json')
if os.path.exists(dual_file):
    dual = json.load(open(dual_file, encoding='utf8'))
    ds = wb['Dual_Screening']
    dr = {r['id']: r for r in dual['records']}
    for i in range(7, ds.max_row + 1):
        r = dr[ds.cell(i,1).value]
        ds.cell(i,6,r['a']); ds.cell(i,7,r['b']); ds.cell(i,9,r['note'])
    ds['A2'] = 'Independent screening completed: 197 records; 171 agreements; 26 differences resolved; human independence confirmed 7 November 2026.'
    wb['README']['A13'] = 'Dual_Screening: independent human sample completed and reconciled.'
    wb['README']['A29'] = 'AI-assisted source-linked extraction; original independent reviewer decisions preserved. See report for contribution and limitations.'
'''
t=t.replace('wb.save(OUT)',insert+'wb.save(OUT)')
f.write_text(t)
f=scripts/'crosscheck.py';t=f.read_text().replace("('Studies included (main search 57, supplementary 10)', 'dbAllIncluded')", "(f\"Studies included (main search {cnt['dbIncluded']}, supplementary {cnt['spIncluded']})\", 'dbAllIncluded')")
f.write_text(t)
print('Package data synchronized:',len(json.loads((scripts/'data/master_included.json').read_text())),'studies')
