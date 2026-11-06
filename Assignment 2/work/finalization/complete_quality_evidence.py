from pathlib import Path
import json,re,csv,importlib.util
H=Path(__file__).resolve().parent;D=H/'review_data';P=H.parents[1]/'outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8/Submission_Package'
positive=json.loads((H/'qa_positive_notes.json').read_text())
master=json.loads((D/'final/master_included.json').read_text());s45=next(r for r in master if r['sid']=='S45')['uid']
note='Q10: Sections5.2 and5.4 report50runs on50instances and sensitivity to5–100runs; random seeds or RNG control are not reported in the inspected v2 (https://arxiv.org/html/2609.00090v2). Partial credit for repetition counts.'
for f in (D/'extract').glob('*.jsonl'):
 rows=[json.loads(s)for s in f.read_text().splitlines()if s.strip()]
 for r in rows:
  if r['uid'] in positive and positive[r['uid']]not in r.get('qa_notes',''):r['qa_notes']=r.get('qa_notes','')+'; '+positive[r['uid']]
  if r['uid']==s45:
   r['qa'][9]=.5
   if note not in r['qa_notes']:r['qa_notes']+='; '+note
  if r['uid']=='U155':r['qa_notes']=r['qa_notes'].replace('score needs reassessment','score corrected from source evidence')
  r['qa_notes']=r.get('qa_notes','').replace('Appendix B.2 tests 50 instances per application with 20 repeats, so this score needs reassessment;', 'Q6: Appendix B.2 tests50instances per application with20repeats; appraisal retains partial credit because cohort uncertainty intervals are not reported;')
 f.write_text('\n'.join(json.dumps(r,ensure_ascii=False)for r in rows)+'\n')
changes=json.loads((H/'quality_score_updates.json').read_text())
if not any(r['uid']==s45 and r['item']=='Q10' for r in changes):changes.append({'uid':s45,'item':'Q10','old':1,'new':.5,'evidence':note})
(H/'quality_score_updates.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
f=P/'report/main.tex';t=f.read_text().replace('four ratings were corrected after explicit dataset, code or seed evidence was located','five ratings were corrected after dataset, code, seed or repetition evidence was checked');f.write_text(t)
print('Source notes updated; S45Q10 corrected to partial.')
