from pathlib import Path
from openpyxl import load_workbook
from pypdf import PdfReader
import json,collections,csv,re,subprocess,sys,hashlib
H=Path(__file__).resolve().parent;P=H.parents[1]/'outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8/Submission_Package'
checks=[]
def check(name,condition,detail=''):
 checks.append({'check':name,'pass':bool(condition),'detail':detail})
 if not condition:print('FAIL',name,detail)
w=load_workbook(P/'screening/MBR_A2_Screening_Workbook.xlsx',data_only=False);v=load_workbook(P/'screening/MBR_A2_Screening_Workbook.xlsx',data_only=True);old=load_workbook(H/'screening_user_completed_backup.xlsx')
m=json.loads((P/'search_scripts/data/master_included.json').read_text());n=len(m)
check('Final master has84 unique Study IDs and record IDs',n==84 and len({r['sid']for r in m})==84 and len({r['uid']for r in m})==84)
check('Original79 Study IDs retained',all(next(x['sid']for x in m if x['uid']==r['uid'])==r['sid']for r in json.loads((P/'search_scripts/data/master_v1.json').read_text())))
check('Included and Extraction have84 records',all(w[s].max_row==85 for s in ['Included','Extraction']))
ft=collections.Counter(r[7]for r in v['FullText'].iter_rows(min_row=2,values_only=True));check('194 full-text outcomes reconcile',ft=={'Include':84,'Exclude':21,'Not retrieved':89},dict(ft))
check('TA decisions are binary',{r[7]for r in v['TA_Screening'].iter_rows(min_row=2,values_only=True)}=={'Include','Exclude'})
check('Original197 independent reviewer decisions preserved',all(old['Dual_Screening'].cell(r,c).value==w['Dual_Screening'].cell(r,c).value for r in range(7,204)for c in [6,7]))
check('Both reviewers complete with binary decisions',all(w['Dual_Screening'].cell(r,c).value in['Include','Exclude']for r in range(7,204)for c in[6,7]))
dis=[r for r in range(7,204)if w['Dual_Screening'].cell(r,6).value!=w['Dual_Screening'].cell(r,7).value]
check('All26 disagreements have resolutions',len(dis)==26 and all('Resolved:'in str(w['Dual_Screening'].cell(r,9).value)for r in dis))
check('Cached agreement statistics match independent calculation',v['Dual_Screening']['B4'].value==197 and abs(v['Dual_Screening']['B5'].value-171/197)<1e-12 and abs(v['Dual_Screening']['F5'].value-.7264181177224657)<1e-12)
check('No cached Excel formula errors',not[(s.title,c.coordinate)for s in v for row in s for c in row if c.data_type=='e'])
check('Quality and linked extraction formulas agree with master',all(abs(v['Quality'].cell(i,15).value-r['qa_pct'])<.01 and abs(v['Extraction'].cell(i,12).value-r['qa_pct'])<.01 for i,r in enumerate(m,2)))
check('All840 quality item records populated and match scores',v['Quality_Evidence'].max_row==841 and all(all(v['Quality_Evidence'].cell(i,c).value is not None for c in range(1,9))and v['Quality_Evidence'].cell(i,5).value==('N/A'if m[(i-2)//10]['qa'][(i-2)%10]=='NA'else m[(i-2)//10]['qa'][(i-2)%10])for i in range(2,842)))
check('Unrelated identification/search/dedup/validation values preserved',all(c.value==w[s][c.coordinate].value for s in ['Search_Log','Refinement_Log','Validation','Identification','Dedup']for row in old[s]for c in row))
check('Original panes and reviewer validation preserved',all(old[s].freeze_panes==w[s].freeze_panes for s in old.sheetnames)and len(w['Dual_Screening'].data_validations.dataValidation)==len(old['Dual_Screening'].data_validations.dataValidation))
check('Native filters cover final record ranges',w['FullText'].auto_filter.ref=='A1:K195'and w['Included'].auto_filter.ref=='A1:O85'and w['Quality_Evidence'].auto_filter.ref=='A1:I841')
check('PRISMA arithmetic checks all true',all(row[2]is True for row in v['PRISMA_Counts'].iter_rows(min_row=48,max_row=53,values_only=True)))
linkrows=list(csv.DictReader((P/'links/included_fulltext_links.csv').open(encoding='utf-8-sig')));check('84 included-study link records',len(linkrows)==84)
pdf=PdfReader(P/'report/main.pdf');text='\n'.join(p.extract_text()or''for p in pdf.pages)
check('PDF contains all84 study references and no missing citations',all(re.search(r'\['+str(i)+r'\]',text)for i in range(1,85))and '[?]'not in text)
check('PDF has no stale appraisal TODO',not re.search(r'needs reassessment|to be completed by the team',text,re.I))
report=(P/'report/main.tex').read_text();synth=report.split('\\section{Thematic synthesis and research gap}')[1].split('\\section{Limitations}')[0]
synth=re.sub(r'\\cite\{[^}]*\}|\\label\{[^}]*\}','',synth);synth=re.sub(r'\\[A-Za-z]+','',synth);wordcount=len(re.findall(r"\b[\w'-]+\b",synth))
check('Thematic synthesis within500–700 words',500<=wordcount<=700,wordcount)
out=subprocess.run([sys.executable,str(P/'search_scripts/crosscheck.py'),str(P)],capture_output=True,text=True);(H/'crosscheck_final.txt').write_text(out.stdout+out.stderr);check('All cross-format count/reference/PRISMA checks pass','ALL PASS'in out.stdout and 'FAIL 'not in out.stdout)
reb=Path('/private/tmp/a2_submission_rebuild_check');check('Archived-data rebuild reproduces final master and counts',json.loads((reb/'review_data/final/master_included.json').read_text())==m and(reb/'report/counts.tex').read_text()==(P/'report/counts.tex').read_text())
check('Recency targets satisfied',sum(r['year']>=2022 for r in m)>=42 and sum(r['year']>=2025 for r in m)>=5)
data={'date':'2026-11-07','all_pass':all(c['pass']for c in checks),'checks':checks,'included':84,'screened':1375,'sought':194,'not_retrieved':89,'assessed':105,'fulltext_excluded':21,'human_sample':197,'agreements':171,'disagreements_resolved':26,'kappa':.7264181177224657,'pdf_pages':len(pdf.pages),'synthesis_word_count':wordcount}
(H/'validation_results.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items()if k!='checks'},indent=2));print('Passed',sum(c['pass']for c in checks),'of',len(checks))
if not data['all_pass']:raise SystemExit(1)
(P/'VALIDATION_REPORT.md').write_text('# Final validation — 7 November 2026\n\nAll '+str(len(checks))+' validation checks passed.\n\n'+''.join('- PASS: '+c['check']+('. '+str(c['detail'])if c['detail']else'')+'\n'for c in checks)+f'\nPDF: {len(pdf.pages)} pages including references and appendices; the main narrative fits the suggested seven-page limit. Synthesis: approximately {wordcount} words.\n\nWorkbook formulas were recalculated, cached outputs checked, changed views rendered and inspected, and original reviewer inputs compared with the source workbook. All PDF pages were inspected in contact sheets, with final revised narrative pages checked separately. The supplied processed data successfully regenerated the final84-study master, counts, evidence map, tables and library exports.\n\nThese checks establish file consistency and completeness, not independent replication of the included studies. Human independence for the197-record sample is based on the team’s confirmation; extraction and quality appraisal remain AI-assisted and are disclosed in the report. Full-text access failures and methodological limitations remain visible. The files have not been submitted online.\n')
