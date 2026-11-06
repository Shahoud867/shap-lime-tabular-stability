"""Prepare cell matrices from the reconciled data; this script does not author Excel files."""
from pathlib import Path
import json, importlib.util, re, collections
from openpyxl import load_workbook
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'work/finalization';DATA=HERE/'review_data'
spec=importlib.util.spec_from_file_location('submission_model',DATA/'data_model.py');dm=importlib.util.module_from_spec(spec);spec.loader.exec_module(dm)
d=dm.load(str(DATA));C=dm.counts(d);M=d['master'];byuid={r['uid']:r for r in M}
base=load_workbook(HERE/'screening_user_completed_backup.xlsx',data_only=False)
payload={'sheets':{},'patches':{},'counts':C,'included_n':len(M)}

ta=dict(d['ta']);ta.update(d['ta_supp']);ta.update(d['ta_chase']);rows=[]
for row in base['TA_Screening'].iter_rows(min_row=2,values_only=True):
    r=list(row);code,reason=ta[r[0]]
    if code=='I':r[7]='Include';r[8]=reason
    elif code=='DUP':r[7]='Exclude';r[8]='Duplicate removed before screening; retained for traceability, not counted among screened records. '+reason
    else:r[7]='Exclude';r[8]=dm.TA_MAP.get(code,code)+': '+reason
    rows.append(r)
payload['sheets']['TA_Screening']={'start':'A2','rows':rows,'columns':9}

fc={r['uid']:r for r in d['ftc']+d['ftc_supp']};fc.update({r.get('cid') or r.get('uid'):r for r in d['ftc_chase']})
rows=[]
for rid,cand in fc.items():
    r=d['ft'][rid];link='https://doi.org/'+cand['doi'] if cand.get('doi') else 'https://arxiv.org/abs/'+cand['arxiv'] if cand.get('arxiv') else ''
    rows.append([rid,'Citation searching' if rid.startswith('C') else 'Database search (supplementary)' if rid.startswith('N') else 'Database search',cand['title'],cand.get('year'),link,'Not retrieved' if r['decision']=='Not retrieved' else 'Retrieved',r.get('read_from') or '',r['decision'],r['reason'] if r['decision']=='Not retrieved' else (dm.CODES[r['reason']] if r['decision']=='Exclude' else 'Meets IC1–IC6'),r.get('evidence') or '',byuid[rid]['sid'] if rid in byuid else ''])
payload['sheets']['FullText']={'start':'A2','rows':rows,'columns':11}

def cite(r):
    authors=', '.join(r['authors'][:6])+(' et al.' if len(r['authors'])>6 else '')
    return f"{authors} ({r['year']}). {r['title']}. {r['venue']}"+(f", {r['volume']}" if r.get('volume') else '')+(f"({r['issue']})" if r.get('issue') else '')+(f", {r['pages']}" if r.get('pages') else '')+'. '+('https://doi.org/'+r['doi'] if r.get('doi') else 'https://arxiv.org/abs/'+r['arxiv'] if r.get('arxiv') else r.get('fulltext_url',''))
payload['sheets']['Included']={'start':'A2','columns':15,'rows':[[r['sid'],r['uid'],cite(r),r['year'],r['kind'],r['venue'],r.get('volume'),r.get('issue'),r.get('pages'),r.get('doi'),r.get('arxiv'),r['status'],r['src'],r['read_from'],r['source_route']] for r in M]}
payload['patches']['Included']={'L1':'Publication / review status','M1':'Metadata verification source'}
payload['sheets']['Extraction']={'start':'A2','columns':13,'rows':[[r['sid'],r['uid'],r['authors'][0].split()[-1]+f" et al. ({r['year']})",r.get('problem') or 'Not reported',r['rq'],r.get('method') or 'Not reported',r.get('data') or 'Not reported',r.get('evaluation') or 'Not reported',r.get('findings') or 'Not reported',r.get('limitations') or 'Not reported',r.get('repro') or 'Not reported',r['qa_pct'],r['evidence']] for r in M]}
qrows=[[r['sid'],r['uid']]+['N/A' if x=='NA' else x for x in r['qa']]+[r['qa_applicable'],r['qa_raw'],r['qa_pct'],'High' if r['qa_pct']>=80 else 'Moderate' if r['qa_pct']>=60 else 'Low',r['qa_notes']] for r in M]
qrows += [[None]*17,['Items']+[None]*16]+[[q,text]+[None]*15 for q,text in dm.QA_ITEMS]+[['Scoring','Yes=1; Partial=0.5; No=0; N/A is excluded. High>=80%; Moderate60–79%; Low<60%. Quality informs interpretation, not exclusion.']+[None]*15]
payload['sheets']['Quality']={'start':'A2','columns':17,'rows':qrows}

dual=json.loads((DATA/'final/dual_screening.json').read_text());changes={'A2':"Independent screening completed for 197 records; original reviewer decisions and all 26 resolution notes preserved. Human independence confirmed by the team on 7 November 2026.", 'A3':'Follow-up: 45 records advanced for retrieval; five included, three excluded after full-text assessment, 37 not retrieved. Column E retains the original pre-reconciliation first pass.'}
for i,r in enumerate(dual['records'],7):
    assert base['Dual_Screening'][f'A{i}'].value==r['id']
    assert base['Dual_Screening'][f'F{i}'].value==r['a'] and base['Dual_Screening'][f'G{i}'].value==r['b']
    changes[f'I{i}']=r['note'] or None
payload['patches']['Dual_Screening']=changes

# The stage-row order is preserved, including the existing arithmetic formulas.
P=[C['idIEEE'],C['idSD'],C['idArxiv'],C['idSS'],C['idDBLP'],C['idTotal'],C['spIEEE'],C['spArxiv'],C['spSS'],C['spDBLP'],C['spTotal'],C['dbAllId'],C['dupRemoved'],C['spDupWithin'],C['spDupMain'],C['dbAllDup'],C['dbAllScreened'],C['dbAllExcluded'],C['dbAllExOne'],C['dbAllExTwo'],C['dbAllExThree'],C['dbAllExFour'],C['dbAllExFive'],C['dbAllSought'],C['dbAllNotRetrieved'],C['dbAllAssessed'],C['dbAllFtExOne'],C['dbAllFtExTwo'],C['dbAllFtExThree'],C['dbAllFtExFour'],C['dbAllIncluded'],C['csHarvested'],C['csYear'],C['csAlready'],C['csDupWithin']+C['csDupManual'],C['csTitleFilter'],C['csScreened'],C['csTaExcluded'],C['csSought'],C['csNotRetrieved'],C['csAssessed'],C['csFtExcluded'],C['csIncluded'],C['totalIncluded']]
payload['patches']['PRISMA_Counts']={f'C{i}':v for i,v in enumerate(P,2)}
payload['patches']['PRISMA_Counts']['B32']=f"Studies included (main search {C['dbIncluded']}, supplementary {C['spIncluded']})"
readme={}
for row in base['README']:
    for cell in row:
        if isinstance(cell.value,str):
            text=re.sub(r'\b79\b',str(len(M)),cell.value)
            text=text.replace('Include / Uncertain at title-abstract stage: both were sought for full text','Include at title/abstract stage means advance for full-text assessment; it is not final inclusion')
            text=text.replace('to be completed by the team','completed; team-confirmed independent screening')
            if text!=cell.value:readme[cell.coordinate]=text
readme['A4']='Searches:24–25 October2026; source checks and screening/retrieval reconciliation:7 November2026. All counts and study tables use the supplied reconciled decision data.'
readme['A29']='The final report discloses AI assistance. The original independent reviewer columns are preserved, and additional full-text eligibility/extraction work is labelled AI-assisted.'
readme['A30']='Rebuild only from supplied data; do not overwrite original reviewer decisions. Included-study identifiers S01–S79 are retained; new inclusions are S80–S84.'
payload['patches']['README']=readme
(HERE/'workbook_payload.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
print('Payload prepared:',{k:len(v['rows']) for k,v in payload['sheets'].items()})
print('Reconciled counts:',{k:C[k] for k in ['totalScreened','totalSought','totalNotRetrieved','totalAssessed','totalIncluded']})
