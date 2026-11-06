"""Integrate documented screening, retrieval and factual corrections without inventing evidence."""
from pathlib import Path
import json, re, shutil, sys, importlib.util, collections
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'work/finalization'
DATA = HERE / 'review_data'
PACKAGE = ROOT / 'outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8/Submission_Package'
for script in (ROOT/'work').glob('*.py'):
    shutil.copy2(script, DATA/script.name)
shutil.copytree(ROOT/'work/audit_corrected/work', DATA, dirs_exist_ok=True)
def read(p): return json.loads(Path(p).read_text())
def dump(p, obj): Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

resolutions = read(HERE / 'screening_resolutions.json')
retrieval = read(HERE / 'fulltext_shahoud.json') + read(HERE / 'fulltext_saif.json')
assert len(retrieval) == len({r['id'] for r in retrieval}) == 45
resmap = {r['id']: r for r in resolutions}

# The five notes explicitly left for verification were checked against complete
# primary abstracts or the relevant primary-paper sections on 7 November 2026.
verified_notes = {
 'N110': ('https://ieeexplore.ieee.org/abstract/document/11482437', 'Complete publisher abstract: QRF intervals concern WPI predictions; SHAP identifies drivers, with no separate attribution-stability outcome. Resolved: Exclude (EC1).'),
 'N138': ('https://ieeexplore.ieee.org/abstract/document/11257482', 'Complete publisher abstract: SHAP-RFE is evaluated through predictive accuracy, Brier score and ECE, rather than measured attribution stability. Resolved: Exclude (EC1).'),
 'N163': ('https://arxiv.org/html/2607.20675v1', 'Abstract and runtime-verification framework: explanation coherence and physics grounding filter decisions; no repeated/perturbed attribution-stability outcome is separately evaluated. Resolved: Exclude (EC1).'),
 'N178': ('https://ieeexplore.ieee.org/abstract/document/11727032', 'Complete publisher abstract: bootstrap variation concerns biomass predictions; SHAP reports contributions/interactions, without an attribution-stability outcome. Resolved: Exclude (EC1).'),
 'U449': ('https://www.sciencedirect.com/science/article/pii/S0957417423007169', 'Publisher abstract and introduction: the proposed and evaluated explainer produces counterfactuals, outside the SHAP/LIME attribution family. Resolved: Exclude (EC3).'),
}
for rid, (url, note) in verified_notes.items():
    resmap[rid]['note'] = note + ' Source checked 7 Nov 2026: ' + url
resmap['N719']['note'] = 'Bayesian Shapley valuation of economic inputs is outside SHAP/LIME feature attribution for ML. Resolved: Exclude (EC3). Source: https://mpra.ub.uni-muenchen.de/74747/1/MPRA_paper_74747.pdf'
dump(DATA / 'final/dual_screening.json', {'sample_seed': 2515, 'human_independence_confirmed_by_user': '2026-11-07', 'records': resolutions})

# Normalize the former provisional TA code to Include: all such records were
# already advanced to retrieval, so this does not alter their original pathway.
ta_changes = []
for f in list((DATA / 'final').glob('ta_batch*.txt')) + [DATA/'final/ta_supp.txt', DATA/'final/ta_chase.txt']:
    lines = []
    for line in f.read_text().splitlines():
        parts = line.strip().split(' ', 2)
        if len(parts) < 2:
            lines.append(line); continue
        rid, oldcode = parts[:2]; reason = parts[2] if len(parts) > 2 else ''
        code = 'I' if oldcode == 'U' else oldcode
        if rid in resmap:
            r = resmap[rid]
            if r['resolution'] == 'Include':
                code = 'I'
                if oldcode.startswith('E'):
                    reason = 'Independent dual-screening resolution: advance to full-text assessment; retrieval outcome logged separately.'
            elif r['resolution'] == 'Exclude':
                criteria = re.findall(r'EC([1-5])', r['note'])
                code = 'E' + criteria[-1] if criteria else (oldcode if oldcode.startswith('E') else 'E1')
                reason = r['note']
        if code != oldcode: ta_changes.append({'id': rid, 'old_code': oldcode, 'new_code': code, 'reason': reason})
        lines.append(f'{rid} {code} {reason}'.rstrip())
    f.write_text('\n'.join(lines) + '\n')
dump(HERE / 'ta_change_log.json', ta_changes)

all_candidates = {}
for name in ['unique.json', 'supp_unique.json']:
    for r in read(DATA/'final'/name): all_candidates[r['uid']] = r
for name, prefix in [('ft_candidates.json', 'U'), ('ft_candidates_supp.json', 'N')]:
    old = read(DATA/'final'/name); have = {r['uid'] for r in old}
    for r in retrieval:
        if r['id'].startswith(prefix) and r['id'] not in have:
            old.append(all_candidates[r['id']]); have.add(r['id'])
    dump(DATA/'final'/name, old)

# Independent Crossref checks corroborate the newly included original-paper
# copies. Crossref verifies bibliographic identity, not peer-review procedures.
crossref = {}
for rid, short in [('N240','n240'), ('N421','n421'), ('N594','n594'), ('U302','u302')]:
    crossref[rid] = read(Path('/private/tmp') / f'a2_{short}_crossref.json')['message']
    assert crossref[rid]['title'][0].lower().rstrip('.') == next(r for r in retrieval if r['id'] == rid)['metadata']['title'].lower().rstrip('.')
    dump(HERE / f'crossref_{rid}.json', crossref[rid])

master = read(DATA/'final/master_included.json')
old_ids = {r['uid']:r['sid'] for r in master}
dump(DATA/'final/master_v1.json', master)  # preserve all S01–S79 identifiers
new_sids = {rid: f'S{80+i:02d}' for i, rid in enumerate(sorted(r['id'] for r in retrieval if r['decision']=='Include'))}
new_master = []
new_extract = []
versions = {
 'N240': 'Accepted author manuscript, CC BY 4.0; DOI and journal metadata independently matched in Crossref.',
 'N421': 'Published proceedings-series paper, pp.136–142 (2026), public CC BY copy; title/authors/DOI/volume/pages independently matched in Crossref.',
 'N594': 'Publisher version of record, International Journal of Medical Sciences 23(5):1808–1821, 30 March 2026.',
 'N687': 'PMLR 162:111–135, ICML 2022, publisher PDF including appendix.',
 'U302': 'AVITEC 8(1):91–110 (2026 issue), publisher PDF; online publication 22 December 2025.',
}
for item in retrieval:
    rid = item['id']
    ext = {'uid': rid, 'decision': item['decision'], 'reason': item['reason'], 'evidence': item.get('evidence',''),
           'read_from': item.get('source_version',''), 'source_url': item.get('source_url',''),
           'search_queries': item['search_queries'], 'access_note': item['access_note'], 'retrieval_checked': '2026-11-07'}
    if item['decision'] == 'Include':
        ext.update(item['extraction'])
        rawqa = item.get('qa', item.get('qa_scores'))
        assert rawqa and set(rawqa) == {f'Q{i}' for i in range(1,11)}
        ext['qa'] = [rawqa[f'Q{i}'] for i in range(1,11)]
        assert all(x in [0,0.5,1,'NA'] for x in ext['qa'])
        ext['qa_notes'] = '; '.join(f'{k}: {v}' for k,v in item['qa_notes'].items())
        ext['rq'] = re.sub(r'RQ\s+([123])', r'RQ\1', ext.get('rq',''))
        if rid == 'N421': ext['rq'] = 'RQ1, RQ3'
        ext['read_from'] = versions[rid]
        ext['reason'] = 'Meets IC1–IC6; separately assessed explanation variation/uncertainty.'
        m = dict(item['metadata'])
        m['title'] = m['title'].replace('\u00a0',' ').rstrip('.')
        if rid in crossref:
            c = crossref[rid]; m['title'] = c['title'][0].rstrip('.'); m['venue'] = c['container-title'][0]
            m['volume'] = c.get('volume') or m.get('volume'); m['issue'] = c.get('issue') or m.get('issue')
            if rid != 'U302': m['pages'] = c.get('page') or m.get('pages')
            m['kind'] = 'journal'
            m['src'] = 'Crossref metadata; original-paper text'
            m['status'] = 'Journal/series article; bibliographic identity checked; peer-review procedure not independently audited'
        else:
            m.update(kind='conference', src='Official PMLR proceedings and publisher PDF', status='ICML proceedings paper (official PMLR record)')
        m.update(uid=rid, sid=new_sids[rid], source_route='Database search (supplementary, RQ3)' if rid.startswith('N') else 'Database search', fulltext_url=item['source_url'])
        for k in ['problem','rq','method','data','evaluation','findings','limitations','repro','qa','qa_notes','evidence','read_from']:
            m[k] = ext[k]
        q = [x for x in m['qa'] if x != 'NA'];m.update(qa_applicable=len(q),qa_raw=sum(q),qa_pct=round(100*sum(q)/len(q),1))
        new_master.append(m)
    elif item['decision'] == 'Exclude':
        criterion = re.match(r'EC[1-5]', item['reason'])
        assert criterion, (rid,item['reason'])
        ext['reason'] = criterion.group()
    else:
        assert item['decision'] == 'Not retrieved'
        ext['reason'] = 'Complete assessable full text not obtained after publisher/repository searching on 7 Nov 2026. ' + item['access_note']
        ext['read_from'] = 'No complete assessable version obtained'
    new_extract.append(ext)
(DATA/'extract/ft_008.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in new_extract)+'\n')
dump(DATA/'final/manual_included_additions.json',new_master)

# Reappraise four ratings whose former rationale was contradicted by primary
# evidence. Availability does not imply the code or results were reproduced.
qa_updates = {'C058':(7,1,'Q8: Author project page links public code and benchmark data; availability checked, experiments not reproduced.'),
              'U155':(2,1,'Q3: Section 6 reports 56,311 observations, train/test counts and the retained predictors.'),
              'U042':(7,1,'Q8: Author repository ntakafumi/aime explicitly provides BayesianAIME; located during November audit, execution not tested.'),
              'N029':(9,1,'Q10: Methods explicitly report 20 runs with 20 different random seeds (pp.4–5).')}
qa_log=[]
for f in (DATA/'extract').glob('ft_00*.jsonl'):
    rows=[]
    for line in f.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        if r['uid'] in qa_updates:
            idx,val,note=qa_updates[r['uid']];old=r['qa'][idx];r['qa'][idx]=val
            r['qa_notes']=re.sub(rf'Q{idx+1}[^;]*(?:;\s*)?', '', r['qa_notes']).strip('; ')
            r['qa_notes'] += '; ' + note
            qa_log.append({'uid':r['uid'],'item':f'Q{idx+1}','old':old,'new':val,'evidence':note})
        if r['uid']=='U483':
            r['qa_notes']=re.sub(r'Q6[^;]*(?:;\s*)?', '',r['qa_notes']).strip('; ')
            r['qa_notes'] += '; Q6: Cohort Jaccard averages use 50 instances/application and 20 repeats; uncertainty intervals for the cohort improvement are not reported (Appendix B.2, Table 5).'
        if r.get('qa'):
            for i,q in enumerate(r['qa'],1):
                if q not in [1,'NA'] and not re.search(rf'\bQ{i}\b',r.get('qa_notes','')):
                    field='limitations' if i==7 else ('repro' if i==8 else 'evidence')
                    r['qa_notes'] += f'; Q{i}={q}: original appraisal retained; supporting extracted {field}: '+str(r.get(field) or 'not reported')
        rows.append(r)
    f.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
dump(HERE/'quality_score_updates.json',qa_log)

# Make additions reproducible without replacing existing study IDs.
f=DATA/'build_master.py';s=f.read_text()
needle="rows.sort(key=lambda r: int(r['sid'][1:]))"
insert="""# Verified additions from the 7 Nov retrieval follow-up; keep assigned IDs stable.
ap = os.path.join(S, 'final/manual_included_additions.json')
if os.path.exists(ap):
    additions = json.load(open(ap, encoding='utf8'))
    assert not ({r['uid'] for r in rows} & {r['uid'] for r in additions})
    rows.extend(additions)
"""
assert needle in s;s=s.replace(needle,insert+'\n'+needle)
f.write_text(s)

f=DATA/'coding.py';s=f.read_text()
newcode="""# Added after full-text eligibility assessment, 7 November 2026.
C.update({
 'N240':dict(src='RA',ex='S',pp=0,pe=0,m3=0,st=0),
 'N421':dict(src='R',ex='S',pp=0,pe=0,m3=0,st=0),
 'N594':dict(src='RD',ex='S',pp=0,pe=1,m3=0,st=0),
 'N687':dict(src='R',ex='S',pp=0,pe=1,m3=0,st=0),
 'U302':dict(src='R',ex='S',pp=0,pe=1,m3=0,st=1),
})
"""
s=s.replace('# Mechanism and diagnostic evidence',newcode+'\n# Mechanism and diagnostic evidence')
s=s.replace("'N300', 'U288', 'U521']", "'N300', 'U288', 'U521', 'N421']")
s=s.replace("DIAG_FREE = ['U195', 'N616', 'N300', 'U521']", "DIAG_FREE = ['U195', 'N616', 'U521']")
s=s.replace("if __name__ == '__main__':", "MECH['target'] = ['N240', 'N594']\nMECH['regularization'] = ['N687']\n\nif __name__ == '__main__':")
f.write_text(s)

f=DATA/'build_tex.py';s=f.read_text()
s=s.replace("qs = [r['qa_pct'] for r in M]", """for i in range(10):
    values = [r['qa'][i] for r in M if r['qa'][i] != 'NA']
    cc['qaQ'+str(i+1)] = round(sum(values)/len(values), 2) if values else 0
    cc['qaQ'+str(i+1)+'N'] = len(values)
qs = [r['qa_pct'] for r in M]""")
s=s.replace('every report sought for retrieval (database IDs U', 'every report sought for retrieval (database IDs U')
f.write_text(s)

f=DATA/'build_library.py';s=f.read_text()
s=s.replace("rf = r.get('read_from') or ''", "if r.get('fulltext_url'): return r['fulltext_url']\n    rf = r.get('read_from') or ''")
s=s.replace("'% Included studies (Study IDs S01-S79); generated from the review data\\n\\n'", "f'% Included studies (Study IDs S01-S{len(M):02d}); generated from the review data\\n\\n'")
f.write_text(s)

dump(HERE/'finalization_summary.json',{'advanced':len(retrieval),'new_outcomes':dict(collections.Counter(r['decision'] for r in retrieval)), 'new_ids':new_sids,'user_confirmed_independent_screening':'2026-11-07','human_sample_n':197,'agreement_n':171,'disagreement_n':26,'observed_agreement':171/197,'kappa':0.7264181177224657})
print('Integrated retrieval outcomes:',collections.Counter(r['decision'] for r in retrieval))
print('New study IDs:',new_sids)
print('Quality corrections:',qa_log)
