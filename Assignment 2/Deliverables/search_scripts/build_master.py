"""Merge verified metadata, overrides and extraction records into one master list of included studies with Study IDs."""
import json, glob, os, sys, re, html, unicodedata
S = sys.argv[1]
meta = json.load(open(os.path.join(S, 'final/meta_included.json'), encoding='utf8'))
ext = {}
for f in sorted(glob.glob(os.path.join(S, 'extract/ft_00*.jsonl'))):
    for line in open(f, encoding='utf8'):
        if line.strip():
            r = json.loads(line); ext[r['uid']] = r

# Venue / status for records with no usable Crossref entry, taken from arXiv comments, Semantic Scholar or the PDF itself.
OV = {
 'U498': dict(kind='workshop', venue='ICLR 2022 Workshop (non-archival); arXiv:2203.06877', year=2022, status='Workshop paper (non-archival); arXiv version', src='arXiv record; authors\' statement'),
 'U481': dict(kind='workshop', venue='ICLR 2023 Tiny Papers track; arXiv:2204.11351', year=2023, status='Peer-reviewed workshop track (per DBLP and Semantic Scholar)', src='DBLP; Semantic Scholar'),
 'U491': dict(kind='conference', venue='Proc. 41st Conference on Uncertainty in Artificial Intelligence (UAI 2025), PMLR 286', pages='1476-1496', year=2025, status='Peer-reviewed conference (per Semantic Scholar); arXiv v5 read', src='Semantic Scholar; arXiv'),
 'U524': dict(kind='preprint', venue='arXiv:2608.02697 (accepted at EXPLAINS 2026 per arXiv comment)', year=2026, status='Accepted, not yet published (authors\' statement); preprint read', src='arXiv comment'),
 'U528': dict(kind='preprint', venue='arXiv:2609.00090 (accepted at XKDD 2026 workshop, ECML-PKDD, per arXiv comment)', year=2026, status='Accepted, not yet published (authors\' statement); preprint read', src='arXiv comment'),
 'C024': dict(kind='conference', venue='Explainable Artificial Intelligence (xAI 2025), Communications in Computer and Information Science, Springer', pages='28-52', year=2025, doi='10.1007/978-3-032-08324-1_2', status='Peer-reviewed conference (Crossref title match)', src='Crossref search; arXiv'),
 'C045': dict(kind='journal', venue='Expert Systems with Applications', volume='246', pages='123154', year=2024, doi='10.1016/j.eswa.2024.123154', status='Peer-reviewed journal (Crossref)', src='Crossref; arXiv journal_ref'),
 'C048': dict(kind='conference', venue='Proc. 27th International Conference on Artificial Intelligence and Statistics (AISTATS 2024), PMLR 238', pages='1378-1386', year=2024, status='Peer-reviewed conference (PDF footer; Semantic Scholar)', src='PDF; Semantic Scholar'),
 'C049': dict(kind='conference', venue='Proc. 27th International Conference on Artificial Intelligence and Statistics (AISTATS 2024), PMLR 238', pages='55-63', year=2024, status='Peer-reviewed conference (PDF footer; arXiv comment)', src='PDF; arXiv comment; Semantic Scholar'),
 'C058': dict(kind='conference', venue='Advances in Neural Information Processing Systems 34 (NeurIPS 2021)', pages='9391-9404', year=2021, status='Peer-reviewed conference (Semantic Scholar)', src='Semantic Scholar'),
 'C061': dict(kind='workshop', venue='AdvML \'20: Workshop on Adversarial Learning Methods for Machine Learning and Data Mining (KDD 2020); arXiv:2006.12302', year=2020, status='Workshop paper (PDF header)', src='PDF; arXiv comment'),
 'C068': dict(kind='workshop', venue='ICML 2019 Workshop on AI for Social Good; arXiv:1904.12991', year=2019, status='Workshop paper (PDF header)', src='PDF'),
 'C069': dict(kind='workshop', venue='ICML 2018 Workshop on Human Interpretability in Machine Learning (WHI); arXiv:1806.08049', year=2018, status='Workshop paper (arXiv comment)', src='arXiv comment'),
 'U521': dict(kind='preprint', venue='arXiv:2605.21492 (also Zenodo, doi 10.5281/zenodo.19468379)', year=2026, status='Preprint (not peer reviewed)', src='arXiv'),
}
AUTH_OV = {'U362': ['Asmaa Elgezawy', 'Hatem Abdul-kader', 'Asmaa Elsaid'], 'U221': ['Mahendra', 'Aparna Khare']}
TITLE_OV = {'U483': 'S-LIME: Stabilized-LIME for Model Explanation', 'U221': 'An Empirical Evaluation of Explainability Drift Under Post-Training Quantization of Machine Learning Models'}

def clean(s):
    return html.unescape(re.sub(r'&lt;/?p&gt;|</?p>', '', s or '')).strip()

rows = []
for u, m in meta.items():
    cr = next((v for v in m['crossref'].values() if 'error' not in v), None)
    crdoi = next((k for k, v in m['crossref'].items() if 'error' not in v), None)
    ax = m.get('arxiv') or {}
    r = {'uid': u}
    if cr:
        kind = {'journal-article': 'journal', 'proceedings-article': 'conference', 'book-chapter': 'conference'}.get(cr['type'], cr['type'])
        r.update(title=clean(cr['title']), authors=cr['authors'], venue=clean(cr['container']), event=cr.get('event'), volume=cr.get('volume'),
                 issue=cr.get('issue'), pages=cr.get('page'), year=cr.get('year'), doi=crdoi, kind=kind, publisher=cr.get('publisher'),
                 status={'journal': 'Peer-reviewed journal (Crossref)', 'conference': 'Peer-reviewed conference proceedings (Crossref)'}.get(kind, kind), src='Crossref')
    else:
        r.update(title=ax.get('title') or m['cand_title'], authors=ax.get('authors') or [], venue='arXiv:' + (m.get('arxiv_id') or ''),
                 year=int((ax.get('published') or '0000')[:4]) or m['cand_year'], doi=None, kind='preprint', status='Preprint (not peer reviewed)', src='arXiv')
    if m.get('arxiv_id'): r['arxiv'] = m['arxiv_id']
    ct = clean(m.get('cand_title') or '')
    if ct and len(ct) > len(r['title']) + 3 and ct.lower().startswith(r['title'].lower()[:12]): r['title'] = ct
    r['venue'] = (r.get('venue') or '').replace('proceesdings', 'proceedings')
    if u in OV: r.update(OV[u])
    if u == 'U373': r['venue'] = 'Machine Learning and Knowledge Extraction (CD-MAKE 2021), Lecture Notes in Computer Science, Springer'
    if u == 'U556': r['venue'] = 'Explainable Artificial Intelligence (xAI 2024), Part II, Communications in Computer and Information Science, Springer'
    if u == 'C024': r['venue'] = 'Explainable Artificial Intelligence (xAI 2025), Part II, Communications in Computer and Information Science, Springer'
    if u == 'U546': r['kind'] = 'conference'; r['status'] = 'Peer-reviewed conference proceedings (Crossref)'
    if u == 'N708': r['venue'] = 'ECAI 2024 (27th European Conference on Artificial Intelligence), Frontiers in Artificial Intelligence and Applications, IOS Press'
    if u in AUTH_OV: r['authors'] = AUTH_OV[u]
    if u in TITLE_OV: r['title'] = TITLE_OV[u]
    if u == 'U133': r['status'] = 'Journal article (Crossref); peer-review process not stated in the PDF'
    if u == 'U017': r['status'] = 'Peer-reviewed conference proceedings (Crossref); Zenodo pre-publication version read'
    e = ext[u]
    for k in ('problem', 'rq', 'method', 'data', 'evaluation', 'findings', 'limitations', 'repro', 'qa', 'qa_notes', 'evidence', 'read_from'):
        r[k] = e.get(k)
    r['source_route'] = 'Citation searching' if u.startswith('C') else ('Database search (supplementary, RQ3)' if u.startswith('N') else 'Database search')
    rqp = os.path.join(S, 'final/rq_map.json')
    if os.path.exists(rqp):
        r['rq'] = json.load(open(rqp, encoding='utf8')).get(u, r['rq'])
    rows.append(r)

def surname(a):
    a = unicodedata.normalize('NFKD', a or '').encode('ascii', 'ignore').decode()
    return (a.split()[-1] if a.split() else 'zz').lower()
rows.sort(key=lambda r: (r['year'] or 0, surname((r['authors'] or ['zz'])[0]), r['title']))
# Study IDs from the first round (S01-S69) are kept; studies added by the supplementary search get the next numbers
old = {r['uid']: r['sid'] for r in json.load(open(os.path.join(S, 'final/master_v1.json'), encoding='utf8'))}
nxt = max(int(v[1:]) for v in old.values()) + 1
for r in rows:
    if r['uid'] in old: r['sid'] = old[r['uid']]
    else: r['sid'] = f'S{nxt:02d}'; nxt += 1
rows.sort(key=lambda r: int(r['sid'][1:]))
for r in rows:
    q = [x for x in r['qa'] if x != 'NA']
    r['qa_applicable'] = len(q); r['qa_raw'] = sum(q); r['qa_pct'] = round(100 * sum(q) / len(q), 1) if q else None
json.dump(rows, open(os.path.join(S, 'final/master_included.json'), 'w', encoding='utf8'), indent=1, ensure_ascii=False)
yrs = [r['year'] for r in rows]
print('n', len(rows), '| >=2022:', sum(y >= 2022 for y in yrs), '| >=2025:', sum(y >= 2025 for y in yrs), '| <2022:', sum(y < 2022 for y in yrs))
import collections
print(collections.Counter(r['kind'] for r in rows))
print('reviews:', [r['sid'] + ' ' + r['uid'] for r in rows if 'review' in (r['evidence'] or '').lower()])
for r in rows:
    print(r['sid'], r['uid'], r['year'], (r['authors'] or ['?'])[0], '|', r['title'][:55], '|', (r['venue'] or '')[:45], '|', r['kind'], '|', r['qa_pct'])
