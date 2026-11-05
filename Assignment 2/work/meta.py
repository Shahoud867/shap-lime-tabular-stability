"""Collect and verify bibliographic metadata for included studies (Crossref for DOIs, arXiv API for preprints)."""
import json, glob, os, sys, re, time, requests
import xml.etree.ElementTree as ET
S = sys.argv[1]
fc = {c['uid']: c for c in json.load(open(os.path.join(S, 'final/ft_candidates.json'), encoding='utf8'))}
for c in json.load(open(os.path.join(S, 'final/ft_candidates_chase.json'), encoding='utf8')):
    fc[c.get('cid') or c.get('uid')] = c
if os.path.exists(os.path.join(S, 'final/ft_candidates_supp.json')):
    for c in json.load(open(os.path.join(S, 'final/ft_candidates_supp.json'), encoding='utf8')): fc[c['uid']] = c
d = {}
for f in sorted(glob.glob(os.path.join(S, 'extract/ft_00*.jsonl'))):
    for line in open(f, encoding='utf8'):
        if line.strip():
            r = json.loads(line); d[r['uid']] = r
inc = [u for u, r in d.items() if r['decision'] == 'Include']
out_p = os.path.join(S, 'final', 'meta_included.json')
out = json.load(open(out_p, encoding='utf8')) if os.path.exists(out_p) else {}
ns = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
def crossref(doi):
    r = requests.get('https://api.crossref.org/works/' + requests.utils.quote(doi), timeout=60, headers={'User-Agent': 'MBR-A2-review/1.0'})
    if r.status_code != 200: return {'error': r.status_code}
    m = r.json()['message']
    return {'type': m.get('type'), 'title': (m.get('title') or [''])[0], 'container': (m.get('container-title') or [''])[0],
            'event': (m.get('event') or {}).get('name') if isinstance(m.get('event'), dict) else None,
            'volume': m.get('volume'), 'issue': m.get('issue'), 'page': m.get('page') or m.get('article-number'),
            'year': ((m.get('published-print') or m.get('published-online') or m.get('issued') or {}).get('date-parts') or [[None]])[0][0],
            'authors': [((a.get('given') or '') + ' ' + (a.get('family') or '')).strip() or a.get('name', '') for a in m.get('author', [])],
            'publisher': m.get('publisher')}
def arxiv(aid):
    r = requests.get('http://export.arxiv.org/api/query', params={'id_list': aid}, timeout=60)
    e = ET.fromstring(r.text).find('a:entry', ns)
    if e is None: return {'error': 'none'}
    g = lambda p: (e.find(p, ns).text if e.find(p, ns) is not None else None)
    return {'title': ' '.join(g('a:title').split()), 'authors': [a.find('a:name', ns).text for a in e.findall('a:author', ns)],
            'published': g('a:published'), 'journal_ref': g('arxiv:journal_ref'), 'doi': g('arxiv:doi'), 'comment': g('arxiv:comment')}
for u in inc:
    if u in out and not out[u].get('retry'): continue
    c = fc.get(u, {}); r = d[u]
    rec = {'uid': u, 'cand_title': c.get('title'), 'cand_year': c.get('year'), 'cand_venue': c.get('venue') or c.get('journal'), 'cand_doi': c.get('doi'), 'cand_arxiv': c.get('arxiv'), 'read_from': r.get('read_from')}
    dois = []
    if c.get('doi') and not c['doi'].lower().startswith('10.48550'): dois.append(c['doi'])
    for m in re.findall(r'10\.\d{4,9}/[^\s;,)]+', r.get('read_from') or ''):
        m = m.rstrip('.')
        if m.lower() not in [x.lower() for x in dois] and not m.lower().startswith('10.48550'): dois.append(m)
    rec['dois'] = dois
    rec['crossref'] = {}
    for x in dois:
        try: rec['crossref'][x] = crossref(x)
        except Exception as e: rec['crossref'][x] = {'error': str(e)[:80]}
        time.sleep(0.4)
    aid = c.get('arxiv')
    if not aid:
        m = re.search(r'arXiv:(\d{4}\.\d{4,5})', r.get('read_from') or '')
        aid = m.group(1) if m else None
    if aid:
        try: rec['arxiv'] = arxiv(aid); rec['arxiv_id'] = aid
        except Exception as e: rec['arxiv'] = {'error': str(e)[:80]}
        time.sleep(3.1)
    out[u] = rec
    print(u, dois, aid, flush=True)
json.dump(out, open(out_p, 'w', encoding='utf8'), indent=1, ensure_ascii=False)
print('done', len(out))
