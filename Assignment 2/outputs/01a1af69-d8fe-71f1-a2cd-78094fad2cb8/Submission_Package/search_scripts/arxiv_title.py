"""Look up arXiv versions of candidate papers by title (metadata only)."""
import json, sys, os, re, time, requests
import xml.etree.ElementTree as ET
base = sys.argv[1]; ids = sys.argv[2].split(',')
rows = json.load(open(os.path.join(base, 'final/ft_candidates.json'), encoding='utf8')) + json.load(open(os.path.join(base, 'final/ft_candidates_chase.json'), encoding='utf8'))
by = {(r.get('uid') or r.get('cid')): r for r in rows}
ns = {'a': 'http://www.w3.org/2005/Atom'}
norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
for k in ids:
    t = by[k]['title']
    words = [w for w in re.findall(r'[A-Za-z0-9]+', t) if len(w) > 3][:8]
    q = ' AND '.join('ti:' + w for w in words)
    r = requests.get('http://export.arxiv.org/api/query', params={'search_query': q, 'max_results': 5}, timeout=60)
    hits = []
    for e in ET.fromstring(r.text).findall('a:entry', ns):
        et = e.find('a:title', ns).text.strip().replace('\n', ' ')
        hits.append((e.find('a:id', ns).text.split('/abs/')[-1], et[:80], norm(et) == norm(t)))
    print(k, '|', t[:70], '|', [h for h in hits if h[2]] or ('near:', hits[:2]))
    time.sleep(3.2)
