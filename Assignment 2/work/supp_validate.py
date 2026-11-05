"""Check which mechanism/diagnostic benchmark papers each supplementary result set retrieves (by title or arXiv id)."""
import json, re, sys, os
KNOWN_SUPP = {
    'Pala2024': ('combination of weak learners explanations', '2402.19025'),
    'Zhang2019': ('understanding uncertainty in lime', '1904.12991'),
    'Khan2024': ('analyzing explainer robustness via probabilistic lipschitzness', '2206.12481'),
    'Hill2024': ('boundary aware uncertainty for feature attribution explainers', '2210.02419'),
    'Mikriukov2026': ('uncertainty gating for cost aware explainable', '2603.29915'),
    'Slack2021': ('reliable post hoc explanations modeling uncertainty', '2008.05030'),
    'Rajhans2026': ('beyond gradient based attacks', '2607.01679'),
    'Dubey2026': ('ubiqtree uncertainty quantification in xai with tree ensembles', '2508.09639'),
}
norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())
def items(path):
    if path.endswith('.txt'):
        out = []
        for l in open(path, encoding='utf8'):
            if l.strip() and not l.startswith('#'):
                y, t, d, v, ti = l.rstrip('\n').split('~', 4); out.append((ti, d))
        return out
    d = json.load(open(path, encoding='utf8'))
    if 'records' in d:
        return [(re.sub(r'<[^>]+>', '', r.get('articleTitle') or r.get('title') or ''), (r.get('arxiv_id') or r.get('doi') or '')) for r in d['records']]
    return [(r.get('title') or '', ((r.get('externalIds') or {}).get('ArXiv') or '')) for r in d.get('data', [])]
def hits(its):
    res = {}
    for k, (t, ax) in KNOWN_SUPP.items():
        res[k] = any(re.search(r'(^| )' + re.escape(t) + r'( |$)', norm(ti)) or (ax and ax in (idv or '').lower()) for ti, idv in its)
    return res
if __name__ == '__main__':
    for p in sys.argv[1:]:
        h = hits(items(p)); print(os.path.basename(p), sum(h.values()), [k for k, v in h.items() if v])
