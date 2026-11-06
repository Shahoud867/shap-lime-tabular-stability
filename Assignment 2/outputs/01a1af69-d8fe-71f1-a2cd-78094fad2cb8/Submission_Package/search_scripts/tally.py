import json, glob, os, sys, collections
S = sys.argv[1]
fc = json.load(open(os.path.join(S, 'final/ft_candidates.json'), encoding='utf8'))
fcc = json.load(open(os.path.join(S, 'final/ft_candidates_chase.json'), encoding='utf8'))
cid = lambda c: c.get('cid') or c.get('uid')
fcc = [c for c in fcc if cid(c) != 'C066']
sought_db = [c['uid'] for c in fc]; sought_ch = [cid(c) for c in fcc]
d = {}
for f in sorted(glob.glob(os.path.join(S, 'extract/*.jsonl'))):
    for line in open(f, encoding='utf8'):
        if line.strip():
            r = json.loads(line)
            if r['uid'] in d: print('DUPLICATE', r['uid'], f)
            d[r['uid']] = r
print('sought db', len(sought_db), 'chase', len(sought_ch))
allc = sought_db + sought_ch
print('missing', [u for u in allc if u not in d], 'extra', [u for u in d if u not in allc])
for src, ids in [('db', sought_db), ('chase', sought_ch), ('all', allc)]:
    cc = collections.Counter((d[u]['decision'], d[u].get('reason', '') if d[u]['decision'] == 'Exclude' else '') for u in ids if u in d)
    print(src, sorted(cc.items()))
