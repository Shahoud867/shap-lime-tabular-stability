"""Shared loader: every deliverable (workbook, library, counts, report tables) is generated from these same files."""
import json, glob, os, re, collections

CODES = {
    'EC1': 'Explanation stability is not a primary aim or a separately reported outcome',
    'EC2': 'Non-tabular data only (images, text, signals, graphs, code)',
    'EC3': 'Explainer outside the SHAP/LIME family, or cross-method agreement rather than stability',
    'EC4': 'Not a full research report (abstract, poster, editorial) or not in English',
    'EC5': 'Off-topic homonym (stability in a physical, chemical or engineering sense)',
}
TA_MAP = {'E1': 'EC1', 'E2': 'EC2', 'E3': 'EC3', 'E4': 'EC4', 'E5': 'EC5'}
QA_ITEMS = [
    ('Q1', 'Aims or research questions are clearly stated'),
    ('Q2', 'Method (models, explainers, settings) is described well enough to replicate'),
    ('Q3', 'Datasets are described (source, size, features)'),
    ('Q4', 'An appropriate comparison is made (baseline, alternative explainer or model)'),
    ('Q5', 'The stability/robustness measure is clearly defined and appropriate'),
    ('Q6', 'Evidence is sufficient for the claims (scale, repetitions, statistical analysis)'),
    ('Q7', 'Limitations or threats to validity are discussed'),
    ('Q8', 'Code and/or data are available'),
    ('Q9', 'Ethical, privacy or regulatory issues are addressed where relevant (N/A for public non-personal data)'),
    ('Q10', 'Sources of randomness are controlled and reported (seeds, repetitions)'),
]

def load(S):
    J = lambda p: json.load(open(os.path.join(S, p), encoding='utf8'))
    d = {}
    d['ident'] = J('final/identification.json')
    d['dedup'] = J('final/dedup_exact.json')
    d['unique'] = J('final/unique.json')
    ta = {}
    for f in sorted(glob.glob(os.path.join(S, 'final/ta_batch*.txt'))):
        for line in open(f, encoding='utf8'):
            p = line.strip().split(' ', 2)
            if len(p) >= 2 and re.match(r'U\d{3}$', p[0]): ta[p[0]] = (p[1], p[2] if len(p) > 2 else '')
    d['ta'] = ta
    tac = {}
    for line in open(os.path.join(S, 'final/ta_chase.txt'), encoding='utf8'):
        p = line.strip().split(' ', 2)
        if len(p) >= 2: tac[p[0]] = (p[1], p[2] if len(p) > 2 else '')
    d['ta_chase'] = tac
    d['chase'] = J('final/chase_candidates.json')
    d['ftc'] = J('final/ft_candidates.json')
    d['ftc_chase'] = [c for c in J('final/ft_candidates_chase.json') if (c.get('cid') or c.get('uid')) != 'C066']
    ft = {}
    for f in sorted(glob.glob(os.path.join(S, 'extract/*.jsonl'))):
        for line in open(f, encoding='utf8'):
            if line.strip():
                r = json.loads(line); ft[r['uid']] = r
    d['ft'] = ft
    d['master'] = J('final/master_included.json')
    # supplementary search (realigned RQ3)
    d['s_ident'] = J('final/supp_identification.json')
    d['s_dedup'] = J('final/supp_dedup.json')
    d['s_unique'] = J('final/supp_unique.json')
    tas = {}
    for line in open(os.path.join(S, 'final/ta_supp.txt'), encoding='utf8'):
        p = line.strip().split(' ', 2)
        if len(p) >= 2: tas[p[0]] = (p[1], p[2] if len(p) > 2 else '')
    d['ta_supp'] = tas
    d['ftc_supp'] = J('final/ft_candidates_supp.json')
    return d

def counts(d):
    c = collections.OrderedDict()
    src = collections.Counter(r['source'] for r in d['ident'])
    c['idIEEE'] = src['IEEE Xplore']; c['idSD'] = src['ScienceDirect']; c['idArxiv'] = src['arXiv']
    c['idSS'] = src['Semantic Scholar']; c['idDBLP'] = src['DBLP']
    c['idTotal'] = len(d['ident']); c['nUnique'] = len(d['unique']); c['dupRemoved'] = c['idTotal'] - c['nUnique']
    ta = collections.Counter(v[0] for v in d['ta'].values())
    c['taScreened'] = len(d['ta']); c['taExcluded'] = sum(v for k, v in ta.items() if k.startswith('E'))
    for k in ['E1', 'E2', 'E3', 'E4', 'E5']: c['taEx' + k[1]] = ta.get(k, 0)
    c['taInclude'] = ta.get('I', 0); c['taUncertain'] = ta.get('U', 0); c['dbSought'] = c['taInclude'] + c['taUncertain']
    db_ids = [x['uid'] for x in d['ftc']]; ch_ids = [x.get('cid') or x.get('uid') for x in d['ftc_chase']]
    def fts(ids):
        z = collections.Counter();
        for u in ids:
            r = d['ft'][u]; z[r['decision']] += 1
            if r['decision'] == 'Exclude': z['EX_' + r['reason']] += 1
        return z
    a, b = fts(db_ids), fts(ch_ids)
    c['dbNotRetrieved'] = a['Not retrieved']; c['dbAssessed'] = len(db_ids) - a['Not retrieved']; c['dbFtExcluded'] = a['Exclude']
    c['dbFtExOne'] = a['EX_EC1']; c['dbFtExTwo'] = a['EX_EC2']; c['dbFtExThree'] = a['EX_EC3']; c['dbIncluded'] = a['Include']
    # citation searching (other methods)
    c['csHarvested'] = 723; c['csYear'] = 95; c['csAlready'] = 44; c['csDupWithin'] = 12; c['csTitleFilter'] = 502
    tac = collections.Counter(v[0] for v in d['ta_chase'].values())
    c['csCandidates'] = len(d['ta_chase']); c['csDupManual'] = tac.get('DUP', 0)
    c['csDupAll'] = c['csAlready'] + c['csDupWithin'] + c['csDupManual']
    c['csScreened'] = c['csCandidates'] - c['csDupManual']
    c['csSought'] = tac.get('I', 0) + tac.get('U', 0); c['csTaExcluded'] = c['csScreened'] - c['csSought']
    c['csNotRetrieved'] = b['Not retrieved']; c['csAssessed'] = len(ch_ids) - b['Not retrieved']; c['csFtExcluded'] = b['Exclude']
    c['csIncluded'] = b['Include']
    # supplementary database search
    ss = collections.Counter(r['source'] for r in d['s_ident'])
    c['spIEEE'] = ss['IEEE Xplore']; c['spArxiv'] = ss['arXiv']; c['spSS'] = ss['Semantic Scholar']; c['spDBLP'] = ss['DBLP']
    c['spTotal'] = len(d['s_ident'])
    c['spDupMain'] = sum(len(o['rec_ids']) for o in d['s_dedup']['overlap'])
    c['spScreened'] = len(d['s_unique']); c['spDupWithin'] = c['spTotal'] - c['spDupMain'] - c['spScreened']
    c['spDupRemoved'] = c['spDupWithin'] + c['spDupMain']
    tsp = collections.Counter(v[0] for v in d['ta_supp'].values())
    c['spExcluded'] = sum(v for k, v in tsp.items() if k.startswith('E'))
    for k in ['E1', 'E2', 'E3', 'E4', 'E5']: c['spEx' + k[1]] = tsp.get(k, 0)
    c['spSought'] = tsp.get('I', 0) + tsp.get('U', 0)
    sp_ids = [x['uid'] for x in d['ftc_supp']]; e = fts(sp_ids)
    c['spNotRetrieved'] = e['Not retrieved']; c['spAssessed'] = len(sp_ids) - e['Not retrieved']; c['spFtExcluded'] = e['Exclude']
    c['spFtExOne'] = e['EX_EC1']; c['spFtExThree'] = e['EX_EC3']; c['spFtExFour'] = e['EX_EC4']; c['spIncluded'] = e['Include']
    # databases combined (main + supplementary), as shown in the PRISMA diagram
    c['dbAllId'] = c['idTotal'] + c['spTotal']; c['dbAllDup'] = c['dupRemoved'] + c['spDupRemoved']
    c['dbAllScreened'] = c['taScreened'] + c['spScreened']; c['dbAllExcluded'] = c['taExcluded'] + c['spExcluded']
    for k in ['One', 'Two', 'Three', 'Four', 'Five']: c['dbAllEx' + k] = c['taEx' + str(['One', 'Two', 'Three', 'Four', 'Five'].index(k) + 1)] + c['spEx' + str(['One', 'Two', 'Three', 'Four', 'Five'].index(k) + 1)]
    c['dbAllSought'] = c['dbSought'] + c['spSought']; c['dbAllNotRetrieved'] = c['dbNotRetrieved'] + c['spNotRetrieved']
    c['dbAllAssessed'] = c['dbAssessed'] + c['spAssessed']; c['dbAllFtExcluded'] = c['dbFtExcluded'] + c['spFtExcluded']
    c['dbAllFtExOne'] = c['dbFtExOne'] + c['spFtExOne']; c['dbAllFtExTwo'] = c['dbFtExTwo']; c['dbAllFtExThree'] = c['dbFtExThree'] + c['spFtExThree']; c['dbAllFtExFour'] = c['spFtExFour']
    c['dbAllIncluded'] = c['dbIncluded'] + c['spIncluded']
    c['totalIncluded'] = c['dbAllIncluded'] + c['csIncluded']
    c['totalSought'] = c['dbAllSought'] + c['csSought']; c['totalNotRetrieved'] = c['dbAllNotRetrieved'] + c['csNotRetrieved']
    c['totalAssessed'] = c['dbAllAssessed'] + c['csAssessed']
    c['totalScreened'] = c['dbAllScreened'] + c['csScreened']
    yrs = [r['year'] for r in d['master']]
    c['incSinceTwentyTwo'] = sum(y >= 2022 for y in yrs); c['incSinceTwentyFive'] = sum(y >= 2025 for y in yrs)
    c['incBeforeTwentyTwo'] = sum(y < 2022 for y in yrs)
    c['pctSinceTwentyTwo'] = round(100 * c['incSinceTwentyTwo'] / len(yrs))
    kinds = collections.Counter(r['kind'] for r in d['master'])
    c['incJournal'] = kinds['journal']; c['incConference'] = kinds['conference']; c['incWorkshop'] = kinds['workshop']; c['incPreprint'] = kinds['preprint']
    # internal consistency checks
    assert c['csHarvested'] - c['csYear'] - c['csAlready'] - c['csDupWithin'] - c['csTitleFilter'] == c['csCandidates']
    assert c['taScreened'] == c['nUnique']
    assert c['taExcluded'] + c['dbSought'] == c['taScreened']
    assert c['dbSought'] == len(db_ids) and c['csSought'] == len(ch_ids)
    assert c['dbAssessed'] == c['dbFtExcluded'] + c['dbIncluded'] and c['csAssessed'] == c['csFtExcluded'] + c['csIncluded']
    assert c['spExcluded'] + c['spSought'] == c['spScreened'] and c['spSought'] == len(sp_ids)
    assert c['spAssessed'] == c['spFtExcluded'] + c['spIncluded'] and c['spDupWithin'] >= 0
    assert c['totalIncluded'] == len(d['master'])
    return c
