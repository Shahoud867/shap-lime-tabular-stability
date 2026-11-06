"""Evidence-map coding of the included studies (from the extraction records) and the counts used in the synthesis and gap."""
import json, os, sys, collections
# src: N = explainer run-to-run sampling noise; R = retraining/resampling/seeds/folds; P = benign input perturbation;
#      A = adversarial manipulation or attack; D = data condition varied (imbalance, correlation, background data, shift, compression); T = theory/bounds
# ex: S = SHAP family, L = LIME family; pp: instance-level prediction-preserving filter; pe: prediction-side quantity related to explanation change
# m3: linear model + random forest + boosted trees all evaluated; st: inferential statistics applied to stability outcomes; rev: review
C = {
 'C069': dict(src='P', ex='SL', pp=0, pe=0, m3=0, st=0), 'C068': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=0),
 'C061': dict(src='A', ex='L', pp=0, pe=0, m3=0, st=0), 'C062': dict(src='A', ex='SL', pp=0, pe=0, m3=0, st=0),
 'C058': dict(src='NP', ex='SL', pp=0, pe=0, m3=0, st=1), 'U373': dict(src='PR', ex='S', pp=0, pe=0, m3=0, st=0),
 'U577': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=0), 'U483': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=0),
 'U498': dict(src='PT', ex='SL', pp=1, pe=1, m3=0, st=0), 'U155': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=1),
 'C040': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=1), 'U493': dict(src='NP', ex='L', pp=0, pe=0, m3=0, st=0),
 'U017': dict(src='PD', ex='S', pp=0, pe=0, m3=0, st=1), 'U481': dict(src='ND', ex='S', pp=0, pe=0, m3=0, st=0),
 'U258': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=0), 'U466': dict(src='A', ex='SL', pp=0, pe=0, m3=0, st=0, rev=1),
 'U362': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=0), 'U556': dict(src='N', ex='S', pp=0, pe=0, m3=0, st=0),
 'C049': dict(src='PT', ex='SL', pp=0, pe=0, m3=0, st=0), 'U490': dict(src='RD', ex='S', pp=0, pe=0, m3=0, st=0),
 'U546': dict(src='N', ex='S', pp=0, pe=0, m3=0, st=0), 'C033': dict(src='R', ex='SL', pp=0, pe=0, m3=0, st=0),
 'C048': dict(src='PT', ex='S', pp=0, pe=1, m3=0, st=0), 'C045': dict(src='NR', ex='L', pp=0, pe=0, m3=0, st=0),
 'U159': dict(src='P', ex='S', pp=1, pe=0, m3=0, st=0), 'U560': dict(src='P', ex='S', pp=0, pe=0, m3=0, st=0),
 'U557': dict(src='PT', ex='SL', pp=0, pe=1, m3=0, st=0), 'U176': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=0),
 'U291': dict(src='RD', ex='SL', pp=0, pe=0, m3=1, st=0), 'U487': dict(src='N', ex='S', pp=0, pe=0, m3=0, st=0),
 'U491': dict(src='N', ex='SL', pp=0, pe=0, m3=0, st=1), 'U311': dict(src='P', ex='SL', pp=0, pe=0, m3=0, st=0),
 'C024': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=0, rev=1), 'U126': dict(src='R', ex='S', pp=0, pe=0, m3=0, st=1),
 'U042': dict(src='R', ex='SL', pp=0, pe=0, m3=0, st=0), 'U124': dict(src='RD', ex='SL', pp=0, pe=0, m3=0, st=1),
 'U340': dict(src='P', ex='S', pp=0, pe=0, m3=0, st=0), 'U148': dict(src='R', ex='S', pp=0, pe=0, m3=0, st=0),
 'U502': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=0), 'U183': dict(src='P', ex='S', pp=0, pe=0, m3=0, st=0),
 'U356': dict(src='RD', ex='S', pp=0, pe=0, m3=0, st=1), 'U123': dict(src='RD', ex='S', pp=0, pe=1, m3=0, st=0),
 'U521': dict(src='RDT', ex='S', pp=0, pe=0, m3=0, st=1), 'U327': dict(src='PT', ex='S', pp=0, pe=0, m3=0, st=0),
 'U528': dict(src='N', ex='SL', pp=0, pe=0, m3=0, st=0), 'U524': dict(src='N', ex='SL', pp=0, pe=0, m3=0, st=0),
 'U409': dict(src='NR', ex='S', pp=0, pe=0, m3=0, st=0), 'U288': dict(src='RP', ex='S', pp=0, pe=0, m3=0, st=1),
 'U510': dict(src='R', ex='S', pp=0, pe=0, m3=0, st=0), 'U221': dict(src='D', ex='S', pp=0, pe=1, m3=1, st=0),
 'U236': dict(src='R', ex='S', pp=0, pe=1, m3=1, st=0), 'U195': dict(src='PA', ex='SL', pp=0, pe=1, m3=0, st=1),
 'U226': dict(src='RD', ex='S', pp=0, pe=0, m3=0, st=0), 'U499': dict(src='NR', ex='L', pp=0, pe=0, m3=1, st=0),
 'U224': dict(src='RD', ex='L', pp=0, pe=0, m3=0, st=0), 'U326': dict(src='R', ex='S', pp=0, pe=0, m3=0, st=0),
 'U045': dict(src='PD', ex='S', pp=0, pe=0, m3=0, st=1), 'U355': dict(src='R', ex='S', pp=0, pe=0, m3=0, st=1),
 'U163': dict(src='A', ex='S', pp=0, pe=1, m3=0, st=0), 'C013': dict(src='A', ex='S', pp=0, pe=1, m3=0, st=0),
 'U233': dict(src='R', ex='S', pp=0, pe=0, m3=1, st=0), 'U336': dict(src='PA', ex='S', pp=1, pe=0, m3=1, st=0),
 'U248': dict(src='RD', ex='S', pp=0, pe=0, m3=0, st=0), 'U141': dict(src='PN', ex='SL', pp=0, pe=0, m3=0, st=0),
 'U105': dict(src='PD', ex='SL', pp=0, pe=0, m3=0, st=1), 'U077': dict(src='RDT', ex='SL', pp=0, pe=0, m3=0, st=0),
 'U210': dict(src='R', ex='SL', pp=0, pe=0, m3=0, st=0), 'U519': dict(src='R', ex='S', pp=0, pe=0, m3=0, st=0),
 'U133': dict(src='R', ex='S', pp=0, pe=0, m3=1, st=0),
 # supplementary search (realigned RQ3)
 'N101': dict(src='RA', ex='SL', pp=0, pe=0, m3=0, st=0), 'N339': dict(src='R', ex='S', pp=0, pe=0, m3=0, st=0),
 'N029': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=0), 'N708': dict(src='DT', ex='S', pp=0, pe=0, m3=0, st=0),
 'N693': dict(src='DT', ex='S', pp=0, pe=0, m3=0, st=0), 'N300': dict(src='RT', ex='SL', pp=0, pe=1, m3=0, st=1),
 'N557': dict(src='N', ex='L', pp=0, pe=0, m3=0, st=0), 'N616': dict(src='PT', ex='SL', pp=0, pe=1, m3=0, st=1),
 'N480': dict(src='P', ex='SL', pp=0, pe=0, m3=0, st=0), 'N383': dict(src='PRD', ex='S', pp=0, pe=1, m3=1, st=0),
}
# Added after full-text eligibility assessment, 7 November 2026.
C.update({
 'N240':dict(src='RA',ex='S',pp=0,pe=0,m3=0,st=0),
 'N421':dict(src='R',ex='S',pp=0,pe=0,m3=0,st=0),
 'N594':dict(src='RD',ex='S',pp=0,pe=1,m3=0,st=0),
 'N687':dict(src='R',ex='S',pp=0,pe=1,m3=0,st=0),
 'U302':dict(src='R',ex='S',pp=0,pe=1,m3=0,st=1),
})

# Mechanism and diagnostic evidence (realigned RQ3 = refined RQ3 mechanism + refined RQ4 predictor)
MECH = {
 'thr': ['C068', 'C049', 'C048', 'U159', 'U163', 'N300', 'N616'],   # split thresholds, segment/decision boundaries, discretisation, piecewise-constant surfaces
 'coll': ['U521', 'U077', 'U226', 'N300'],                            # correlated features / interactions
 'samp': ['C068', 'U155', 'U483', 'U577', 'U546', 'U556', 'U491', 'U524', 'U528', 'U362', 'U258', 'U176', 'U141', 'U487', 'C058', 'U502', 'N029', 'N557', 'U499'],
 'dist': ['U481', 'U490', 'N708'],                                    # background / reference distribution
 'smooth': ['C048', 'U557', 'N616'],                                  # smoothness (Lipschitz) of the predictor
 'adv': ['C062', 'C061', 'U466'],                                     # off-manifold samples exploited by attacks
}
DIAG = ['C058', 'C040', 'C045', 'C049', 'U409', 'U195', 'U042', 'U519', 'N339', 'N616', 'N693', 'N708', 'N300', 'U288', 'U521', 'N421']   # uncertainty or reliability estimates for explanations
DIAG_FREE = ['U195', 'N616', 'U521']   # predictor available without re-running perturbations, validated against observed instability
MECH['target'] = ['N240', 'N594']
MECH['regularization'] = ['N687']

if __name__ == '__main__':
    S = sys.argv[1]
    M = json.load(open(os.path.join(S, 'final/master_included.json'), encoding='utf8'))
    assert set(C) == {r['uid'] for r in M}, set(C) ^ {r['uid'] for r in M}
    sid = {r['uid']: r['sid'] for r in M}
    n = len(M)
    out = collections.OrderedDict()
    for k, lab in [('N', 'noise'), ('R', 'retrain'), ('P', 'perturb'), ('A', 'adversarial'), ('D', 'datacond'), ('T', 'theory')]:
        out['src_' + lab] = [sid[u] for u, c in C.items() if k in c['src']]
    out['shap_only'] = [sid[u] for u, c in C.items() if c['ex'] == 'S']; out['lime_only'] = [sid[u] for u, c in C.items() if c['ex'] == 'L']
    out['both'] = [sid[u] for u, c in C.items() if c['ex'] == 'SL']
    for k in ('pp', 'pe', 'm3', 'st'): out[k] = [sid[u] for u, c in C.items() if c[k]]
    out['review'] = [sid[u] for u, c in C.items() if c.get('rev')]
    out['pp_and_both'] = [sid[u] for u, c in C.items() if c['pp'] and c['ex'] == 'SL']
    out['pp_and_m3'] = [sid[u] for u, c in C.items() if c['pp'] and c['m3']]
    out['pp_both_m3'] = [sid[u] for u, c in C.items() if c['pp'] and c['m3'] and c['ex'] == 'SL']
    out['noise_and_perturb'] = [sid[u] for u, c in C.items() if 'N' in c['src'] and 'P' in c['src']]
    for k, v in MECH.items(): out['mech_' + k] = [sid[u] for u in v]
    out['mech_any'] = sorted({sid[u] for v in MECH.values() for u in v})
    out['diag'] = [sid[u] for u in DIAG]; out['diag_free'] = [sid[u] for u in DIAG_FREE]
    # realigned RQ mapping: RQ1/RQ2 kept from extraction; RQ3 = mechanism or predictor evidence
    rqmap = {}
    mech_u = {u for v in MECH.values() for u in v} | set(DIAG)
    for r in M:
        q = [x for x in ('RQ1', 'RQ2') if x in (r['rq'] or '')]
        if r['uid'] in mech_u: q.append('RQ3')
        if not q: q = ['RQ1']
        rqmap[r['uid']] = ', '.join(q)
    json.dump(rqmap, open(os.path.join(S, 'final/rq_map.json'), 'w', encoding='utf8'), indent=1)
    rq = collections.Counter()
    for u, v in rqmap.items():
        for q in ('RQ1', 'RQ2', 'RQ3'):
            if q in v: rq[q] += 1
    res = {k: sorted(v) for k, v in out.items()}
    res['counts'] = {k: len(v) for k, v in out.items()}; res['n'] = n; res['rq'] = dict(rq)
    json.dump({'codes': {sid[u]: c for u, c in C.items()}, **res}, open(os.path.join(S, 'final/coding.json'), 'w', encoding='utf8'), indent=1)
    print(json.dumps(res['counts'])); print(res['rq'])
    for k in ('pp', 'pe', 'm3', 'pp_and_both', 'pp_and_m3', 'noise_and_perturb', 'review', 'mech_thr', 'diag', 'diag_free'): print(k, res[k])
