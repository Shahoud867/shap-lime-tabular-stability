"""Write the search log, refinement log and validation table (final/search_log.json) from the saved query files."""
import json, os, sys, re
S = sys.argv[1]
J = lambda p: json.load(open(os.path.join(S, p), encoding='utf8'))
V1 = J('final/search_log_v1.json')
Q1 = {r['db']: r['string'] for r in V1['log']}
sd = J('sd_urls.json')
dblp_head = open(os.path.join(S, 'final/dblp_raw.txt'), encoding='utf8').readline().strip()
dblp_q = dblp_head.split('query: ', 1)[1]
DATE = '2026-10-24'
log = [
    dict(db='IEEE Xplore (REST search, web interface)', string=Q1['IEEE Xplore (REST search, web interface)'], fields='All Metadata', filters='Publication year 2016-2026', results=48, kept=48,
         notes='Final string after piloting (see Refinement_Log). Trailing * used for truncation.'),
    dict(db='ScienceDirect (web search) SD1', string='Find articles with these terms: ' + sd['SD1']['qs'] + '  |  Title, abstract or author-specified keywords: ' + sd['SD1']['tak'], fields='qs: full record; tak: title-abstract-keywords', filters='Years 2016-2026', results=59, kept=59,
         notes='Braces force exact phrases. ScienceDirect allows at most 8 Boolean connectors per field, so the stability concept was split across SD1-SD3.'),
    dict(db='ScienceDirect (web search) SD2', string='Find articles with these terms: ' + sd['SD2']['qs'] + '  |  Title, abstract or author-specified keywords: ' + sd['SD2']['tak'], fields='qs: full record; tak: title-abstract-keywords', filters='Years 2016-2026', results=37, kept=37, notes='Second part of the stability concept.'),
    dict(db='ScienceDirect (web search) SD3', string='Find articles with these terms: ' + sd['SD3']['qs'] + '  |  Title, abstract or author-specified keywords: ' + sd['SD3']['tak'], fields='qs: full record; tak: title-abstract-keywords', filters='Years 2016-2026', results=0, kept=0, notes='Third part of the stability concept; no records.'),
    dict(db='arXiv (export API)', string=Q1['arXiv (export API)'], fields='abs (abstract)', filters='submittedDate 2016-01-01 to 2026-12-31', results=29, kept=29, notes='arXiv does not support truncation, so * was removed.'),
    dict(db='Semantic Scholar (Graph API, bulk search)', string=Q1['Semantic Scholar (Graph API, bulk search)'], fields='Title and abstract (API default)', filters='year 2016-2026', results=364, kept=364, notes='Multidisciplinary source used in place of OpenAlex after the OpenAlex daily API budget ran out during piloting. | = OR, + = AND.'),
    dict(db='DBLP (web search)', string=dblp_q, fields='Title (DBLP indexes titles only)', filters='Year 2016-2026 applied after export', results=174, kept=157, notes='$ = exact word, | = OR, space = AND. 17 records before 2016 removed.'),
    dict(db='Citation searching (Semantic Scholar citation graph)', string='Backward (references) and forward (citations) of 7 seed studies: Ballegeer 2025, Agarwal 2022, Visani 2022, Zhou 2021, Tzionis 2026, Rajhans 2026, Pala 2024', fields='Reference and citation lists', filters='Year 2016-2026; title must contain a stability term AND an explanation term; not already identified', results=723, kept=70, notes='Reported in the PRISMA "other methods" column.'),
    dict(db='ACM Digital Library', string='Prepared (see acm_q.txt in search_scripts) but not executed', fields='Title, Abstract, Keyword', filters='-', results='Not run', kept=0, notes='The site showed a Cloudflare bot check; it was not bypassed. ACM conference papers are partly covered through DBLP and Semantic Scholar.'),
    dict(db='SpringerLink', string='Prepared (see springer_q.txt) but not used', fields='-', filters='-', results='Not run', kept=0, notes='No title/abstract field restriction (searches full text), so results could not be made comparable.'),
]
for r in log: r['date'] = DATE
# supplementary search for the realigned RQ3 (mechanisms and pre-perturbation predictors)
dblp_s = open(os.path.join(S, 'final/supp_dblp_raw.txt'), encoding='utf8').readline().strip().split('query: ', 1)[1]
SQS = '(stability OR instability OR robustness OR fragility) AND (tabular OR "random forest" OR XGBoost OR "gradient boosting")'
log += [
    dict(db='IEEE Xplore - supplementary (RQ3)', string=J('final/supp_ieee.json')['query'], fields='All Metadata', filters='Publication year 2016-2026', results=J('final/supp_ieee.json')['total'], kept=len(J('final/supp_ieee.json')['records']),
         date='2026-10-25', notes='Explainer AND word-level stability AND mechanism/diagnostic AND tabular/tree concepts. First run 24 Oct (289); rerun 25 Oct with identical results after the working files were lost.'),
    dict(db='arXiv - supplementary (RQ3)', string=J('final/supp_arxiv.json')['query'], fields='abs (abstract)', filters='submittedDate 2016-2026', results=J('final/supp_arxiv.json')['total'], kept=len(J('final/supp_arxiv.json')['records']),
         date='2026-10-25', notes='Same concepts with truncated terms spelled out.'),
    dict(db='Semantic Scholar - supplementary (RQ3)', string=J('final/supp_s2.json')['query'], fields='Title and abstract (API default)', filters='year 2016-2026', results=J('final/supp_s2.json')['total'], kept=len(J('final/supp_s2.json')['data']),
         date='2026-10-25', notes='| = OR, + = AND.'),
    dict(db='DBLP - supplementary (RQ3)', string=dblp_s, fields='Title', filters='Year 2016-2026 applied after export', results=41, kept=35, date='2026-10-24',
         notes='Explainer names AND mechanism/diagnostic words in titles.'),
    dict(db='ScienceDirect - supplementary (attempted, not used)', string='Find articles with these terms: ' + SQS + '  |  Title, abstract or author-specified keywords: (SHAP OR LIME) AND ({split threshold} OR {decision boundary} OR {piecewise constant} OR {weak learner} OR {tree structure} OR {decision path})',
         fields='qs: full record; tak: title-abstract-keywords', filters='Years 2016-2026', results='464 (variants: 4,928; 343; 540; 645)', kept=0, date='2026-10-24',
         notes='Every variant returned mostly applied studies because phrase matching stayed loose; not used. Elsevier journals remain covered through Semantic Scholar.'),
]
for i, r in enumerate(log, 1): r['n'] = i
refine = [
    dict(src='OpenAlex', variant='SHAP/LIME terms AND stability phrases AND perturbation terms AND tabular terms (V1)', results=40, known='2', decision='Too narrow: the perturbation concept removed most benchmark papers.'),
    dict(src='OpenAlex', variant='SHAP/LIME terms AND single-word stability terms AND perturbation terms AND tabular terms (V2)', results=892, known='3', decision='Single words such as stable/robust add noise; recall still low.'),
    dict(src='OpenAlex', variant='SHAP/LIME terms AND single-word stability terms AND tabular terms, no perturbation concept (V5)', results=3589, known='5', decision='Too broad to screen: stability words often describe model performance, not explanations.'),
    dict(src='OpenAlex', variant='Stability words in title AND (SHAP/LIME terms AND tabular terms) in title/abstract (V6)', results=1177, known='6', decision='Better recall (the only variant that found Hwang et al. 2025) but still broad; the free OpenAlex API budget then ran out, so OpenAlex was used for piloting only.'),
    dict(src='Semantic Scholar', variant='SHAP/LIME terms + 19 stability phrases + perturbation terms + tabular terms (T2)', results=49, known='3', decision='Dropped the perturbation concept: it lost benchmark papers.'),
    dict(src='Semantic Scholar', variant='SHAP/LIME terms + 19 stability phrases + tabular terms (T1)', results=364, known='6', decision='Adopted. The phrase-level stability concept was then used in every database.'),
    dict(src='ScienceDirect', variant='Stability phrases in double quotes', results='257-344', known='-', decision='ScienceDirect matched quoted phrases loosely; switched to braces for exact phrases and split into SD1-SD3 because of the 8-connector limit.'),
    dict(src='DBLP', variant='Prefix matching on shap/lime', results='-', known='-', decision='Prefix matching added noise (e.g., "shap" matched "shape"); switched to exact words with $.'),
]
refine += [
    dict(src='Semantic Scholar (suppl.)', variant='Explainer + word-level stability + 20 mechanism/diagnostic terms + tabular/tree terms (v1)', results=388, known='4 of 8 (RQ3 set)', decision='Missed papers framed by smoothness/Lipschitz, weak learners or credible intervals; added those terms.'),
    dict(src='Semantic Scholar (suppl.)', variant='v1 + weak learner, smoothness, Lipschitz, credible/uncertainty intervals, sampling variance/noise/randomness (v2)', results=494, known='5 of 8', decision='Adopted for IEEE, arXiv and Semantic Scholar. The 3 papers still missed have no tabular term in their abstracts and were already included through citation searching.'),
    dict(src='Semantic Scholar (suppl.)', variant='v2 without the tabular/tree concept', results=1089, known='-', decision='Not used: doubles the set with mostly non-tabular work; kept the tabular concept for consistency with the main search.'),
    dict(src='ScienceDirect (suppl.)', variant='Five variants (explainer in full text, title or title-abstract field; phrase or word-level mechanism terms)', results='343-4,928', known='-', decision='Dropped: braces did not enforce exact phrases, so results were dominated by applied studies.'),
    dict(src='DBLP (suppl.)', variant='Generic explanation/attribution words AND mechanism words in titles', results=247, known='-', decision='Mostly uncertainty-explanation and decision-theory titles; narrowed to SHAP/LIME/Shapley in the title (41 records).'),
]
for i, r in enumerate(refine, 1): r['step'] = i
src = open(os.path.join(S, 'validate.py'), encoding='utf8').read()
KNOWN = eval(src[src.index('{'): src.index('}') + 1])
# the four remaining stability papers cited in the Assignment 1 proposal (refs 13, 14, 19, 21), added after the final searches
KNOWN.update({
 "Jin2025": "probabilistic stability guarantees for feature attributions",
 "Decker2025": "improving perturbation based explanations by understanding the role of uncertainty calibration",
 "Takefuji2025": "the stability paradox why high prediction accuracy does not guarantee reliable feature importance",
 "Hasnain2026": "cross method explanation stability under prediction preserving perturbations",
})
norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())
ident = J('final/identification.json'); chase = J('final/chase_candidates.json')
cols = ['IEEE Xplore', 'ScienceDirect', 'arXiv', 'Semantic Scholar', 'DBLP']
val = []
for k, t in KNOWN.items():
    tt = norm(t)
    hit = lambda x: re.search(r'(^| )' + re.escape(tt) + r'( |$)', norm(x)) is not None
    row = [k, t] + ['Yes' if any(hit(r['title']) for r in ident if r['source'] == c) else 'No' for c in cols]
    row.append('Yes' if 'Yes' in row[2:] else 'No')
    row.append('Yes' if any(hit(c['title']) for c in chase) else 'No')
    val.append(row)
import supp_validate as sv
sfiles = [('IEEE Xplore', 'final/supp_ieee.json'), ('arXiv', 'final/supp_arxiv.json'), ('Semantic Scholar', 'final/supp_s2.json'), ('DBLP', 'final/supp_dblp_raw.txt')]
sh = {c: sv.hits(sv.items(os.path.join(S, p))) for c, p in sfiles}
chit = {k: any(re.search(r'(^| )' + re.escape(t) + r'( |$)', norm(c['title'])) for c in chase) for k, (t, ax) in sv.KNOWN_SUPP.items()}
inc = {r['uid'] for r in J('final/master_included.json')}
supp_val = [[k, t] + ['Yes' if sh[c][k] else 'No' for c, _ in sfiles] + ['Yes' if any(sh[c][k] for c, _ in sfiles) else 'No'] for k, (t, ax) in sv.KNOWN_SUPP.items()]
json.dump({'log': log, 'refine': refine, 'val_cols': cols, 'validation': val, 'supp_val_cols': [c for c, _ in sfiles], 'supp_validation': supp_val}, open(os.path.join(S, 'final/search_log.json'), 'w', encoding='utf8'), indent=1, ensure_ascii=False)
print(sum(r[-2] == 'Yes' for r in val), 'found by databases;', sum(r[-2] == 'No' and r[-1] == 'Yes' for r in val), 'added by citation searching')
