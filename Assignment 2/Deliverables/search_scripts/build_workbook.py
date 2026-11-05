"""Build the screening workbook (xlsx) from the review data files."""
import json, os, sys, re, random, collections
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
import data_model as dm

S, OUT = sys.argv[1], sys.argv[2]
d = dm.load(S); C = dm.counts(d)
HDR = PatternFill('solid', fgColor='18274D'); HF = Font(bold=True, color='FFFFFF'); WR = Alignment(wrap_text=True, vertical='top')
thin = Side(style='thin', color='C8CDD8'); BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
wb = Workbook()

def sheet(name, headers, rows, widths=None, first=False):
    ws = wb.active if first else wb.create_sheet()
    ws.title = name
    ws.append(headers)
    for c in ws[1]: c.fill = HDR; c.font = HF; c.alignment = Alignment(wrap_text=True, vertical='center')
    for r in rows: ws.append(r)
    for i, w in enumerate(widths or [18] * len(headers), 1): ws.column_dimensions[get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row: c.alignment = WR
    ws.freeze_panes = 'A2'; ws.auto_filter.ref = ws.dimensions
    return ws

# README
ws = wb.active; ws.title = 'README'
readme = [
    ['MG 3010 Assignment 2 - Literature search, screening and evidence review: screening workbook'],
    ['Group: Shahoud Shahid (23i-2515) and Saif Shahzad (23i-2634), FAST-NUCES Islamabad'],
    ['Topic: stability of SHAP and LIME explanations for tabular machine-learning models'],
    ['Main searches were run on 24 October 2026; the supplementary search for the realigned RQ3 (mechanisms and pre-perturbation predictors) on 24-25 October 2026. Every count in this workbook is generated from the same data files used for the report and PRISMA diagram.'],
    [''],
    ['Sheets'],
    ['Search_Log', 'Executed searches: date, database, exact string, fields, filters, results, records kept, changes'],
    ['Refinement_Log', 'Pilot strings that were tested and why they were changed or rejected'],
    ['Validation', 'Known benchmark papers and whether each source retrieved them'],
    ['Identification', 'All records exported: main search (R...) and supplementary RQ3 search (Q...)'],
    ['Dedup', 'Duplicate groups and the matching rule that linked them (main and supplementary)'],
    ['TA_Screening', 'Title/abstract decision and reason for every screened record (U... main, N... supplementary, C... citation search)'],
    ['Dual_Screening', 'Random 15% subset for independent dual screening, with Cohen\'s kappa formulas (to be completed by the team)'],
    ['FullText', 'Every report sought for retrieval: retrieval status, decision, criterion and page/section evidence'],
    ['Included', 'The included studies with verified citation details and peer-review status'],
    ['Extraction', 'Data extraction table for the included studies'],
    ['Quality', 'Quality appraisal (Yes = 1, Partial = 0.5, No = 0, N/A) with notes'],
    ['PRISMA_Counts', 'Counts for every PRISMA 2020 box, with arithmetic checks'],
    [''],
    ['Decision codes'],
    ['I / U', 'Include / Uncertain at title-abstract stage: both were sought for full text'],
] + [[k, v] for k, v in dm.CODES.items()] + [
    ['DUP', 'Duplicate of a record already identified (citation searching)'],
    ['Not retrieved', 'Full text could not be obtained (paywall, no open copy, or publisher bot check that was not bypassed)'],
    [''],
    ['Integrity note', 'First-pass screening, full-text assessment and extraction were prepared with AI assistance (Claude) from the records and full texts listed here. The team must check these decisions; the dual-screening sheet is left blank for the two members to complete, and no agreement statistic is reported until they do.'],
]
for r in readme: ws.append(r)
ws.column_dimensions['A'].width = 22; ws.column_dimensions['B'].width = 120
ws['A1'].font = Font(bold=True, size=13)
for r in (6, 20): ws.cell(r, 1).font = Font(bold=True)

# Search log
SEARCH = json.load(open(os.path.join(S, 'final/search_log.json'), encoding='utf8'))
sheet('Search_Log', ['#', 'Date', 'Database / interface', 'Exact search string', 'Fields searched', 'Filters', 'Results', 'Records exported', 'Changes / notes'],
      [[s['n'], s['date'], s['db'], s['string'], s['fields'], s['filters'], s['results'], s['kept'], s['notes']] for s in SEARCH['log']],
      [4, 11, 22, 90, 18, 22, 9, 10, 50])
sheet('Refinement_Log', ['Step', 'Source', 'String variant', 'Results', 'Known papers found (of 12)', 'Decision and reason'],
      [[s['step'], s['src'], s['variant'], s['results'], s['known'], s['decision']] for s in SEARCH['refine']], [5, 16, 70, 9, 12, 70])
ws = sheet('Validation', ['Known paper', 'Title fragment'] + SEARCH['val_cols'] + ['Any database', 'Citation searching'],
      SEARCH['validation'], [18, 50] + [11] * (len(SEARCH['val_cols']) + 2))
ws.append([]); ws.append(['Supplementary search (RQ3) benchmark set', ''] + SEARCH['supp_val_cols'] + ['Any source'])
for c in ws[ws.max_row]: c.font = Font(bold=True)
for r in SEARCH['supp_validation']: ws.append(r)

# Identification
sheet('Identification', ['Record ID', 'Source', 'Sub-query', 'Title', 'Year', 'Venue', 'DOI', 'arXiv'],
      [[r['rec_id'], r['source'], r.get('sub') or '', r.get('title'), r.get('year'), r.get('venue'), r.get('doi'), r.get('arxiv')] for r in d['ident']]
      + [[r['rec_id'], r['source'], 'supplementary (RQ3)', r.get('title'), r.get('year'), r.get('venue'), r.get('doi'), r.get('arxiv')] for r in d['s_ident']],
      [10, 16, 9, 70, 6, 30, 28, 12])

# Dedup
par = d['dedup']['parent']; why = {}
for a, b, w in d['dedup']['links']: why[b] = {'doi': 'DOI', 'arx': 'arXiv ID', 's2': 'Semantic Scholar ID', 't': 'normalised title'}.get(w, w)
why['R0227'] = 'manual check (same arXiv paper split into two Semantic Scholar entries)'
byrec = {r['rec_id']: r for r in d['ident']}
rows = []
for u in d['unique']:
    if len(u['rec_ids']) < 2: continue
    for i, rid in enumerate(u['rec_ids']):
        r = byrec[rid]
        rows.append([u['uid'], rid, r['source'], r.get('title'), 'Retained (merged record)' if i == 0 else 'Removed as duplicate', '' if i == 0 else why.get(rid, 'manual check (same paper split into two Semantic Scholar entries)')])
basis = collections.Counter(x[5] for x in rows if x[4].startswith('Removed'))
sby = {r['rec_id']: r for r in d['s_ident']}; swhy = d['s_dedup']['why']
WHY = {'doi': 'DOI', 'arx': 'arXiv ID', 's2': 'Semantic Scholar ID', 't': 'normalised title'}
for u in d['s_unique']:
    if len(u['rec_ids']) < 2: continue
    for i, rid in enumerate(u['rec_ids']):
        r = sby[rid]; rows.append([u['uid'], rid, r['source'] + ' (suppl.)', r.get('title'), 'Retained (merged record)' if i == 0 else 'Removed as duplicate', '' if i == 0 else WHY.get(swhy.get(rid), 'linked within group')])
for o in d['s_dedup']['overlap']:
    for rid in o['rec_ids']:
        r = sby[rid]; rows.append([o['matches'], rid, r['source'] + ' (suppl.)', r.get('title'), 'Removed: already identified by main search', 'matches ' + o['matches']])
ws = sheet('Dedup', ['Unique ID', 'Record ID', 'Source', 'Title', 'Status', 'Matched on'], rows, [9, 10, 22, 70, 30, 30])
ws.append([]); ws.append(['Summary', f"Main search: identified {C['idTotal']}; duplicates removed {C['dupRemoved']}; unique records {C['nUnique']}"])
for k, v in basis.items(): ws.append(['', f'Removed via {k}: {v}'])
ws.append(['', f"Supplementary search: identified {C['spTotal']}; duplicates within the supplementary set {C['spDupWithin']}; already identified by the main search {C['spDupMain']}; new unique records {C['spScreened']}"])

# Title/abstract screening
rows = []
for u in d['unique']:
    code, why_ = d['ta'][u['uid']]
    lab = {'I': 'Include - sought', 'U': 'Uncertain - sought'}.get(code, 'Exclude ' + dm.TA_MAP.get(code, code))
    rows.append([u['uid'], 'Database search', u['title'], u.get('year'), u.get('venue'), u.get('doi') or (('arXiv:' + u['arxiv']) if u.get('arxiv') else ''), '; '.join(u['sources']), lab, why_])
for u in d['s_unique']:
    code, why_ = d['ta_supp'][u['uid']]
    lab = {'I': 'Include - sought', 'U': 'Uncertain - sought'}.get(code, 'Exclude ' + dm.TA_MAP.get(code, code))
    rows.append([u['uid'], 'Database search (supplementary, RQ3)', u['title'], u.get('year'), u.get('venue'), u.get('doi') or (('arXiv:' + u['arxiv']) if u.get('arxiv') else ''), '; '.join(u['sources']), lab, why_])
cc = {c['cid']: c for c in d['chase']}
for k, (code, why_) in d['ta_chase'].items():
    c = cc[k]
    lab = {'I': 'Include - sought', 'U': 'Uncertain - sought', 'DUP': 'Duplicate (removed before screening)'}.get(code, 'Exclude ' + dm.TA_MAP.get(code, code))
    rows.append([k, 'Citation searching (' + c['found_via'].replace('|', ', ') + ')', c['title'], c['year'], c.get('venue'), c.get('doi') or (('arXiv:' + c['arxiv']) if c.get('arxiv') else ''), 'Semantic Scholar citation graph', lab, why_])
sheet('TA_Screening', ['ID', 'Route', 'Title', 'Year', 'Venue', 'DOI / arXiv', 'Sources', 'Decision', 'Reason'], rows, [7, 22, 70, 6, 28, 26, 26, 22, 55])

# Dual screening subset
rng = random.Random(2515)
pool = [u['uid'] for u in d['unique']] + [u['uid'] for u in d['s_unique']]
ND = DUAL_N = round(0.15 * len(pool))
ids = sorted(rng.sample(pool, ND))
uu = {u['uid']: u for u in d['unique'] + d['s_unique']}
tall = dict(d['ta']); tall.update(d['ta_supp'])
last = 6 + ND
ws = wb.create_sheet('Dual_Screening')
ws.append([f'Dual screening of a random 15% subset ({ND} of {len(pool)} database records; random.Random(2515).sample). Each member screens title and abstract independently, without looking at the other columns, and enters Include or Exclude. Column E (AI-assisted first pass) is hidden so it does not bias either screener; unhide it only after both columns are complete.'])
ws.append(['Cohen\'s kappa is calculated only when both columns are complete. It has not been calculated yet, and no agreement figure is reported until it is.'])
ws.append([])
ws.append(['n complete', f'=SUMPRODUCT((F7:F{last}<>"")*(G7:G{last}<>""))'])
ws.append(['Observed agreement', f'=IF(B4={ND},SUMPRODUCT((F7:F{last}=G7:G{last})*(F7:F{last}<>""))/B4,"pending")', 'Expected agreement', f'=IF(B4={ND},(COUNTIF(F7:F{last},"Include")/B4)*(COUNTIF(G7:G{last},"Include")/B4)+(COUNTIF(F7:F{last},"Exclude")/B4)*(COUNTIF(G7:G{last},"Exclude")/B4),"pending")', 'Cohen\'s kappa', f'=IF(B4={ND},IF(D5=1,"undefined",(B5-D5)/(1-D5)),"pending")'])
ws.append(['ID', 'Title', 'Year', 'DOI / arXiv', 'AI-assisted first pass (reference only)', 'Shahoud: Include/Exclude', 'Saif: Include/Exclude', 'Agree?', 'Notes on disagreement and resolution'])
for c in ws[6]: c.fill = HDR; c.font = HF; c.alignment = Alignment(wrap_text=True)
for i, k in enumerate(ids, 7):
    code = tall[k][0]
    ws.append([k, uu[k]['title'], uu[k].get('year'), uu[k].get('doi') or (('arXiv:' + uu[k]['arxiv']) if uu[k].get('arxiv') else ''), 'Include' if code in ('I', 'U') else 'Exclude', '', '', f'=IF(AND(F{i}<>"",G{i}<>""),IF(F{i}=G{i},"Yes","No"),"")', ''])
dv = DataValidation(type='list', formula1='"Include,Exclude"', allow_blank=True); ws.add_data_validation(dv); dv.add(f'F7:G{last}')
for col, w in zip('ABCDEFGHI', [7, 70, 6, 26, 18, 18, 18, 8, 40]): ws.column_dimensions[col].width = w
ws.column_dimensions['E'].hidden = True
for row in ws.iter_rows(min_row=7):
    for c in row: c.alignment = WR
ws['A1'].font = Font(bold=True); ws.freeze_panes = 'A7'

# Full-text
fc = {x['uid']: x for x in d['ftc']}; fcc = {(x.get('cid') or x.get('uid')): x for x in d['ftc_chase']}; fsp = {x['uid']: x for x in d['ftc_supp']}
master = {r['uid']: r for r in d['master']}
rows = []
for u, cand in list(fc.items()) + list(fsp.items()) + list(fcc.items()):
    r = d['ft'][u]
    link = ('https://doi.org/' + cand['doi']) if cand.get('doi') and not cand['doi'].startswith('10.48550') else (('https://arxiv.org/abs/' + cand['arxiv']) if cand.get('arxiv') else '')
    dec = r['decision'] + (' ' + r['reason'] if r['decision'] == 'Exclude' else '')
    rows.append([u, 'Citation searching' if u.startswith('C') else ('Database search (supplementary)' if u.startswith('N') else 'Database search'), cand['title'], cand.get('year'), link,
                 'Not retrieved' if r['decision'] == 'Not retrieved' else 'Retrieved', r.get('read_from', ''), dec,
                 r.get('reason', '') if r['decision'] == 'Not retrieved' else (dm.CODES.get(r.get('reason'), '') if r['decision'] == 'Exclude' else 'Meets IC1-IC6'),
                 r.get('evidence', ''), master[u]['sid'] if u in master else ''])
sheet('FullText', ['ID', 'Route', 'Title', 'Year', 'Link', 'Retrieval', 'Version read', 'Decision', 'Criterion / reason', 'Evidence (page/section)', 'Study ID'], rows,
      [7, 16, 60, 6, 34, 11, 40, 14, 45, 60, 7])

def cite(r):
    au = r['authors'] or []
    a = ', '.join(au[:6]) + (' et al.' if len(au) > 6 else '')
    parts = [a, f"({r['year']})", r['title'] + '.', r['venue'] + ('' if not r.get('volume') else f", {r['volume']}") + (f"({r['issue']})" if r.get('issue') else '') + (f", {r['pages']}" if r.get('pages') else '') + '.']
    if r.get('doi'): parts.append('https://doi.org/' + r['doi'])
    elif r.get('arxiv'): parts.append('https://arxiv.org/abs/' + r['arxiv'])
    return ' '.join(p for p in parts if p)
M = d['master']
sheet('Included', ['Study ID', 'Record ID', 'Full citation (APA-like)', 'Year', 'Type', 'Venue', 'Volume', 'Issue', 'Pages / article no.', 'DOI', 'arXiv', 'Peer-review status (verified)', 'Verified via', 'Version read', 'Route'],
      [[r['sid'], r['uid'], cite(r), r['year'], r['kind'], r['venue'], r.get('volume'), r.get('issue'), r.get('pages'), r.get('doi'), r.get('arxiv'), r['status'], r['src'], r['read_from'], r['source_route']] for r in M],
      [7, 8, 80, 6, 11, 40, 7, 6, 10, 26, 12, 34, 18, 40, 14])
NR = lambda x: x if (x not in (None, '')) else 'Not reported'
sheet('Extraction', ['Study ID', 'Record ID', 'Short citation', 'Problem / aim', 'RQ(s) informed', 'Method', 'Dataset(s)', 'Evaluation / metrics', 'Key findings', 'Limitations', 'Reproducibility', 'QA %', 'Decision evidence (page/section)'],
      [[r['sid'], r['uid'], f"{(r['authors'] or ['?'])[0].split()[-1]} et al. ({r['year']})" if len(r['authors'] or []) > 2 else (' & '.join(a.split()[-1] for a in r['authors']) + f" ({r['year']})"),
        NR(r['problem']), NR(r['rq']), NR(r['method']), NR(r['data']), NR(r['evaluation']), NR(r['findings']), NR(r['limitations']), NR(r['repro']), r['qa_pct'], r['evidence']] for r in M],
      [7, 8, 20, 45, 9, 60, 45, 45, 60, 45, 35, 7, 45])
def band(p): return 'High' if p >= 80 else ('Moderate' if p >= 60 else 'Low')
ws = sheet('Quality', ['Study ID', 'Record ID'] + [q for q, _ in dm.QA_ITEMS] + ['Applicable items', 'Raw score', 'Score %', 'Band', 'Notes'],
           [[r['sid'], r['uid']] + [('N/A' if x == 'NA' else x) for x in r['qa']] + [r['qa_applicable'], r['qa_raw'], r['qa_pct'], band(r['qa_pct']), r['qa_notes']] for r in M],
           [7, 8] + [5] * 10 + [9, 8, 8, 9, 70])
ws.append([]); ws.append(['Items']); [ws.append([q, t]) for q, t in dm.QA_ITEMS]
ws.append(['Scoring', 'Yes = 1, Partial = 0.5, No = 0, N/A excluded from the denominator; Score % = raw / applicable x 100; High >= 80, Moderate 60-79, Low < 60. Scores weight the synthesis; no study was excluded for quality.'])

# PRISMA counts
ws = wb.create_sheet('PRISMA_Counts')
P_ROWS = [
    ('Identification (databases, main search)', 'Records identified: IEEE Xplore', C['idIEEE']), ('', 'ScienceDirect', C['idSD']), ('', 'arXiv', C['idArxiv']),
    ('', 'Semantic Scholar', C['idSS']), ('', 'DBLP', C['idDBLP']), ('', 'Total identified (main search)', C['idTotal']),
    ('Identification (databases, supplementary RQ3 search)', 'Records identified: IEEE Xplore', C['spIEEE']), ('', 'arXiv', C['spArxiv']), ('', 'Semantic Scholar', C['spSS']), ('', 'DBLP', C['spDBLP']),
    ('', 'Total identified (supplementary search)', C['spTotal']),
    ('Identification (databases, combined)', 'Total identified from databases', C['dbAllId']),
    ('', 'Duplicates removed: main search', C['dupRemoved']), ('', 'Duplicates removed: supplementary (within set)', C['spDupWithin']), ('', 'Duplicates removed: supplementary (already in main search)', C['spDupMain']),
    ('', 'Duplicate records removed before screening (combined)', C['dbAllDup']),
    ('Screening (databases)', 'Records screened (title/abstract)', C['dbAllScreened']), ('', 'Records excluded', C['dbAllExcluded']),
    ('', '  EC1 stability not a primary aim', C['dbAllExOne']), ('', '  EC2 non-tabular', C['dbAllExTwo']), ('', '  EC3 explainer/agreement', C['dbAllExThree']), ('', '  EC4 not a full report', C['dbAllExFour']), ('', '  EC5 homonym', C['dbAllExFive']),
    ('', 'Reports sought for retrieval', C['dbAllSought']), ('', 'Reports not retrieved', C['dbAllNotRetrieved']), ('', 'Reports assessed for eligibility', C['dbAllAssessed']),
    ('', 'Reports excluded: EC1', C['dbAllFtExOne']), ('', 'Reports excluded: EC2', C['dbAllFtExTwo']), ('', 'Reports excluded: EC3', C['dbAllFtExThree']), ('', 'Reports excluded: EC4', C['dbAllFtExFour']),
    ('Included (databases)', 'Studies included (main search ' + str(C['dbIncluded']) + ', supplementary ' + str(C['spIncluded']) + ')', C['dbAllIncluded']),
    ('Identification (other methods)', 'Records identified from citation searching (7 seeds, backward and forward)', C['csHarvested']),
    ('', 'Removed before screening: outside 2016-2026', C['csYear']), ('', 'Removed before screening: already identified in databases', C['csAlready']),
    ('', 'Removed before screening: duplicates within citation set', C['csDupWithin'] + C['csDupManual']), ('', 'Removed before screening: automated title filter (no stability AND explanation term)', C['csTitleFilter']),
    ('Screening (other methods)', 'Records screened', C['csScreened']), ('', 'Records excluded', C['csTaExcluded']), ('', 'Reports sought for retrieval', C['csSought']),
    ('', 'Reports not retrieved', C['csNotRetrieved']), ('', 'Reports assessed for eligibility', C['csAssessed']), ('', 'Reports excluded: EC1', C['csFtExcluded']),
    ('Included (other methods)', 'Studies included', C['csIncluded']), ('Total', 'Studies included in the review', C['totalIncluded']),
]
ws.append(['Stage', 'Box', 'Count'])
for c in ws[1]: c.fill = HDR; c.font = HF
for r in P_ROWS: ws.append(list(r))
ws.append([]); ws.append(['Checks', 'Formula', 'Result'])
R_ = lambda b: next(i + 2 for i, (_, x, _) in enumerate(P_ROWS) if x == b)
sp0 = R_('Total identified (main search)') + 1
chk = [
    ('Main-search databases sum to total', f'=SUM(C2:C{R_("Total identified (main search)") - 1})=C{R_("Total identified (main search)")}'),
    ('Supplementary databases sum to total', f'=SUM(C{sp0}:C{R_("Total identified (supplementary search)") - 1})=C{R_("Total identified (supplementary search)")}'),
    ('Main + supplementary = combined', f'=C{R_("Total identified (main search)")}+C{R_("Total identified (supplementary search)")}=C{R_("Total identified from databases")}'),
    ('Identified - duplicates = screened', f'=C{R_("Total identified from databases")}-C{R_("Duplicate records removed before screening (combined)")}=C{R_("Records screened (title/abstract)")}'),
    ('Screened - excluded = sought', f'=C{R_("Records screened (title/abstract)")}-C{R_("Records excluded")}=C{R_("Reports sought for retrieval")}'),
    ('Sought - not retrieved = assessed', f'=C{R_("Reports sought for retrieval")}-C{R_("Reports not retrieved")}=C{R_("Reports assessed for eligibility")}'),
]
for name, f in chk: ws.append([name, f[1:], f])
ws.column_dimensions['A'].width = 30; ws.column_dimensions['B'].width = 80; ws.column_dimensions['C'].width = 10
wb.save(OUT)
print('saved', OUT, 'sheets', wb.sheetnames, 'dedup basis', dict(basis))
