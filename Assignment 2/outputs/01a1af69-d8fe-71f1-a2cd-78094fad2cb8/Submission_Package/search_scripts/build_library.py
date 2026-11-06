"""Write BibTeX/RIS libraries and the full-text links list from the review data."""
import json, os, sys, re, csv
import data_model as dm
S, D = sys.argv[1], sys.argv[2]
d = dm.load(S); M = d['master']

def tex(s):
    s = (s or '').replace('–', '--').replace('—', '---').replace('‘', "'").replace('’', "'").replace('“', '``').replace('”', "''")
    s = s.replace(' ', ' ').replace('‐', '-').replace('‑', '-')
    for a, b in [('\\', ''), ('&', '\\&'), ('%', '\\%'), ('_', '\\_'), ('#', '\\#'), ('$', '\\$')]: s = s.replace(a, b)
    return s
def bibauthors(au):
    # a name whose last token is a lone initial (e.g. 'Thackshanaramana B') is braced so the style does not abbreviate it
    return ' and '.join(('{' + tex(a) + '}') if re.search(r'\s[A-Z]\.?$', a) else tex(a) for a in au) if au else 'Anonymous'
def version_link(r):
    if r.get('fulltext_url'): return r['fulltext_url']
    rf = r.get('read_from') or ''
    if 'Europe PMC' in rf: return 'https://europepmc.org/article/PMC/PMC13014406'
    if 'Zenodo' in rf and r['uid'] == 'U017': return 'https://zenodo.org/records/11192435'
    if 'arXiv' in rf and r.get('arxiv'): return 'https://arxiv.org/abs/' + r['arxiv']
    if r.get('doi'): return 'https://doi.org/' + r['doi']
    if r.get('arxiv'): return 'https://arxiv.org/abs/' + r['arxiv']
    return ''

bib = []
for r in M:
    k = r['sid']; title = '{{' + tex(r['title']) + '}}'
    f = {'author': bibauthors(r['authors']), 'title': title, 'year': str(r['year'])}
    if r['kind'] == 'journal':
        typ = 'article'; f['journal'] = tex(r['venue'])
        if r.get('volume'): f['volume'] = str(r['volume'])
        if r.get('issue'): f['number'] = str(r['issue'])
        if r.get('pages'): f['pages'] = str(r['pages']).replace('-', '--')
    elif r['kind'] in ('conference', 'workshop'):
        typ = 'inproceedings'; f['booktitle'] = tex(re.sub(r';\s*arXiv:\S+$', '', r['venue']))
        if r.get('pages'): f['pages'] = str(r['pages']).replace('-', '--')
    else:
        typ = 'misc'; f['howpublished'] = 'arXiv preprint arXiv:' + (r.get('arxiv') or '')
        if r['uid'] in ('U524', 'U528'): f['note'] = tex(r['status'])
    if r.get('doi'): f['doi'] = r['doi']
    if r.get('arxiv') and r['kind'] != 'preprint': f['note'] = 'arXiv:' + r['arxiv'] + ' version read'
    bib.append(f'@{typ}{{{k},\n' + ',\n'.join(f'  {a} = {{{b}}}' if a != 'title' else f'  title = {b}' for a, b in f.items()) + '\n}\n')
open(os.path.join(D, 'library', 'included_studies.bib'), 'w', encoding='utf8').write(f'% Included studies (Study IDs S01-S{len(M):02d}); generated from the review data\n\n' + '\n'.join(bib))

def ris_rec(ty, title, authors, year, venue, doi=None, arxiv=None, vol=None, issue=None, pages=None, note=None, rid=None):
    L = [f'TY  - {ty}']
    for a in authors or []: L.append(f'AU  - {a}')
    L.append(f'TI  - {title}')
    if year: L.append(f'PY  - {year}')
    if venue: L.append(('JO  - ' if ty == 'JOUR' else 'T2  - ') + venue)
    if vol: L.append(f'VL  - {vol}')
    if issue: L.append(f'IS  - {issue}')
    if pages:
        p = str(pages).split('-'); L.append(f'SP  - {p[0]}')
        if len(p) > 1: L.append(f'EP  - {p[1]}')
    if doi: L.append(f'DO  - {doi}'); L.append(f'UR  - https://doi.org/{doi}')
    elif arxiv: L.append(f'UR  - https://arxiv.org/abs/{arxiv}')
    if rid: L.append(f'ID  - {rid}')
    if note: L.append(f'N1  - {note}')
    L.append('ER  - ')
    return '\n'.join(L)
TY = {'journal': 'JOUR', 'conference': 'CPAPER', 'workshop': 'CPAPER', 'preprint': 'UNPB'}
open(os.path.join(D, 'library', 'included_studies.ris'), 'w', encoding='utf8').write('\n\n'.join(
    ris_rec(TY[r['kind']], r['title'], r['authors'], r['year'], r['venue'], r.get('doi'), r.get('arxiv'), r.get('volume'), r.get('issue'), r.get('pages'),
            f"{r['sid']} ({r['uid']}); {r['status']}; QA {r['qa_pct']}%", r['sid']) for r in M) + '\n')

recs = []
for u in d['unique']:
    code, why = d['ta'][u['uid']]
    ft = d['ft'].get(u['uid'])
    dec = ('TA: ' + ({'I': 'include', 'U': 'uncertain'}.get(code) or ('exclude ' + dm.TA_MAP.get(code, code)))) + (f"; FT: {ft['decision']}" + (f" {ft['reason']}" if ft and ft['decision'] == 'Exclude' else '') if ft else '')
    doi = u.get('doi') if u.get('doi') and not u['doi'].startswith('10.48550') else None
    recs.append(ris_rec('GEN', u['title'], u.get('authors'), u.get('year'), u.get('venue'), doi, u.get('arxiv'), note=f"{u['uid']}; sources: {', '.join(u['sources'])}; {dec}", rid=u['uid']))
for u in d['s_unique']:
    code, why = d['ta_supp'][u['uid']]
    ft = d['ft'].get(u['uid'])
    dec = ('TA: ' + ({'I': 'include', 'U': 'uncertain'}.get(code) or ('exclude ' + dm.TA_MAP.get(code, code)))) + (f"; FT: {ft['decision']}" + (f" {ft['reason']}" if ft and ft['decision'] == 'Exclude' else '') if ft else '')
    doi = u.get('doi') if u.get('doi') and not u['doi'].startswith('10.48550') else None
    recs.append(ris_rec('GEN', u['title'], u.get('authors'), u.get('year'), u.get('venue'), doi, u.get('arxiv'), note=f"{u['uid']}; supplementary search (RQ3); sources: {', '.join(u['sources'])}; {dec}", rid=u['uid']))
cc = {c['cid']: c for c in d['chase']}
for k, (code, why) in d['ta_chase'].items():
    c = cc[k]; ft = d['ft'].get(k)
    dec = 'TA: ' + ({'I': 'include', 'U': 'uncertain', 'DUP': 'duplicate'}.get(code) or ('exclude ' + dm.TA_MAP.get(code, code))) + (f"; FT: {ft['decision']}" + (f" {ft['reason']}" if ft['decision'] == 'Exclude' else '') if ft else '')
    recs.append(ris_rec('GEN', c['title'], c.get('authors'), c.get('year'), c.get('venue'), c.get('doi') if c.get('doi') and not c['doi'].startswith('10.48550') else None, c.get('arxiv'),
                        note=f"{k}; citation searching via {c['found_via']}; {dec}", rid=k))
open(os.path.join(D, 'library', 'all_screened_records.ris'), 'w', encoding='utf8').write('\n\n'.join(recs) + '\n')

# links
rows = [[r['sid'], r['uid'], (r['authors'] or ['?'])[0], r['year'], r['title'], ('https://doi.org/' + r['doi']) if r.get('doi') else '', version_link(r), r['read_from']] for r in M]
with open(os.path.join(D, 'links', 'included_fulltext_links.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['Study ID', 'Record ID', 'First author', 'Year', 'Title', 'Publisher DOI link', 'Full text read (link)', 'Version read']); w.writerows(rows)
fc = {x['uid']: x for x in d['ftc']}; fc.update({(x.get('cid') or x.get('uid')): x for x in d['ftc_chase']}); fc.update({x['uid']: x for x in d['ftc_supp']})
nr = [(u, r) for u, r in d['ft'].items() if r['decision'] == 'Not retrieved']
with open(os.path.join(D, 'links', 'reports_not_retrieved.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['Record ID', 'Title', 'Year', 'DOI link', 'Reason not retrieved'])
    for u, r in nr:
        c = fc[u]; w.writerow([u, c['title'], c.get('year'), ('https://doi.org/' + c['doi']) if c.get('doi') else '', r['reason']])
md = ['# Full-text links', '', 'Copies of the PDFs are not redistributed here. Each included study is listed with its publisher DOI and the exact version that was read.', '',
      '## Included studies', '', '| Study | First author (year) | Title | DOI | Version read |', '|---|---|---|---|---|']
for r in rows: md.append(f"| {r[0]} | {r[2].split()[-1]} ({r[3]}) | {r[4]} | {r[5] or '-'} | {r[6] or '-'} |")
md += ['', '## Reports sought but not retrieved', '', 'These can be requested through the HEC Digital Library or the FAST library, then assessed with the same criteria.', '', '| Record | Title | Year | DOI | Reason |', '|---|---|---|---|---|']
for u, r in nr:
    c = fc[u]; md.append(f"| {u} | {c['title']} | {c.get('year')} | {('https://doi.org/' + c['doi']) if c.get('doi') else '-'} | {r['reason']} |")
open(os.path.join(D, 'links', 'fulltext_links.md'), 'w', encoding='utf8').write('\n'.join(md) + '\n')
print('bib', len(bib), 'ris all', len(recs), 'not retrieved', len(nr))
