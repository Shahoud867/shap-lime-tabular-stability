"""Cross-check that the PDF, workbook, libraries and counts agree."""
import sys, os, re, json, collections
import openpyxl
from pypdf import PdfReader
D = sys.argv[1]
R = os.path.join(D, 'report')
cnt = {}
for line in open(os.path.join(R, 'counts.tex'), encoding='utf8'):
    m = re.match(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', line)
    if m: cnt[m.group(1)] = m.group(2)
ok = True
def check(name, cond, detail=''):
    global ok
    ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name + (' | ' + str(detail) if detail else ''))

wb = openpyxl.load_workbook(sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, 'screening', 'MBR_A2_Screening_Workbook.xlsx'))
vrows = list(wb['Validation'].iter_rows(min_row=2, values_only=True))
blank = next(i for i, r in enumerate(vrows) if not (r and r[0]))
check('workbook validation rows = nKnown', blank == int(cnt['nKnown']), blank)
vsupp = [r for r in vrows[blank + 2:] if r and r[0]]
check('workbook supplementary validation rows = suppKnown', len(vsupp) == int(cnt['suppKnown']), len(vsupp))
rows = [(r[0], r[1], r[2]) for r in wb['PRISMA_Counts'].iter_rows(min_row=2, values_only=True) if r and r[1] and isinstance(r[2], int)]
pc = {}
for st, box, v in rows:
    pc.setdefault(box, v)  # first occurrence = database column
cs = {}
seen_other = False
for st, box, v in rows:
    if st and 'other methods' in st: seen_other = True
    if seen_other: cs.setdefault(box, v)
for box, mac in [('Reports sought for retrieval', 'csSought'), ('Reports not retrieved', 'csNotRetrieved'), ('Reports assessed for eligibility', 'csAssessed'), ('Studies included', 'csIncluded')]:
    check(f'workbook (other methods) {box} = {mac}', str(cs.get(box)) == cnt[mac], (cs.get(box), cnt[mac]))
pairs = [('Total identified (main search)', 'idTotal'), ('Total identified (supplementary search)', 'spTotal'), ('Total identified from databases', 'dbAllId'),
         ('Duplicates removed: main search', 'dupRemoved'), ('Duplicate records removed before screening (combined)', 'dbAllDup'), ('Records screened (title/abstract)', 'dbAllScreened'),
         ('Records excluded', 'dbAllExcluded'), ('Reports sought for retrieval', 'dbAllSought'), ('Reports not retrieved', 'dbAllNotRetrieved'), ('Reports assessed for eligibility', 'dbAllAssessed'),
         (f"Studies included (main search {cnt['dbIncluded']}, supplementary {cnt['spIncluded']})", 'dbAllIncluded'),
         ('Studies included in the review', 'totalIncluded'), ('Records identified from citation searching (7 seeds, backward and forward)', 'csHarvested'), ('Records screened', 'csScreened')]
for box, mac in pairs:
    check(f'workbook {box} = counts.tex {mac}', str(pc.get(box)) == cnt[mac], (pc.get(box), cnt[mac]))
n_ta = wb['TA_Screening'].max_row - 1; n_ft = wb['FullText'].max_row - 1
check('TA sheet rows = main + supplementary + citation candidates', n_ta == int(cnt['nUnique']) + int(cnt['spScreened']) + int(cnt['csCandidates']), n_ta)
check('FullText rows = reports sought', n_ft == int(cnt['totalSought']), n_ft)
ft = collections.Counter(r[7].split()[0] for r in wb['FullText'].iter_rows(min_row=2, values_only=True))
check('FullText includes = included', ft['Include'] == int(cnt['totalIncluded']), dict(ft))
check('FullText not retrieved', ft['Not'] == int(cnt['totalNotRetrieved']), ft['Not'])
for sh in ('Included', 'Extraction'):
    check(f'{sh} rows = totalIncluded', wb[sh].max_row - 1 == int(cnt['totalIncluded']), wb[sh].max_row - 1)
q = [r for r in wb['Quality'].iter_rows(min_row=2, values_only=True) if r and isinstance(r[0], str) and re.match(r'S\d\d$', r[0])]
check('Quality rows = totalIncluded', len(q) == int(cnt['totalIncluded']), len(q))
bib = open(os.path.join(D, 'library', 'included_studies.bib'), encoding='utf8').read()
check('BibTeX entries = totalIncluded', len(re.findall(r'^@\w+\{S\d\d,', bib, re.M)) == int(cnt['totalIncluded']))
ris = open(os.path.join(D, 'library', 'included_studies.ris'), encoding='utf8').read()
check('RIS included = totalIncluded', ris.count('TY  - ') == int(cnt['totalIncluded']))
allr = open(os.path.join(D, 'library', 'all_screened_records.ris'), encoding='utf8').read()
check('RIS all records = TA sheet rows', allr.count('TY  - ') == n_ta, allr.count('TY  - '))
pdf = PdfReader(os.path.join(R, 'main.pdf'))
txt = '\n'.join(p.extract_text() or '' for p in pdf.pages)
ref = txt[re.search(r'References\s*\n?\s*\[1\]', txt).start():]
m = re.search(r'\[29\]\s*([^\[]{0,160})', ref)
check('PDF reference [29] is Ballegeer (S29)', m and 'Ballegeer' in m.group(1), m.group(1)[:80] if m else None)
m = re.search(r'\[62\]\s*([^\[]{0,120})', ref)
check('PDF reference [62] is Tolay (S62)', m and 'Tolay' in m.group(1), m.group(1)[:80] if m else None)
for n, who in [(75, 'Thackshanaramana'), (77, 'Breeden'), (79, 'Xia')]:
    m = re.search(r'\[%d\]\s*([^\[]{0,120})' % n, ref)
    check(f'PDF reference [{n}] is {who} (S{n})', m and who in m.group(1), m.group(1)[:80] if m else None)
ref_page = next(i for i, pg in enumerate(pdf.pages) if re.search(r'References\s*\n?\s*\[1\]', pg.extract_text() or ''))
check('main body ends by page 7 (references start on page <= 7)', ref_page + 1 <= 7, ref_page + 1)
p4 = '\n'.join((pdf.pages[i].extract_text() or '') for i in range(3, 7))
for mac in ['dbAllId', 'dbAllDup', 'dbAllScreened', 'dbAllExcluded', 'dbAllSought', 'dbAllNotRetrieved', 'dbAllAssessed', 'dbAllFtExcluded', 'totalIncluded', 'csHarvested', 'csScreened']:
    label = 'n = ' + cnt[mac]
    check(f'PRISMA figure shows {label} ({mac})', label.replace(' ', '') in p4.replace(' ', '').replace('\u00a0', ''), cnt[mac])
c = int(cnt['taScreened']) + int(cnt['spScreened']) + int(cnt['csScreened'])
check('totalScreened = main + supplementary + citation screened', c == int(cnt['totalScreened']), c)
check('dbAllIncluded + csIncluded = totalIncluded', int(cnt['dbAllIncluded']) + int(cnt['csIncluded']) == int(cnt['totalIncluded']))
print('ALL PASS' if ok else 'SOME CHECKS FAILED')
