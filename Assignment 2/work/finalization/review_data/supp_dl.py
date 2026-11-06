"""Download open-access PDFs for supplementary full-text candidates into ft_pdfs (data only) and extract text with pypdf."""
import requests, sys, os, json, time, re
from pypdf import PdfReader
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36", "Accept": "*/*"}
s = requests.Session(); s.headers.update(UA)
logp = 'final/supp_download_log.json'
log = json.load(open(logp)) if os.path.exists(logp) else {}
def ieee_pdf(doi):
    r = s.get('https://doi.org/' + doi, timeout=60, allow_redirects=True)
    m = re.search(r'/document/(\d+)', r.url)
    return f'https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={m.group(1)}&ref=' if m else None
for arg in sys.argv[1:]:
    uid, url = arg.split('=', 1)
    if url.startswith('ieee:'): url = ieee_pdf(url[5:]) or url
    try:
        r = s.get(url, timeout=120, allow_redirects=True)
        ok = r.status_code == 200 and r.content[:4] == b'%PDF'
        if ok:
            open(f'ft_pdfs/{uid}.pdf', 'wb').write(r.content)
            pages = [(p.extract_text() or '') for p in PdfReader(f'ft_pdfs/{uid}.pdf').pages]
            open(f'ft_txt/{uid}.txt', 'w', encoding='utf-8').write('\n'.join(f'<<PAGE {i+1}>>\n{x}' for i, x in enumerate(pages)))
        log[uid] = {'ok': ok, 'url': url, 'note': f"{r.status_code} {r.headers.get('content-type','')[:30]}"}
        print(uid, 'OK' if ok else '--', log[uid]['note'], flush=True)
    except Exception as e:
        log[uid] = {'ok': False, 'url': url, 'note': str(e)[:90]}; print(uid, 'ERR', str(e)[:90])
    time.sleep(2)
json.dump(log, open(logp, 'w'), indent=1)
