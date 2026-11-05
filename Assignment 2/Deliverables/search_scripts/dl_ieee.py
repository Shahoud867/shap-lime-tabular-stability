"""Resolve IEEE DOIs to article numbers and fetch the PDF only where IEEE serves it openly."""
import requests, re, sys, os, json, time
base, out_dir = sys.argv[1], sys.argv[2]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
s = requests.Session(); s.headers.update(UA)
items = [a.split('=') for a in sys.argv[3:]]
logp = os.path.join(base, 'final', 'ft_download_log_ieee.json')
log = json.load(open(logp)) if os.path.exists(logp) else {}
for uid, doi in items:
    try:
        r = s.get('https://doi.org/' + doi, timeout=60, allow_redirects=True)
        m = re.search(r'/document/(\d+)', r.url)
        if not m:
            log[uid] = {'ok': False, 'note': 'no arnumber ' + r.url[:80]}; print(uid, '-- no arnumber', r.url[:80]); continue
        arn = m.group(1)
        p = s.get(f'https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arn}&ref=', timeout=90)
        ok = p.status_code == 200 and p.content[:4] == b'%PDF'
        if ok:
            open(os.path.join(out_dir, uid + '.pdf'), 'wb').write(p.content)
        log[uid] = {'ok': ok, 'arnumber': arn, 'note': f'{p.status_code} {p.headers.get("content-type","")[:30]}'}
        print(uid, arn, 'OK' if ok else '--', p.status_code, p.headers.get('content-type','')[:30], flush=True)
    except Exception as e:
        log[uid] = {'ok': False, 'note': str(e)[:80]}; print(uid, 'ERR', str(e)[:80])
    time.sleep(2)
json.dump(log, open(logp, 'w'), indent=1)
