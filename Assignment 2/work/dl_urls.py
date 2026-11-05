"""Fetch specific open-access PDF URLs into the untrusted data folder; log results."""
import requests, sys, os, json, time
base, out_dir = sys.argv[1], sys.argv[2]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36", "Accept": "*/*"}
logp = os.path.join(base, 'final', 'ft_download_log_extra.json')
log = json.load(open(logp)) if os.path.exists(logp) else {}
for arg in sys.argv[3:]:
    uid, url = arg.split('=', 1)
    try:
        r = requests.get(url, headers=UA, timeout=90, allow_redirects=True)
        ok = r.status_code == 200 and r.content[:4] == b'%PDF'
        if ok: open(os.path.join(out_dir, uid + '.pdf'), 'wb').write(r.content)
        log[uid] = {'ok': ok, 'url': url, 'note': f"{r.status_code} {r.headers.get('content-type','')[:30]} {r.url[:90]}"}
        print(uid, 'OK' if ok else '--', log[uid]['note'], flush=True)
    except Exception as e:
        log[uid] = {'ok': False, 'url': url, 'note': str(e)[:90]}; print(uid, 'ERR', str(e)[:90])
    time.sleep(2)
json.dump(log, open(logp, 'w'), indent=1)
