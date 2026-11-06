"""Fetch lawful open-access full texts for full-text candidates into a separate data folder (no execution of content)."""
import json, sys, os, re, time, requests
base, out_dir = sys.argv[1], sys.argv[2]
rows = json.load(open(os.path.join(base, "final", "ft_candidates_chase.json"), encoding="utf-8"))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
log = {}
def try_get(url):
    try:
        r = requests.get(url, headers=UA, timeout=90, allow_redirects=True)
        ct = r.headers.get("content-type", "")
        if r.status_code == 200 and (r.content[:4] == b"%PDF" or "pdf" in ct): return r.content, r.url
        return None, f"{r.status_code} {ct[:40]} {r.url[:80]}"
    except Exception as e: return None, str(e)[:80]
for x in rows:
    k = x["uid"]; cands = []
    if x.get("arxiv"): cands.append("https://arxiv.org/pdf/" + x["arxiv"])
    if x.get("oa_pdf"): cands.append(x["oa_pdf"])
    if x.get("doi") and x["doi"].startswith("10.1109/access"):
        pass
    got = None; notes = []
    for c in cands:
        data, info = try_get(c)
        if data:
            open(os.path.join(out_dir, k + ".pdf"), "wb").write(data); got = info; break
        notes.append(f"{c[:70]} -> {info}")
        time.sleep(1)
    log[k] = {"ok": bool(got), "src": got, "notes": notes}
    print(k, "OK" if got else "--", (got or "; ".join(notes))[:140], flush=True)
    time.sleep(1.5)
json.dump(log, open(os.path.join(base, "final", "ft_download_log_chase.json"), "w"), indent=1)
