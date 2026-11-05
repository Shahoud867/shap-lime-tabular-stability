import json, requests, time, re, sys
FIELDS = "title,abstract,year,venue,publicationTypes,externalIds,authors,journal,publicationVenue,url,isOpenAccess,openAccessPdf,publicationDate"
def s2_batch(ids):
    out = {}
    for i in range(0, len(ids), 400):
        chunk = ids[i:i+400]
        for a in range(8):
            r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch", params={"fields": FIELDS}, json={"ids": chunk}, timeout=120)
            if r.status_code == 200: break
            time.sleep(4 * (a + 1))
        else: raise SystemExit(f"batch fail {r.status_code} {r.text[:200]}")
        for k, v in zip(chunk, r.json()): out[k] = v
        time.sleep(2)
    return out
def s2_match(title):
    for a in range(8):
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search/match", params={"query": title, "fields": FIELDS}, timeout=60)
        if r.status_code == 200: return (r.json().get("data") or [None])[0]
        if r.status_code == 404: return None
        time.sleep(4 * (a + 1))
    return None
