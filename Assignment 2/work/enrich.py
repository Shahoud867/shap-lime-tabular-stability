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
# ScienceDirect
sd = []
cur = None
for line in open("final/sd_raw.txt", encoding="utf-8"):
    line = line.strip()
    if line.startswith("#"): cur = line.split()[0][1:]; continue
    if not line: continue
    doi, typ = line.split("~", 1)
    sd.append({"source": "ScienceDirect", "sub": cur, "doi": doi.lower(), "sd_type": typ})
dblp = []
for line in open("final/dblp_raw.txt", encoding="utf-8"):
    if line.startswith("#") or not line.strip(): continue
    yr, typ, doi, venue, title = line.rstrip("\n").split("~", 4)
    dblp.append({"source": "DBLP", "year": int(yr), "dblp_type": typ, "doi": doi.lower(), "venue": venue, "title": title.strip()})
print("SD", len(sd), "DBLP", len(dblp))
ids = sorted({"DOI:" + r["doi"] for r in sd + dblp if r["doi"]})
meta = s2_batch(ids)
json.dump(meta, open("final/s2_by_doi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
miss = [r for r in dblp if not r["doi"] or not meta.get("DOI:" + r["doi"])]
m2 = {}
for r in miss:
    m2[r["title"]] = s2_match(r["title"].rstrip("."))
    time.sleep(1.2)
json.dump(m2, open("final/s2_by_title.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("doi lookups", len(ids), "found", sum(1 for v in meta.values() if v), "title lookups", len(m2), "found", sum(1 for v in m2.values() if v))
