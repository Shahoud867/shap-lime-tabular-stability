import sys, requests, json, time
FIELDS = "title,abstract,year,venue,publicationTypes,externalIds,authors,journal,publicationVenue,url,isOpenAccess,openAccessPdf,publicationDate"
def fetch(q, years="2016-2026", maxn=5000):
    out, token, total = [], None, None
    while True:
        params = {"query": q, "fields": FIELDS, "year": years}
        if token: params["token"] = token
        for attempt in range(8):
            r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", params=params, timeout=90)
            if r.status_code == 200: break
            time.sleep(3 * (attempt + 1))
        else: raise SystemExit(f"HTTP {r.status_code} {r.text[:300]}")
        d = r.json(); total = d.get("total"); out += d.get("data", [])
        token = d.get("token")
        if not token or len(out) >= maxn: break
        time.sleep(1.5)
    return total, out
if __name__ == "__main__":
    q, outp = sys.argv[1], sys.argv[2]
    total, recs = fetch(q)
    json.dump({"query": q, "total": total, "data": recs}, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(total, len(recs))
