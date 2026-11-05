import sys, requests, json, re, time
def rebuild(inv):
    if not inv: return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs: pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))
def fetch(q, years="2016-2026"):
    out, page = [], 1
    while True:
        r = requests.get("https://api.openalex.org/works", params={
            "filter": f"title_and_abstract.search:{q},publication_year:{years}",
            "per-page": 200, "page": page,
            "select": "id,doi,display_name,publication_year,type,primary_location,authorships,abstract_inverted_index,language,biblio"}, timeout=90)
        d = r.json()
        if "results" not in d: raise SystemExit(json.dumps(d)[:500])
        for w in d["results"]:
            loc = w.get("primary_location") or {}
            src = (loc.get("source") or {})
            out.append({"openalex": w["id"], "doi": w.get("doi"), "title": w["display_name"], "year": w["publication_year"],
                        "type": w.get("type"), "venue": src.get("display_name"), "source_type": src.get("type"),
                        "authors": [a["author"]["display_name"] for a in w.get("authorships", [])],
                        "language": w.get("language"), "biblio": w.get("biblio"),
                        "abstract": rebuild(w.get("abstract_inverted_index"))})
        if len(out) >= d["meta"]["count"] or not d["results"]: break
        page += 1
        time.sleep(0.3)
    return out
if __name__ == "__main__":
    q, outp = sys.argv[1], sys.argv[2]
    recs = fetch(q)
    json.dump(recs, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(recs))
