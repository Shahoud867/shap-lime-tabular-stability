import sys, requests, json, time
from oa_fetch import rebuild
def fetch(filters, years="2016-2026"):
    out, page = [], 1
    flt = ",".join(filters + [f"publication_year:{years}"])
    while True:
        for attempt in range(5):
            r = requests.get("https://api.openalex.org/works", params={"filter": flt, "per-page": 200, "page": page,
                "select": "id,doi,display_name,publication_year,type,primary_location,authorships,abstract_inverted_index,language,biblio"}, timeout=90)
            d = r.json()
            if "results" in d: break
            time.sleep(2)
        else: raise SystemExit(json.dumps(d)[:400])
        for w in d["results"]:
            loc = w.get("primary_location") or {}; src = loc.get("source") or {}
            out.append({"openalex": w["id"], "doi": w.get("doi"), "title": w["display_name"], "year": w["publication_year"],
                "type": w.get("type"), "venue": src.get("display_name"), "source_type": src.get("type"),
                "authors": [a["author"]["display_name"] for a in w.get("authorships", [])], "language": w.get("language"),
                "biblio": w.get("biblio"), "abstract": rebuild(w.get("abstract_inverted_index"))})
        if len(out) >= d["meta"]["count"] or not d["results"]: break
        page += 1; time.sleep(1.2)
    return out
if __name__ == "__main__":
    outp = sys.argv[1]; filters = sys.argv[2:]
    recs = fetch(filters); json.dump(recs, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(len(recs))
