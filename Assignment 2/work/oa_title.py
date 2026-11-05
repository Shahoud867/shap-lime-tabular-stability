import sys, requests
sys.path.insert(0, sys.argv[1]); from oa_fetch import rebuild
for t in sys.argv[2:]:
    r = requests.get("https://api.openalex.org/works", params={"filter": f"title.search:{t}", "per-page": 3,
        "select": "id,display_name,publication_year,type,abstract_inverted_index,primary_location"}, timeout=60).json()
    print("###", t, "->", r.get("meta", {}).get("count"))
    for w in r.get("results", [])[:2]:
        ab = rebuild(w.get("abstract_inverted_index"))
        print("  ", w["publication_year"], w["type"], (w.get("primary_location") or {}).get("source", {}) and w["primary_location"]["source"].get("display_name") if w.get("primary_location") and w["primary_location"].get("source") else None, "|", w["display_name"][:90], "| abstract words:", len(ab.split()))
