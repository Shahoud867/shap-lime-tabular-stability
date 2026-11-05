import sys, requests, json
q = sys.argv[1]
extra = sys.argv[2] if len(sys.argv) > 2 else ""
url = "https://api.openalex.org/works"
flt = f"title_and_abstract.search:{q}"
if extra: flt += "," + extra
r = requests.get(url, params={"filter": flt, "per-page": 10, "select": "id,display_name,publication_year,doi"}, timeout=60)
d = r.json()
if "meta" not in d:
    print(json.dumps(d)[:800]); sys.exit()
print("COUNT", d["meta"]["count"])
for w in d["results"]:
    print(w["publication_year"], "|", w["display_name"][:110])
