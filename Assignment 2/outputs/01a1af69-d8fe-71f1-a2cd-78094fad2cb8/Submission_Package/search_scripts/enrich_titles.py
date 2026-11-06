import json, time, sys
from enrich_lib import s2_match
meta = json.load(open("final/s2_by_doi.json", encoding="utf-8"))
dblp = []
for line in open("final/dblp_raw.txt", encoding="utf-8"):
    if line.startswith("#") or not line.strip(): continue
    yr, typ, doi, venue, title = line.rstrip("\n").split("~", 4)
    dblp.append((doi.lower(), title.strip()))
miss = [t for d, t in dblp if not d or not meta.get("DOI:" + d)]
print("to look up:", len(miss), flush=True)
out = {}
for t in miss:
    out[t] = s2_match(t.rstrip("."))
    print(("OK " if out[t] else "-- ") + t[:70], flush=True)
    time.sleep(1.1)
json.dump(out, open("final/s2_by_title.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("done", sum(1 for v in out.values() if v), "/", len(out), flush=True)
