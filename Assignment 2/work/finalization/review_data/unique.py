import json, re, collections
recs = json.load(open("final/identification.json", encoding="utf-8"))
par = json.load(open("final/dedup_exact.json"))["parent"]
MANUAL = {"R0227": "R0260"}  # S2 split entry of the same arXiv paper (Chalasani et al. 2018/2020); same authors and lineage
def find(x):
    while par[x] != x: x = par[x]
    return x
for a, b in MANUAL.items(): par[find(a)] = find(b)
groups = collections.defaultdict(list)
for r in recs: groups[find(r["rec_id"])].append(r)
PR = {"IEEE Xplore": 0, "ScienceDirect": 0}
def peer(r):
    pt = [str(t) for t in (r.get("pubtypes") or [])]
    if r["source"] in PR: return True
    if r.get("dblp_type") in ("Jour", "Conf"): return True
    if any(t in ("JournalArticle", "Conference", "Review") for t in pt) and "arxiv" not in (r.get("venue") or "").lower(): return True
    return False
uniq = []
for i, (g, rs) in enumerate(sorted(groups.items(), key=lambda kv: min(r["rec_id"] for r in kv[1])), 1):
    prim = sorted(rs, key=lambda r: (0 if peer(r) else 1, 0 if r.get("abstract") else 1, r["rec_id"]))[0]
    abst = max((r.get("abstract") or "" for r in rs), key=len)
    u = {"uid": f"U{i:03d}", "rec_ids": [r["rec_id"] for r in rs], "sources": sorted({r["source"] + (f" ({r['sub']})" if r.get("sub") else "") for r in rs}),
         "title": prim["title"], "year": prim.get("year"), "venue": prim.get("venue"), "doi": next((r["doi"] for r in [prim] + rs if r.get("doi") and not r["doi"].startswith("10.48550")), None) or prim.get("doi"),
         "arxiv": next((r["arxiv"] for r in rs if r.get("arxiv")), None), "authors": next((r["authors"] for r in [prim] + rs if r.get("authors")), []),
         "abstract": abst, "peer_reviewed_version": any(peer(r) for r in rs), "pubtypes": sorted({str(t) for r in rs for t in (r.get("pubtypes") or []) if t}),
         "url": prim.get("url"), "oa_pdf": next((r.get("oa_pdf") for r in rs if r.get("oa_pdf")), None)}
    uniq.append(u)
json.dump(uniq, open("final/unique.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("identified", len(recs), "unique", len(uniq), "duplicates removed", len(recs) - len(uniq))
print("within-source dup groups:", sum(1 for rs in groups.values() if len(rs) > 1))
print(collections.Counter(len(u["sources"]) for u in uniq))
