import json, re, difflib, collections
recs = json.load(open("final/identification.json", encoding="utf-8"))
norm_t = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())
rank = lambda r: (0 if (r.get("pubtypes") and any(t in ("JournalArticle", "Conference", "Journals", "Conferences", "Journal Article", "Conference Paper") for t in (r.get("pubtypes") or []))) or r["source"] in ("IEEE Xplore", "ScienceDirect") or (r.get("dblp_type") in ("Jour", "Conf")) else 1)
# union-find
parent = {r["rec_id"]: r["rec_id"] for r in recs}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b, why):
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra; links.append((a, b, why))
links = []
key = collections.defaultdict(list)
for r in recs:
    if r.get("doi"): key["doi:" + r["doi"]].append(r["rec_id"])
    if r.get("arxiv"): key["arx:" + r["arxiv"]].append(r["rec_id"])
    if r.get("s2id"): key["s2:" + r["s2id"]].append(r["rec_id"])
    t = norm_t(r.get("title"))
    if len(t) > 15: key["t:" + t].append(r["rec_id"])
for k, ids in key.items():
    for x in ids[1:]: union(ids[0], x, k.split(":")[0])
# fuzzy titles
byid = {r["rec_id"]: r for r in recs}
titles = [(r["rec_id"], norm_t(r.get("title"))) for r in recs if len(norm_t(r.get("title"))) > 15]
fuzzy = []
for i in range(len(titles)):
    for j in range(i + 1, len(titles)):
        a, b = titles[i][1], titles[j][1]
        if find(titles[i][0]) == find(titles[j][0]): continue
        if abs(len(a) - len(b)) > 25: continue
        s = difflib.SequenceMatcher(None, a, b).ratio()
        if s >= 0.75: fuzzy.append((titles[i][0], titles[j][0], round(s, 3), byid[titles[i][0]]["title"], byid[titles[j][0]]["title"]))
json.dump({"fuzzy": fuzzy}, open("final/fuzzy_candidates.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("fuzzy candidates:", len(fuzzy))
for f in fuzzy: print(f)

miss = [r for r in recs if r.get("needs_meta")]
print("records lacking metadata:", len(miss))
for r in miss: print("  ", r["rec_id"], r["source"], r.get("doi"), r.get("title","")[:80])
groups = collections.defaultdict(list)
for r in recs: groups[find(r["rec_id"])].append(r["rec_id"])
multi = [g for g in groups.values() if len(g) > 1]
print("unique after exact rules:", len(groups), "multi-member groups:", len(multi))
json.dump({"links": links, "parent": {k: find(k) for k in parent}}, open("final/dedup_exact.json", "w", encoding="utf-8"), indent=1)
