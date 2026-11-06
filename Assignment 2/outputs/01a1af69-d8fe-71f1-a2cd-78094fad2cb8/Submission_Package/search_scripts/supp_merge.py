"""Build the supplementary identification table, de-duplicate it, and remove records already identified by the main search.
Same duplicate rules as the main search: DOI, arXiv id, Semantic Scholar id, exact normalised title; plus one manual link
(DBLP title variant of C068, arXiv 1904.12991)."""
import json, re, collections, os, time
from enrich_lib import s2_batch, s2_match
norm_t = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())
def nd(d):
    if not d: return None
    d = d.lower().strip(); d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d); return d or None
def s2rec(p, source, extra=None):
    ext = p.get("externalIds") or {}
    r = {"source": source, "title": (p.get("title") or "").strip(), "year": p.get("year"), "venue": p.get("venue") or ((p.get("journal") or {}).get("name")),
         "doi": nd(ext.get("DOI")), "arxiv": ext.get("ArXiv"), "s2id": p.get("paperId"), "authors": [a.get("name") for a in (p.get("authors") or [])],
         "pubtypes": p.get("publicationTypes"), "abstract": p.get("abstract") or "", "journal": p.get("journal"), "url": p.get("url"),
         "oa_pdf": (p.get("openAccessPdf") or {}).get("url")}
    if extra: r.update(extra)
    return r
recs = []
ie = json.load(open("final/supp_ieee.json", encoding="utf-8"))
for r in ie["records"]:
    recs.append({"source": "IEEE Xplore", "title": re.sub(r"<[^>]+>", "", r.get("articleTitle", "")).strip(), "year": int(r.get("publicationYear") or 0) or None,
                 "venue": r.get("displayPublicationTitle") or r.get("publicationTitle"), "doi": nd(r.get("doi")), "arxiv": None, "s2id": None,
                 "authors": [a.get("preferredName") or a.get("normalizedName") for a in r.get("authors", [])], "pubtypes": [r.get("contentType")],
                 "abstract": re.sub(r"<[^>]+>", "", r.get("abstract") or ""), "url": "https://ieeexplore.ieee.org/document/" + str(r.get("articleNumber")), "oa_pdf": None,
                 "access": r.get("accessType", {}).get("type") if isinstance(r.get("accessType"), dict) else r.get("accessType")})
for p in json.load(open("final/supp_s2.json", encoding="utf-8"))["data"]: recs.append(s2rec(p, "Semantic Scholar"))
for r in json.load(open("final/supp_arxiv.json", encoding="utf-8"))["records"]:
    aid = re.sub(r"v\d+$", "", r["arxiv_id"])
    recs.append({"source": "arXiv", "title": r["title"], "year": int(r["published"][:4]), "venue": r.get("journal_ref") or "arXiv preprint", "doi": nd(r.get("doi")),
                 "arxiv": aid, "s2id": None, "authors": r["authors"], "pubtypes": ["Preprint"], "abstract": r["abstract"], "url": "https://arxiv.org/abs/" + aid,
                 "oa_pdf": "https://arxiv.org/pdf/" + aid})
dblp = []
for line in open("final/supp_dblp_raw.txt", encoding="utf-8"):
    if line.startswith("#") or not line.strip(): continue
    yr, typ, doi, venue, title = line.rstrip("\n").split("~", 4); dblp.append((int(yr), typ, doi.lower() or None, venue, title.strip().rstrip(".")))
cache_p = "final/supp_dblp_meta.json"
cache = json.load(open(cache_p, encoding="utf-8")) if os.path.exists(cache_p) else {}
need = [("DOI:" + d) if not d.startswith("10.48550") else ("ARXIV:" + d.split("arxiv.")[1]) for _, _, d, _, _ in dblp if d]
need = [k for k in need if k not in cache]
if need: cache.update(s2_batch(need))
for yr, typ, doi, venue, title in dblp:
    key = (("DOI:" + doi) if not doi.startswith("10.48550") else ("ARXIV:" + doi.split("arxiv.")[1])) if doi else None
    p = cache.get(key) if key else None
    if not p:
        tk = "T:" + title
        if tk not in cache: cache[tk] = s2_match(title); time.sleep(1.1)
        p = cache.get(tk)
    base = {"title": title, "year": yr, "venue": venue, "doi": doi, "dblp_type": typ}
    if p:
        r = s2rec(p, "DBLP"); r.update(base); r["doi"] = doi or r["doi"]
    else:
        r = dict(base, source="DBLP", arxiv=None, s2id=None, authors=[], pubtypes=[typ], abstract="", url=("https://doi.org/" + doi) if doi else None, oa_pdf=None)
    if doi and doi.startswith("10.48550/arxiv."): r["arxiv"] = doi.split("arxiv.")[1]
    recs.append(r)
json.dump(cache, open(cache_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for i, r in enumerate(recs, 1):
    r["rec_id"] = f"Q{i:04d}"
    if r.get("arxiv"): r["arxiv"] = re.sub(r"v\d+$", "", r["arxiv"].lower())
    if r.get("doi") and r["doi"].startswith("10.48550/arxiv.") and not r.get("arxiv"): r["arxiv"] = r["doi"].split("arxiv.")[1]
json.dump(recs, open("final/supp_identification.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
parent = {r["rec_id"]: r["rec_id"] for r in recs}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
why = {}
key = collections.defaultdict(list)
for r in recs:
    if r.get("doi") and not r["doi"].startswith("10.48550"): key["doi:" + r["doi"]].append(r["rec_id"])
    if r.get("arxiv"): key["arx:" + r["arxiv"]].append(r["rec_id"])
    if r.get("s2id"): key["s2:" + r["s2id"]].append(r["rec_id"])
    t = norm_t(r["title"])
    if len(t) > 15: key["t:" + t].append(r["rec_id"])
for k, ids in key.items():
    for x in ids[1:]:
        a, b = find(ids[0]), find(x)
        if a != b: parent[b] = a; why.setdefault(x, k.split(":")[0])
groups = collections.defaultdict(list)
for r in recs: groups[find(r["rec_id"])].append(r)
uniq = json.load(open("final/unique.json", encoding="utf-8")); chase = json.load(open("final/chase_candidates.json", encoding="utf-8"))
main_doi, main_ax, main_s2, main_t = {}, {}, {}, {}
for u in uniq:
    if u.get("doi") and not u["doi"].startswith("10.48550"): main_doi[u["doi"]] = u["uid"]
    if u.get("arxiv"): main_ax[u["arxiv"]] = u["uid"]
    main_t[norm_t(u["title"])] = u["uid"]
ident = json.load(open("final/identification.json", encoding="utf-8"))
rec2uid = {rid: u["uid"] for u in uniq for rid in u["rec_ids"]}
for r in ident:
    if r.get("s2id"): main_s2[r["s2id"]] = rec2uid[r["rec_id"]]
    if r.get("doi") and not r["doi"].startswith("10.48550"): main_doi.setdefault(r["doi"], rec2uid[r["rec_id"]])
    if r.get("arxiv"): main_ax.setdefault(r["arxiv"], rec2uid[r["rec_id"]])
    if norm_t(r.get("title")): main_t.setdefault(norm_t(r["title"]), rec2uid[r["rec_id"]])
for c in chase:
    if c.get("doi") and not c["doi"].startswith("10.48550"): main_doi.setdefault(c["doi"], c["cid"])
    if c.get("arxiv"): main_ax.setdefault(c["arxiv"], c["cid"])
    if c.get("s2id"): main_s2.setdefault(c["s2id"], c["cid"])
    main_t.setdefault(norm_t(c["title"]), c["cid"])
main_ax.setdefault("1904.12991", "C068")  # manual: same paper under its DBLP title
def peer(r):
    pt = [str(t) for t in (r.get("pubtypes") or [])]
    return r["source"] == "IEEE Xplore" or r.get("dblp_type") in ("Jour", "Conf") or (any(t in ("JournalArticle", "Conference", "Review") for t in pt) and "arxiv" not in (r.get("venue") or "").lower())
new, overlap, n = [], [], 0
for g, rs in sorted(groups.items(), key=lambda kv: min(r["rec_id"] for r in kv[1])):
    hit = None
    for r in rs:
        hit = hit or (main_doi.get(r["doi"]) if r.get("doi") else None) or (main_ax.get(r["arxiv"]) if r.get("arxiv") else None) or (main_s2.get(r["s2id"]) if r.get("s2id") else None) or main_t.get(norm_t(r["title"]))
    if hit:
        overlap.append({"rec_ids": [r["rec_id"] for r in rs], "matches": hit, "title": rs[0]["title"]}); continue
    n += 1
    prim = sorted(rs, key=lambda r: (0 if peer(r) else 1, 0 if r.get("abstract") else 1, r["rec_id"]))[0]
    new.append({"uid": f"N{n:03d}", "rec_ids": [r["rec_id"] for r in rs], "sources": sorted({r["source"] for r in rs}), "title": prim["title"], "year": prim.get("year"),
                "venue": prim.get("venue"), "doi": next((r["doi"] for r in [prim] + rs if r.get("doi") and not r["doi"].startswith("10.48550")), None) or prim.get("doi"),
                "arxiv": next((r["arxiv"] for r in rs if r.get("arxiv")), None), "authors": next((r["authors"] for r in [prim] + rs if r.get("authors")), []),
                "abstract": max((r.get("abstract") or "" for r in rs), key=len), "pubtypes": sorted({str(t) for r in rs for t in (r.get("pubtypes") or []) if t}),
                "url": prim.get("url"), "oa_pdf": next((r.get("oa_pdf") for r in rs if r.get("oa_pdf")), None), "access": next((r.get("access") for r in rs if r.get("access")), None)})
json.dump({"why": why, "overlap": overlap}, open("final/supp_dedup.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(new, open("final/supp_unique.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
src = collections.Counter(r["source"] for r in recs)
print("identified", len(recs), dict(src), "| within-supp duplicates", len(recs) - len(groups), "| records already in main search", sum(len(o["rec_ids"]) for o in overlap), "| new unique", len(new))
