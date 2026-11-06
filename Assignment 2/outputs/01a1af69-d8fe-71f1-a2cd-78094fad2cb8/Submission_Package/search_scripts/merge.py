"""Build the identification table (one row per retrieved record) and de-duplicate.
Duplicate rule (documented in the report): two records are the same study if
 (a) their normalised DOIs match, or (b) one is an arXiv preprint whose DOI/arXiv id matches the other's arXiv id, or
 (c) their normalised titles match exactly, or (d) title similarity >= 0.92 (difflib) AND same first-author surname, confirmed manually.
The retained version is the peer-reviewed one when available (journal/conference > preprint), otherwise the earliest.
"""
import json, re, difflib, collections
norm_t = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())
def nd(d):
    if not d: return None
    d = d.lower().strip(); d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d); return d or None
def s2_to_rec(p, source, sub=None, extra=None):
    ext = p.get("externalIds") or {}
    rec = {"source": source, "sub": sub, "title": (p.get("title") or "").strip(), "year": p.get("year"),
           "venue": p.get("venue") or ((p.get("journal") or {}).get("name")), "doi": nd(ext.get("DOI")),
           "arxiv": ext.get("ArXiv"), "s2id": p.get("paperId"), "authors": [a.get("name") for a in (p.get("authors") or [])],
           "pubtypes": p.get("publicationTypes"), "abstract": p.get("abstract") or "", "journal": p.get("journal"),
           "url": p.get("url"), "oa_pdf": (p.get("openAccessPdf") or {}).get("url")}
    if extra: rec.update(extra)
    return rec
recs = []
# IEEE Xplore
ie = json.load(open("final/ieee.json", encoding="utf-8"))
for r in ie["records"]:
    recs.append({"source": "IEEE Xplore", "sub": None, "title": re.sub(r"<[^>]+>", "", r.get("articleTitle", "")).strip(), "year": int(r.get("publicationYear") or 0) or None,
                 "venue": r.get("displayPublicationTitle") or r.get("publicationTitle"), "doi": nd(r.get("doi")), "arxiv": None, "s2id": None,
                 "authors": [a.get("preferredName") or a.get("normalizedName") for a in r.get("authors", [])], "pubtypes": [r.get("contentType")],
                 "abstract": re.sub(r"<[^>]+>", "", r.get("abstract") or ""), "url": "https://ieeexplore.ieee.org/document/" + str(r.get("articleNumber")),
                 "oa_pdf": None, "access": r.get("accessType", {}).get("type") if isinstance(r.get("accessType"), dict) else r.get("accessType")})
# Semantic Scholar
s2 = json.load(open("final/s2.json", encoding="utf-8"))
for p in s2["data"]: recs.append(s2_to_rec(p, "Semantic Scholar"))
# arXiv
ax = json.load(open("final/arxiv.json", encoding="utf-8"))
for r in ax["records"]:
    aid = re.sub(r"v\d+$", "", r["arxiv_id"])
    recs.append({"source": "arXiv", "sub": None, "title": r["title"], "year": int(r["published"][:4]), "venue": r.get("journal_ref") or "arXiv preprint",
                 "doi": nd(r.get("doi")), "arxiv": aid, "s2id": None, "authors": r["authors"], "pubtypes": ["Preprint"], "abstract": r["abstract"],
                 "url": "https://arxiv.org/abs/" + aid, "oa_pdf": "https://arxiv.org/pdf/" + aid})
# ScienceDirect (metadata via S2 batch by DOI)
meta = json.load(open("final/s2_by_doi.json", encoding="utf-8"))
cur = None
for line in open("final/sd_raw.txt", encoding="utf-8"):
    line = line.strip()
    if line.startswith("#"): cur = line.split()[0][1:]; continue
    if not line: continue
    doi, typ = line.split("~", 1); doi = doi.lower()
    p = meta.get("DOI:" + doi)
    if p: recs.append(s2_to_rec(p, "ScienceDirect", cur, {"doi": doi, "sd_type": typ}))
    else: recs.append({"source": "ScienceDirect", "sub": cur, "title": "", "year": None, "venue": None, "doi": doi, "arxiv": None, "s2id": None,
                       "authors": [], "pubtypes": [typ], "abstract": "", "url": "https://doi.org/" + doi, "oa_pdf": None, "sd_type": typ, "needs_meta": True})
# DBLP
mt = json.load(open("final/s2_by_title.json", encoding="utf-8"))
for line in open("final/dblp_raw.txt", encoding="utf-8"):
    if line.startswith("#") or not line.strip(): continue
    yr, typ, doi, venue, title = line.rstrip("\n").split("~", 4); doi = doi.lower() or None
    p = meta.get("DOI:" + doi) if doi else None
    if not p: p = mt.get(title.strip())
    base = {"source": "DBLP", "sub": None, "title": title.strip(), "year": int(yr), "venue": venue, "doi": doi, "dblp_type": typ}
    if p:
        r = s2_to_rec(p, "DBLP"); r.update({"title": title.strip().rstrip("."), "year": int(yr), "venue": venue, "doi": doi or r["doi"], "dblp_type": typ})
        if doi and doi.startswith("10.48550/arxiv."): r["arxiv"] = doi.split("arxiv.")[1]
        recs.append(r)
    else:
        base.update({"arxiv": doi.split("arxiv.")[1] if doi and doi.startswith("10.48550/arxiv.") else None, "s2id": None, "authors": [],
                     "pubtypes": [typ], "abstract": "", "url": ("https://doi.org/" + doi) if doi else None, "oa_pdf": None, "needs_meta": True})
        recs.append(base)
for i, r in enumerate(recs, 1):
    r["rec_id"] = f"R{i:04d}"
    if r.get("doi") and r["doi"].startswith("10.48550/arxiv."): r["arxiv"] = r.get("arxiv") or r["doi"].split("arxiv.")[1]
    if r.get("arxiv"): r["arxiv"] = re.sub(r"v\d+$", "", r["arxiv"].lower())
print(collections.Counter(r["source"] for r in recs), len(recs))
json.dump(recs, open("final/identification.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
