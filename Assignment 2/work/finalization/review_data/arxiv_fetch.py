import requests, json, sys, re, time, xml.etree.ElementTree as ET
from strings import I, O, P
strip = lambda t: t.replace("*", "")
f = lambda terms: "(" + " OR ".join(f"abs:{strip(t)}" for t in terms) + ")"
q = f"{f(I)} AND {f(O)} AND {f(P)} AND submittedDate:[201601010000 TO 202612312359]"
ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/", "x": "http://arxiv.org/schemas/atom"}
out, start, total = [], 0, None
while True:
    r = requests.get("https://export.arxiv.org/api/query", params={"search_query": q, "start": start, "max_results": 200}, timeout=120)
    root = ET.fromstring(r.text)
    total = int(root.find("o:totalResults", ns).text)
    ents = root.findall("a:entry", ns)
    for e in ents:
        out.append({"arxiv_id": e.find("a:id", ns).text.split("/abs/")[-1], "title": " ".join(e.find("a:title", ns).text.split()),
                    "abstract": " ".join(e.find("a:summary", ns).text.split()), "published": e.find("a:published", ns).text[:10],
                    "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
                    "doi": (e.find("x:doi", ns).text if e.find("x:doi", ns) is not None else None),
                    "journal_ref": (e.find("x:journal_ref", ns).text if e.find("x:journal_ref", ns) is not None else None)})
    start += len(ents)
    if not ents or start >= total: break
    time.sleep(3)
json.dump({"query": q, "total": total, "records": out}, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(total, len(out))
