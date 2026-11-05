"""Run the supplementary search in IEEE Xplore, arXiv and Semantic Scholar and save raw results (data only)."""
import sys, json, time, requests, datetime
import xml.etree.ElementTree as ET
import supp_strings as ss
which = sys.argv[1]
stamp = datetime.datetime.now().isoformat(timespec='minutes')
if which == 'ieee':
    q = ss.ieee(); s = requests.Session()
    h = {"Content-Type": "application/json", "Accept": "application/json, text/plain, */*", "Origin": "https://ieeexplore.ieee.org",
         "Referer": "https://ieeexplore.ieee.org/search/searchresult.jsp", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
    s.get("https://ieeexplore.ieee.org/Xplore/home.jsp", headers={"User-Agent": h["User-Agent"]}, timeout=60)
    out, page, total = [], 1, None
    while True:
        body = {"action": "search", "newsearch": True, "matchBoolean": True, "queryText": q, "highlight": False, "returnType": "SEARCH",
                "matchPubs": True, "rowsPerPage": 100, "pageNumber": page, "ranges": ["2016_2026_Year"]}
        r = s.post("https://ieeexplore.ieee.org/rest/search", headers=h, data=json.dumps(body), timeout=90)
        d = r.json(); total = d.get("totalRecords"); out += d.get("records", [])
        if len(out) >= (total or 0) or not d.get("records"): break
        page += 1; time.sleep(2)
    json.dump({"query": q, "filters": "Year 2016-2026", "total": total, "records": out, "run": stamp}, open("final/supp_ieee.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print('IEEE', total, len(out))
elif which == 'arxiv':
    q = ss.arxiv(); ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/", "x": "http://arxiv.org/schemas/atom"}
    out, start = [], 0
    while True:
        r = requests.get("https://export.arxiv.org/api/query", params={"search_query": q, "start": start, "max_results": 200}, timeout=120)
        root = ET.fromstring(r.text); total = int(root.find("o:totalResults", ns).text); ents = root.findall("a:entry", ns)
        for e in ents:
            out.append({"arxiv_id": e.find("a:id", ns).text.split("/abs/")[-1], "title": " ".join(e.find("a:title", ns).text.split()),
                        "abstract": " ".join(e.find("a:summary", ns).text.split()), "published": e.find("a:published", ns).text[:10],
                        "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
                        "doi": (e.find("x:doi", ns).text if e.find("x:doi", ns) is not None else None),
                        "journal_ref": (e.find("x:journal_ref", ns).text if e.find("x:journal_ref", ns) is not None else None)})
        start += len(ents)
        if not ents or start >= total: break
        time.sleep(3.2)
    json.dump({"query": q, "total": total, "records": out, "run": stamp}, open("final/supp_arxiv.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print('arXiv', total, len(out))
elif which == 's2':
    from s2_fetch import fetch
    q = ss.s2(); total, recs = fetch(q)
    json.dump({"query": q, "total": total, "data": recs, "run": stamp}, open("final/supp_s2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print('S2', total, len(recs))
