import requests, json, sys
from strings import ieee
q = ieee()
s = requests.Session()
h = {"Content-Type": "application/json", "Accept": "application/json, text/plain, */*", "Origin": "https://ieeexplore.ieee.org",
     "Referer": "https://ieeexplore.ieee.org/search/searchresult.jsp", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
s.get("https://ieeexplore.ieee.org/Xplore/home.jsp", headers={"User-Agent": h["User-Agent"]}, timeout=60)
body = {"action": "search", "newsearch": True, "matchBoolean": True, "queryText": q, "highlight": False, "returnType": "SEARCH",
        "matchPubs": True, "rowsPerPage": 100, "pageNumber": 1, "ranges": ["2016_2026_Year"]}
r = s.post("https://ieeexplore.ieee.org/rest/search", headers=h, data=json.dumps(body), timeout=90)
print(r.status_code)
d = r.json(); print(d.get("totalRecords"), len(d.get("records", [])))
json.dump({"query": q, "filters": "Year 2016-2026", "total": d.get("totalRecords"), "records": d.get("records", [])}, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
