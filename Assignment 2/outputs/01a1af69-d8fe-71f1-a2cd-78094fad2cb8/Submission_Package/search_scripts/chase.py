import json, re, time, requests
seeds = {"U291 Ballegeer2025": "DOI:10.1016/j.ejor.2025.05.039", "U498 Agarwal2022": "ARXIV:2203.06877", "U155 Visani2022": "DOI:10.1080/01605682.2020.1865846",
         "U483 Zhou2021": "DOI:10.1145/3447548.3467274", "U141 Tzionis2026": "DOI:10.1080/08839514.2026.2640686", "U163 Rajhans2026": "ARXIV:2607.01679", "U159 Pala2024": "ARXIV:2402.19025"}
F = "title,year,externalIds,venue,abstract,publicationTypes,authors"
def get(url, params):
    for a in range(8):
        r = requests.get(url, params=params, timeout=90)
        if r.status_code == 200: return r.json()
        time.sleep(5 * (a + 1))
    return {"data": []}
out = {}
for name, pid in seeds.items():
    for direction in ("references", "citations"):
        d = get(f"https://api.semanticscholar.org/graph/v1/paper/{pid}/{direction}", {"fields": F, "limit": 1000})
        key = "citedPaper" if direction == "references" else "citingPaper"
        lst = [x[key] for x in d.get("data", []) if x.get(key) and x[key].get("title")]
        out[f"{name}|{direction}"] = lst
        print(name, direction, len(lst), flush=True)
        time.sleep(2.5)
json.dump(out, open("final/chase_raw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
