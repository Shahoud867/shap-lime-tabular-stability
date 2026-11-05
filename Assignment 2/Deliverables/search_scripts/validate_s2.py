import sys, json, re
from validate import KNOWN
norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split())
d = json.load(open(sys.argv[1], encoding="utf-8")); titles = [norm(r["title"]) for r in d["data"]]
print(" ".join(f"{k}:{'HIT' if any(norm(t) in x for x in titles) else '-'}" for k, t in KNOWN.items()))
