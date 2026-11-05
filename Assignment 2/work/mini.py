import sys, re
txt_dir, uids = sys.argv[1], sys.argv[2].split(",")
def win(flat, rx, w, mx, gap=500):
    out, last = [], -10**9
    for m in re.finditer(rx, flat, re.I):
        if m.start() - last < gap: continue
        last = m.start(); pg = flat[:m.start()].count("<<PAGE")
        out.append(f"[p{pg}] " + flat[max(0, m.start()-w): m.end()+w])
        if len(out) >= mx: break
    return out
for u in uids:
    flat = re.sub(r"\s+", " ", open(f"{txt_dir}/{u}.txt", encoding="utf-8").read())
    m = re.search(r"\b(references|bibliography)\b(?!.{0,40}\bet al)", flat[len(flat)//2:], re.I)
    body = flat[: len(flat)//2 + m.start()] if m else flat
    a = re.search(r"abstract", body[:6000], re.I)
    s = a.start() if a else 0
    print(f"\n########## {u} | pages={flat.count('<<PAGE')}\nTITLE/ABS: " + body[max(0, s-350): s+1900])
    for lab, rx, w, mx in [("DATA", r"(dataset|data set)s? (is|are|was|were|consists|contains|comprising|from)|samples|instances|records", 160, 3),
                           ("SETUP", r"(we (train|use|employ|evaluate|perturb|generate|repeat|compute))|random seeds?|repetitions|iterations|noise (level|with)", 150, 3),
                           ("METRIC", r"jaccard|spearman|kendall|rank correlation|lipschitz|cosine similarity|coefficient of variation|stability (index|score|metric)", 150, 2),
                           ("STATS", r"wilcoxon|friedman|t-test|anova|mann-whitney|p\s?<|confidence interval", 120, 2),
                           ("CODE", r"github\.com|zenodo|figshare|code (is|are|and data)? ?(publicly )?available|data availability", 110, 2),
                           ("LIMIT", r"limitation|threats to validity|future (work|research)", 220, 2)]:
        for x in win(body if lab != "CODE" else flat, rx, w, mx): print(f"  <{lab}> {x}")
    tail = body[-1500:]
    print("  <TAIL> " + tail)
