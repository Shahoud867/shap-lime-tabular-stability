"""Extract text from downloaded PDFs and print a reading digest (data only; nothing is executed)."""
import sys, os, re
from pypdf import PdfReader
pdf_dir, txt_dir, uid = sys.argv[1], sys.argv[2], sys.argv[3]
mode = sys.argv[4] if len(sys.argv) > 4 else "digest"
p = os.path.join(pdf_dir, uid + ".pdf"); t = os.path.join(txt_dir, uid + ".txt")
if not os.path.exists(t):
    pages = [(pg.extract_text() or '') for pg in PdfReader(p).pages]
    open(t, "w", encoding="utf-8").write("\n".join(f"<<PAGE {i+1}>>\n{x}" for i, x in enumerate(pages)))
txt = open(t, encoding="utf-8").read()
flat = re.sub(r"[ \t]+", " ", txt)
if mode == "full":
    print(flat); sys.exit()
m = re.search(r"\n\s*(references|bibliography)\s*\n", flat, re.I)
body = flat[: m.start()] if m else flat
print(f"### {uid} | chars={len(flat)} pages={flat.count('<<PAGE')}")
print("== HEAD ==\n" + body[:2600])
keys = r"(dataset|data set|datasets|experiment(al)? setup|we (use|used|evaluate|conduct)|perturb|noise|seed|bootstrap|jaccard|spearman|kendall|rank correlation|lipschitz|stability metric|stability index|wilcoxon|t-test|friedman|anova|significan)"
hits, last = [], -10000
for mm in re.finditer(keys, body, re.I):
    if mm.start() - last > 700:
        s = max(0, mm.start() - 200); hits.append(body[s: mm.start() + 420].replace("\n", " ")); last = mm.start()
print("== METHOD/RESULT SNIPPETS ==")
for h in hits[2:14]: print("..", h)
print("== TAIL (conclusion/limitations) ==\n" + body[-2600:])
for lab, rx in [("CODE/DATA", r"(github\.com|gitlab|code (is|are) (publicly )?available|available at|data availability|zenodo|figshare|osf\.io)"), ("ETHICS", r"(ethic|privacy|consent|GDPR|fairness)"), ("LIMITATION", r"(limitation|threats to validity|future work)")]:
    xs = [flat[max(0, mm.start()-150): mm.start()+250].replace("\n", " ") for mm in re.finditer(rx, flat, re.I)][:3]
    print(f"== {lab} ({len(xs)}) =="); [print("..", x) for x in xs]
