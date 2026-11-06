import sys, re
txt_dir, uid, rx = sys.argv[1], sys.argv[2], sys.argv[3]
w = int(sys.argv[4]) if len(sys.argv) > 4 else 220; mx = int(sys.argv[5]) if len(sys.argv) > 5 else 4
flat = re.sub(r"\s+", " ", open(f"{txt_dir}/{uid}.txt", encoding="utf-8").read())
for i, m in enumerate(re.finditer(rx, flat, re.I)):
    if i >= mx: break
    pg = flat[:m.start()].count("<<PAGE")
    print(f"[{uid} p{pg}] ..{flat[max(0,m.start()-w):m.end()+w]}..")
