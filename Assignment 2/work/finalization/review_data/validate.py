import sys, json, re
KNOWN = {
 "Ballegeer2025": "evaluating the stability of model explanations in instance-dependent cost-sensitive credit scoring",
 "Visani2022": "statistical stability indices for lime",
 "AlvarezMelis2018": "on the robustness of interpretability methods",
 "Agarwal2022": "rethinking stability for attribution-based explanations",
 "Zhou2021": "s-lime",
 "Tzionis2026": "improving lime stability via density",
 "Slack2020": "fooling lime and shap",
 "Hwang2025": "shap-based explanations are sensitive to feature representation",
 "Lin2023": "on the robustness of removal-based feature attributions",
 "Rajhans2026": "beyond gradient-based attacks",
 "Vaduva2026": "measuring the fragility of trust",
 "Pala2024": "combination of weak learners explanations",
}
if __name__ == "__main__":
  pass
norm = lambda s: re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
def _main():
  recs = json.load(open(sys.argv[1], encoding="utf-8"))
  titles = [norm(r["title"]) for r in recs]
  for k, t in KNOWN.items():
      tt = norm(t)
      hit = any(" ".join(tt.split()) in " ".join(x.split()) for x in titles)
      print(f"{k:18s} {'HIT' if hit else '-'}")
  
if __name__ == "__main__": _main()
