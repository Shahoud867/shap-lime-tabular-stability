import json, sys, os
S = sys.argv[1]
pay = "No open-access version found (Unpaywall/Semantic Scholar not OA; publisher returned an access or sign-in page; no arXiv version)"
blk = "Open access, but automated access was stopped by the publisher's bot verification or CAPTCHA, which was not bypassed"
groups = {
 pay: "U001 U005 U009 U012 U019 U028 U029 U030 U031 U032 U041 U334 U351 U506 U579 C009 C018 U081 U255 U444 U581 C037 C067".split(),
 "No full-text location found (conference paper without DOI or link in the indexes searched)": "U534 U587 C054".split(),
 blk: "U055 U396 U397 U404 U406 U408 U416 U438 U453 U454 U121 U277".split(),
 "Repository copy exists but the server refused automated download (HTTP 403/202 challenge)": "U509 U346".split(),
 "Full text offered only as a file download from a script-driven page; not downloaded in this session": "U308".split(),
 "Publisher page did not render its content in the browser": "U098".split(),
}
with open(os.path.join(S, 'extract', 'ft_notretrieved.jsonl'), 'w', encoding='utf8') as f:
    for reason, ids in groups.items():
        for u in ids:
            f.write(json.dumps({'uid': u, 'decision': 'Not retrieved', 'reason': reason}) + '\n')
print(sum(len(v) for v in groups.values()))
