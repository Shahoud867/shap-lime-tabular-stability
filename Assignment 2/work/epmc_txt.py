"""Fetch Europe PMC open-access full-text XML and save plain text (data only)."""
import requests, sys, re, os
uid, pmcid, out = sys.argv[1], sys.argv[2], sys.argv[3]
r = requests.get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML', timeout=90)
x = r.text
x = re.sub(r'<(title|sec-title)>', '\n## ', x)
t = re.sub(r'<[^>]+>', ' ', x)
t = re.sub(r'[ \t]+', ' ', t)
open(os.path.join(out, uid + '.txt'), 'w', encoding='utf-8').write('<<PAGE 1>>\n' + t)
print(uid, r.status_code, len(t))
