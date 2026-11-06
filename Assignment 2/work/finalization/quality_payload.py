from pathlib import Path
import json,re,csv,importlib.util
H=Path(__file__).resolve().parent;P=H.parents[1]/'outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8/Submission_Package'
spec=importlib.util.spec_from_file_location('dm',H/'review_data/data_model.py');dm=importlib.util.module_from_spec(spec);spec.loader.exec_module(dm)
m=json.loads((H/'review_data/final/master_included.json').read_text());p=json.loads((H/'workbook_payload.json').read_text())
headers=['Study ID','Record ID','Item','Criterion','Score','Evidence / appraisal basis','Source locator','Version inspected','Stable link']
rows=[];fields={1:'problem',2:'method',3:'data',4:'evaluation',5:'evaluation',6:'findings',7:'limitations',8:'repro'}
for r in m:
 for i,(q,criterion)in enumerate(dm.QA_ITEMS,1):
  score=r['qa'][i-1];notes=r.get('qa_notes','')
  specific=[s.strip()for s in notes.split(';')if re.search(r'Q'+str(i)+r'(?!\d)',s)]
  if specific:note='; '.join(specific)
  elif i in fields:note=f"Extracted {fields[i]} supporting the retained appraisal: "+r.get(fields[i],'Not reported')
  elif i==9 and score=='NA':note='Not applicable in the original appraisal; dataset context: '+r.get('data','Not reported')
  elif i==9:note='Ethics/privacy context in the original appraisal: '+notes
  else:note='Randomness/repetition judgment retained from source-linked appraisal: '+notes
  rows.append([r['sid'],r['uid'],q,criterion,'N/A'if score=='NA'else score,note,r['evidence'],r['read_from'],r.get('fulltext_url')or('https://doi.org/'+r['doi']if r.get('doi')else 'https://arxiv.org/abs/'+r['arxiv']if r.get('arxiv')else '')])
p['sheets']['Quality_Evidence']={'start':'A2','columns':9,'rows':rows,'headers':headers}
(H/'workbook_payload.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
with(P/'screening/quality_item_evidence.csv').open('w',newline='',encoding='utf-8-sig')as f: w=csv.writer(f);w.writerow(headers);w.writerows(rows)
print(len(rows),'quality item evidence records prepared')
