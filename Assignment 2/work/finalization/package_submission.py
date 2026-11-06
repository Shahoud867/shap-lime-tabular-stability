from pathlib import Path
import shutil,json,hashlib,zipfile
H=Path(__file__).resolve().parent;O=H.parents[1]/'outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8';P=O/'Submission_Package'
for name in ['qa_positive_notes.json','qa_positive_notes_sources.json']:
 shutil.copy2(H/name,P/'search_scripts/data'/name)
shutil.copy2(H/'validation_results.json',P/'validation_results.json')
assert json.loads((P/'validation_results.json').read_text())['all_pass']
for p in list(P.rglob('*')):
 if p.is_file()and(p.name=='.DS_Store'or p.suffix in['.log','.blg','.pyc']or p.name.endswith('.inspect.ndjson')):p.unlink()
for p in list(P.rglob('__pycache__')):shutil.rmtree(p)
manifest=P/'SHA256SUMS.txt'
files=sorted(p for p in P.rglob('*')if p.is_file()and p!=manifest)
manifest.write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(P).as_posix()for p in files)+'\n')
z=O/'MG3010_Assignment2_23i2515_23i2634.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=9)as f:
 for p in sorted(P.rglob('*')):
  if p.is_file():f.write(p,Path('Submission_Package')/p.relative_to(P))
with zipfile.ZipFile(z)as f:
 assert f.testzip()is None
 assert len(f.namelist())==len(files)+1
print(json.dumps({'zip':str(z),'files':len(files)+1,'bytes':z.stat().st_size,'sha256':hashlib.sha256(z.read_bytes()).hexdigest()},indent=2))
