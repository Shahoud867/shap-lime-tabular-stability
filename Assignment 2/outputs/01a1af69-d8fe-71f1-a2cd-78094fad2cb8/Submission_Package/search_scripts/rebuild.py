"""Regenerate derived deliverables from archived processed decisions, without network calls."""
from pathlib import Path
import argparse,shutil,subprocess,sys
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
src=Path(__file__).resolve().parent;out=args.output.resolve()
if out.exists()and any(out.iterdir()):raise SystemExit('Choose a new or empty output folder.')
stage=out/'review_data';(stage/'final').mkdir(parents=True);(stage/'extract').mkdir()
for f in(src/'data').iterdir():
 if f.is_file():shutil.copy2(f,stage/'final'/f.name)
for f in(src/'data/fulltext_decisions').glob('*.jsonl'):shutil.copy2(f,stage/'extract'/f.name)
(out/'report').mkdir();(out/'library').mkdir();(out/'links').mkdir()
for name,tail in [('build_master.py',[stage]),('coding.py',[stage]),('build_master.py',[stage]),('build_tex.py',[stage,out/'report']),('build_library.py',[stage,out])]:
 subprocess.run([sys.executable,str(src/name),*map(str,tail)],check=True,stdout=subprocess.DEVNULL)
print('Rebuilt master, counts, evidence coding, report tables, libraries and links in',out)
