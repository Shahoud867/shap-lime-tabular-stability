"""Restore existing Excel filter controls omitted by the artifact exporter.
Only worksheet autoFilter XML is changed; cell data, styles and formulas are untouched.
"""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from xml.etree import ElementTree as E
import os
H=Path(__file__).resolve().parent;P=H.parents[1]/'outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8/Submission_Package'
f=P/'screening/MBR_A2_Screening_Workbook.xlsx'
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main';E.register_namespace('',NS)
refs={2:'A1:I16',3:'A1:F14',4:'A1:I17',5:'A1:H1550',6:'A1:F407',7:'A1:I1382',9:'A1:K195',10:'A1:O85',11:'A1:M85',12:'A1:Q85',14:'A1:I841'}
# Element ordering follows SpreadsheetML CT_Worksheet.
after={'mergeCells','phoneticPr','conditionalFormatting','dataValidations','hyperlinks','printOptions','pageMargins','pageSetup','headerFooter','rowBreaks','colBreaks','customProperties','cellWatches','ignoredErrors','smartTags','drawing','legacyDrawing','legacyDrawingHF','picture','oleObjects','controls','webPublishItems','tableParts','extLst'}
tmp=f.with_suffix('.tmp')
with ZipFile(f)as source,ZipFile(tmp,'w',ZIP_DEFLATED)as dest:
 for info in source.infolist():
  raw=source.read(info.filename)
  for n,ref in refs.items():
   if info.filename==f'xl/worksheets/sheet{n}.xml':
    root=E.fromstring(raw)
    for old in list(root.findall('{'+NS+'}autoFilter')):root.remove(old)
    auto=E.Element('{'+NS+'}autoFilter',{'ref':ref});at=next((i for i,ch in enumerate(root)if ch.tag.split('}')[-1]in after),len(root));root.insert(at,auto)
    raw=E.tostring(root,encoding='utf-8',xml_declaration=True)
  dest.writestr(info,raw)
os.replace(tmp,f)
print('Restored original filters and extended affected ranges to final row counts.')
