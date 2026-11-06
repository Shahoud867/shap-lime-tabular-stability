import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const here=path.dirname(fileURLToPath(import.meta.url));
const out=path.resolve(here,'../../outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8/Submission_Package/screening');
const p=JSON.parse(await fs.readFile(path.join(here,'workbook_payload.json'),'utf8'));
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(here,'screening_user_completed_backup.xlsx')));
await fs.mkdir(path.join(here,'previews'),{recursive:true});
async function preview(name,range,file){const b=await wb.render({sheetName:name,range,scale:1,format:'png'});await fs.writeFile(path.join(here,'previews',file+'.png'),new Uint8Array(await b.arrayBuffer()));}
await preview('Dual_Screening','A4:I10','before_dual');
if(process.argv.includes('--preview-only'))process.exit(0);
const ends={FullText:149,Included:79,Extraction:79};
for(const [name,s] of Object.entries(p.sheets)){
 const sh=s.headers?wb.worksheets.add(name):wb.worksheets.getItem(name);
 if(s.headers){sh.getRange('A1').write([s.headers]);sh.getRange('A1:I1').format={fill:'#18274D',font:{bold:true,color:'#FFFFFF'},rowHeight:32,wrapText:true};const widths=[9,10,7,45,9,90,65,50,50];for(let c=0;c<widths.length;c++)sh.getRangeByIndexes(0,c,s.rows.length+1,1).format.columnWidth=widths[c];sh.getRange(`A2:I${s.rows.length+1}`).format.wrapText=true;sh.getRange(`A2:I${s.rows.length+1}`).format.verticalAlignment='top';sh.getRange(`A2:I${s.rows.length+1}`).format.rowHeight=95;sh.freezePanes.freezeRows(1);}
 if(ends[name])for(let r=ends[name]+2;r<=s.rows.length+1;r++)sh.getRangeByIndexes(r-1,0,1,s.columns).copyFrom(sh.getRangeByIndexes(1,0,1,s.columns),'all');
 if(name==='Quality'){for(let r=81;r<=85;r++)sh.getRange(`A${r}:Q${r}`).copyFrom(sh.getRange('A2:Q2'),'all');sh.getRange('A2:Q105').clear({applyTo:'contents'});}
 sh.getRange(s.start).write(s.rows);
 if(name==='Quality')sh.getRange('A2:Q85').format.wrapText=true;
}
for(const [name,patch]of Object.entries(p.patches))for(const [addr,value]of Object.entries(patch))wb.worksheets.getItem(name).getRange(addr).values=[[value]];
const q=wb.worksheets.getItem('Quality'),e=wb.worksheets.getItem('Extraction');
for(let r=2;r<=85;r++){
 q.getRange(`M${r}:P${r}`).formulas=[[`=COUNT(C${r}:L${r})`,`=SUM(C${r}:L${r})`,`=ROUND(100*N${r}/M${r},1)`,`=IF(O${r}>=80,"High",IF(O${r}>=60,"Moderate","Low"))`]];
 e.getRange(`L${r}`).formulas=[[`=VLOOKUP(A${r},Quality!$A$2:$O$85,15,FALSE)`]];
}
wb.worksheets.getItem('Dual_Screening').getRange('A2:I3').format.rowHeight=40;
wb.worksheets.getItem('README').getRange('A29:A30').format.rowHeight=42;
for(const name of ['Included','Extraction'])wb.worksheets.getItem(name).getRange('A81:O85').format.rowHeight=150;
wb.worksheets.getItem('FullText').getRange('A151:K195').format.rowHeight=135;
wb.recalculate();
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:30},summary:'Final formula scan',maxChars:2500})).ndjson);
console.log('Dual summary',JSON.stringify(wb.worksheets.getItem('Dual_Screening').getRange('A4:F5').values));
for(const [name,range,file]of [['Dual_Screening','A4:F5','dual_summary'],['PRISMA_Counts','A24:C33','prisma'],['Included','A81:F85','included_new'],['Extraction','A81:F83','extraction_new'],['Quality','A1:P6','quality'],['FullText','F151:K154','fulltext_new'],['TA_Screening','G1:I5','ta'],['Quality_Evidence','C1:G5','quality_evidence']])await preview(name,range,file);
console.log(wb.help('worksheet.autoFilter',{include:'index,examples,notes',maxChars:2000}).ndjson);
await fs.mkdir(out,{recursive:true});
await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'MBR_A2_Screening_Workbook.xlsx'));
console.log('Exported',out);
