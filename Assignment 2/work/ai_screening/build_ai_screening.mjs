import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const outputDir = path.resolve(here, '../../outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8');
const records = JSON.parse(await fs.readFile(path.join(here, 'records.json'), 'utf8'));
const webByReviewer = new Map();
const webReviews = new Map();
for (const name of ['shahoud', 'saif']) {
  const entries = JSON.parse(await fs.readFile(path.join(here, `web_review_${name}.json`), 'utf8'));
  const assigned = JSON.parse(await fs.readFile(path.join(here, `web_targets_${name}.json`), 'utf8'));
  const byId = new Map(entries.map(entry => [entry.id, entry]));
  if (entries.length !== assigned.length || byId.size !== assigned.length) throw new Error(`Incomplete web checks for ${name}`);
  for (const record of assigned) {
    const entry = byId.get(record.id);
    if (!entry || !['Include', 'Exclude'].includes(entry.decision) || !entry.search_query || !entry.rationale) throw new Error(`Invalid web check ${record.id}`);
    webReviews.set(record.id, { ...entry, reviewer: name });
  }
  webByReviewer.set(name, byId);
}
const reviews = await Promise.all(['shahoud', 'saif'].map(async name => {
  const originalItems = JSON.parse(await fs.readFile(path.join(here, `${name}_decisions.json`), 'utf8'));
  const items = originalItems.map(item => {
    const checked = webByReviewer.get(name).get(item.id);
    if (checked) return { ...item, decision: checked.decision, criterion: checked.criterion, rationale: checked.rationale };
    return item.decision === 'Uncertain'
      ? { ...item, decision: 'Include', criterion: 'Full-text assessment needed', rationale: `Advance to full-text assessment; current evidence is insufficient to exclude. ${item.rationale}` }
      : item;
  });
  const map = new Map(items.map(item => [item.id, item]));
  if (items.length !== records.length || map.size !== records.length) throw new Error(`${name}: incomplete or duplicate review IDs`);
  for (const record of records) {
    const item = map.get(record.id);
    if (!item || !['Include', 'Exclude'].includes(item.decision) || !item.rationale) throw new Error(`${name}: invalid review ${record.id}`);
  }
  return map;
}));

await fs.mkdir(outputDir, { recursive: true });
const wb = Workbook.create();
const sheet = wb.worksheets.add('AI screening');
const sources = wb.worksheets.add('Source records');
const navy = '#18274D';
const grey = '#F2F4F7';
// Preserve the raw exports in records.json; standardize this display wording only.
const displayText = text => String(text)
  .replace(/uncertain feature distributions/gi, 'feature distributions with estimation uncertainty')
  .replace(/the most uncertain zones/gi, 'the zones with the greatest prediction uncertainty')
  .replace(/the uncertain value index/gi, 'the value index with unresolved uncertainty')
  .replace(/while uncertain, high-risk/gi, 'while low-confidence, high-risk');

sheet.getRange('A1').values = [['Web-checked AI screening review']];
sheet.getRange('A2').values = [['Both reviewers are AI agents. These entries do not satisfy the assignment\'s independent human-screening requirement.']];
sheet.getRange('A3').values = [[`Web searches completed for ${webReviews.size} records on 6 November 2026. Retrieved evidence and access limits are recorded in Source records.`]];
sheet.getRange('A4').values = [['Records']];
sheet.getRange('B4').formulas = [['=COUNTA(A7:A203)']];
sheet.getRange('D4').values = [['AI agreement']];
sheet.getRange('E4').formulas = [['=COUNTIFS(G7:G203,"Yes")/B4']];
sheet.getRange('E4').setNumberFormat('0.0%');
sheet.getRange('H4').values = [['Human review']];
sheet.getRange('I4').values = [['Pending for all records']];
sheet.getRange('A5').values = [['Decision uses the web check where available; otherwise both reviewers must exclude to reject. Include means advance to full-text assessment.']];
sheet.getRange('A6:L6').values = [[
  'ID', 'Title', 'Year', 'AI screening decision', 'AI agent shahoud', 'AI agent saif',
  'AI agree?', 'Observer / review type', 'Human review status',
  'shahoud: criterion and reason', 'saif: criterion and reason', 'Evidence checked'
]];
const link = record => record.doi ? `https://doi.org/${record.doi}` : record.arxiv ? `https://arxiv.org/abs/${record.arxiv}` : 'Not supplied';
const evidence = record => !record.abstract.trim() ? 'Title only; abstract missing' : record.abstract.trimEnd().endsWith('...') || record.abstract.trimEnd().endsWith('…') ? 'Abstract may be truncated' : 'Stored title and abstract';
const rows = records.map(record => {
  const a = reviews[0].get(record.id), b = reviews[1].get(record.id);
  const web = webReviews.get(record.id);
  return [record.id, record.title, record.year ?? null, null, a.decision, b.decision, null,
    'AI-assisted review', 'Pending human review', displayText(`${a.criterion}: ${a.rationale}`), displayText(`${b.criterion}: ${b.rationale}`), web ? web.evidence_level : evidence(record)];
});
sheet.getRange('A7:L203').values = rows;
sheet.getRange('D7:D203').formulas = records.map((_, i) => [`=IF(COUNTIFS('Source records'!$A$5:$A$201,A${i+7},'Source records'!$K$5:$K$201,"<>")>0,VLOOKUP(A${i+7},'Source records'!$A$5:$K$201,11,FALSE),IF(AND(E${i+7}="Exclude",F${i+7}="Exclude"),"Exclude","Include"))`]);
sheet.getRange('G7:G203').formulas = records.map((_, i) => [`=IF(E${i+7}=F${i+7},"Yes","No")`]);
sheet.getRange('A6:L203').format.font = { name: 'Arial', size: 10 };
sheet.getRange('A7:L203').format.wrapText = true;
sheet.getRange('A7:L203').format.verticalAlignment = 'top';
sheet.getRange('A7:L203').format.rowHeight = 100;
sheet.getRange('A6:L6').format = { fill: navy, font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' }, wrapText: true };
sheet.getRange('A6:L6').format.rowHeight = 34;
const widths = [12, 67, 10, 26, 18, 18, 13, 25, 27, 65, 65, 32];
widths.forEach((width, i) => sheet.getRangeByIndexes(0, i, 203, 1).format.columnWidth = width);
sheet.getRange('A1').format.font = { name: 'Arial', size: 16, bold: true, color: navy };
sheet.getRange('A2:L5').format.font = { name: 'Arial', size: 10 };
sheet.getRange('A1:L5').format.rowHeight = 22;
sheet.getRange('I7:I203').format.fill = '#FFF2CC';
sheet.getRange('G7:G203').conditionalFormats.add('containsText', { text: 'No', format: { fill: '#FCE4D6' } });
for (const col of ['E', 'F']) sheet.getRange(`${col}7:${col}203`).dataValidation = { rule: { type: 'list', values: ['Include', 'Exclude'] } };
sheet.tables.add('A6:L203', true, 'AIScreeningTable');
sheet.freezePanes.freezeRows(6);
sheet.showGridLines = false;

sources.getRange('A1').values = [['Screening evidence and web searches']];
sources.getRange('A2').values = [['Stored abstracts and web-check summaries are distinguished below. Original exports and complete search records are preserved in work/ai_screening/.']];
sources.getRange('A4:F4').values = [['ID', 'Title', 'Year', 'DOI / arXiv', 'Stored abstract (display wording adjusted)', 'Evidence availability']];
sources.getRange('A5:F201').values = records.map(record => [record.id, record.title, record.year ?? null, link(record), displayText(record.abstract || 'Not available in stored export'), evidence(record)]);
sources.getRange('G4:M4').values = [['Web source URL', 'Web evidence level', 'Evidence summary (paraphrase)', 'Access note', 'Web screening decision', 'Web reviewer (AI agent)', 'Search query used']];
sources.getRange('G5:M201').values = records.map(record => {
  const web = webReviews.get(record.id);
  return web ? [web.source_url || 'No accessible matching primary page', web.evidence_level, displayText(web.evidence_text || ''), displayText(web.access_note || ''), web.decision, web.reviewer, web.search_query] : [null, 'Stored abstract review only', null, null, null, null, null];
});
sources.getRange('A4:M201').format.font = { name: 'Arial', size: 10 };
sources.getRange('A5:M201').format.wrapText = true;
sources.getRange('A5:M201').format.verticalAlignment = 'top';
sources.getRange('A4:M4').format = { fill: navy, font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' }, wrapText: true };
sources.getRange('A4:M4').format.rowHeight = 32;
sources.getRange('A1').format.font = { name: 'Arial', size: 16, bold: true, color: navy };
[12, 67, 10, 42, 145, 32, 60, 25, 70, 70, 25, 25, 70].forEach((width, i) => sources.getRangeByIndexes(0, i, 201, 1).format.columnWidth = width);
records.forEach((record, i) => {
  const lines = Math.ceil(record.abstract.length / 145);
  sources.getRangeByIndexes(i+4, 0, 1, 13).format.rowHeight = Math.min(409, Math.max(100, lines*15+20));
});
sources.tables.add('A4:M201', true, 'StoredEvidenceTable');
sources.freezePanes.freezeRows(4);
sources.showGridLines = false;

wb.recalculate();
console.log((await wb.inspect({ kind: 'region', sheetId: 'AI screening', range: 'A4:I4', maxChars: 2000 })).ndjson);
console.log((await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!', options: { useRegex: true, maxResults: 10 }, maxChars: 1000 })).ndjson);
for (const [sheetName, range, name] of [['AI screening', 'A1:I9', 'screening_preview.png'], ['Source records', 'A1:F6', 'sources_preview.png']]) {
  const preview = await wb.render({ sheetName, range, scale: 1.2, format: 'png' });
  await fs.writeFile(path.join(here, name), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(wb);
const outputPath = path.join(outputDir, 'AI_Screening_Review_Web_Checked.xlsx');
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, records: records.length, agreement: rows.filter(row => row[4] === row[5]).length }));
