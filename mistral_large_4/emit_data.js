#!/usr/bin/env node
/* ml4_data.json + the figure reads -> figread.json + data.js (window.ML4).

   Every figure reads its numbers from here. A scene that hardcodes a score is a
   defect you can grep for, which is the entire point of keeping this step.
   Node port of emit_data_js.py, so it runs on the login node.

   Figure reads come from ../../ml4_work:
     fig6 fig13 fig15 fig17   vector figures, exact (figvec.js, fig13.js)
     figread/fig*.json        raster figures, read from a 600 dpi render, approximate */
const fs = require('fs'), path = require('path');
const HERE = __dirname, WORK = path.resolve(HERE, '../../ml4_work');
const js = f => JSON.parse(fs.readFileSync(f, 'utf8'));

const data = js(path.join(HERE, 'ml4_data.json'));

const fr = {
  exact: ['fig6', 'fig13', 'fig15', 'fig17'],
  fig6: js(path.join(WORK, 'fig6.json')),
  fig13: js(path.join(WORK, 'fig13.json')),
  fig15: js(path.join(WORK, 'fig15.json')),
  fig17: js(path.join(WORK, 'fig17.json')),
};
const rasters = [];
for (const n of [2, 4, 5, 7, 8, 11, 14, 16]) {
  const f = path.join(WORK, 'figread', `fig${n}.json`);
  if (fs.existsSync(f)) { fr['fig' + n] = js(f); rasters.push('fig' + n); }
}
fr.approx = rasters;
data.figread = fr;
fs.writeFileSync(path.join(HERE, 'figread.json'), JSON.stringify(fr, null, 1) + '\n');

// Table rows carry the score source in brackets. Split it out once, here.
const LABEL = {
  'AA-Briefcase (AA)': ['AA-Briefcase', 'AA'],
  'AutomationBench (AA)': ['AutomationBench', 'AA'],
  'GDP.pdf (AA)': ['GDP.pdf', 'AA'],
  'BrowseComp subset (internal)': ['BrowseComp subset', 'internal'],
  'HLE-Diamond (AA)': ['HLE-Diamond', 'AA'],
  'CritPt v2 (Hard Subset, AA)': ['CritPt v2, hard subset', 'AA'],
  'AA-Omniscience v1.1 Accuracy (AA)': ['AA-Omniscience accuracy', 'AA'],
  'AA-Omniscience v1.1 Non-Hallucination Rate (AA)': ['AA-Omniscience non-hallucination', 'AA'],
  'ArXivMath 03/26 to 05/26 (internal)': ['ArXivMath, Mar to May 2026', 'internal'],
  'ArXivMath 06/26 to 09/26 (internal)': ['ArXivMath, Jun to Sep 2026', 'internal'],
  'Beyond-AIME (internal)': ['Beyond-AIME', 'internal'],
  'AA Coding Agent Index v2 (AA)': ['AA Coding Agent Index v2', 'AA'],
  'FrontierCode 2.0 (Cognition)': ['FrontierCode 2.0', 'Cognition'],
  'Terminal-Bench 2.1': ['Terminal-Bench 2.1', ''],
  'Terminal-Bench 4.0 (AA Coding Agent Index)': ['Terminal-Bench 4.0', 'AA'],
  'DeepSWE 1.1': ['DeepSWE 1.1', ''],
  'AA Cyber Index (AA)': ['AA Cyber Index', 'AA'],
  'CWE-Bench-AA (AA)': ['CWE-Bench-AA', 'AA'],
  'DeepsecBench-AA (AA)': ['DeepsecBench-AA', 'AA'],
  'CyberGym-E2E-AA (AA)': ['CyberGym-E2E-AA', 'AA'],
  'Cybench (internal)': ['Cybench', 'internal'],
  'KORA Benchmark': ['KORA (0 to 2 scale)', 'internal'],
  'B3 agentic security benchmark': ['B3 agentic security', 'internal'],
  'JailbreakBench (cyber subset): safe-response rate': ['JailbreakBench, cyber: safe responses', 'internal'],
  'StrongREJECT (cyber subset): refusal rate': ['StrongREJECT, cyber: refusals', 'internal'],
  'AgentHarm (cyber subset): 100 minus harm score': ['AgentHarm, cyber: 100 minus harm', 'internal'],
  'Vals Finance Agent v2.1': ['Vals Finance Agent v2.1', 'Vals'],
  'Finch (internal)': ['Finch', 'internal'],
  'HLAB-v1.1 (Vals)': ['Harvey Legal Agent Bench v1.1', 'Vals'],
};
data.short = ['GLM-5.2', 'GLM-5.3', 'DeepSeek V4-Pro', 'DeepSeek V4.1-Flash', 'Kimi K3',
  'Muse Spark Open', 'Mistral Large 4'];

const ml4 = data.ml4;
let top = 0, topOut = 0, rowsAll = 0;
const names = new Set();
for (const t of data.tables) for (const r of t.rows) {
  if (!LABEL[r.name]) throw new Error('no label for row ' + r.name);
  names.add(r.name);
  [r.label, r.src] = LABEL[r.name];
  const rep = r.vals.filter(v => v !== null);
  // competition ranking: ties share a rank, the next rank skips
  r.rank = r.vals.map(v => v === null ? null : 1 + rep.filter(w => w > v).length);
  r.n = rep.length;
  r.tied = r.rank[ml4] === 1 && rep.filter(v => v === r.vals[ml4]).length > 1;
  rowsAll++;
  if (r.rank[ml4] === 1) { top++; if (!r.tied) topOut++; }
}
if (names.size !== Object.keys(LABEL).length) throw new Error('label map and table rows differ');
data.summary = { rows: rowsAll, ml4_top: top, ml4_top_outright: topOut };

// Extra prose facts, one file per scene writer: ml4_work/facts_extra/<part>.json holds
// {"key": {"v": 12, "quote": "verbatim sentence from the paper"}}. Each quote must occur
// in the paper text and contain its value, or the build stops.
const PAPER_TXT = fs.readFileSync(path.join(WORK, 'ref', 'ml4_v3.txt'), 'utf8')
  .split('\n').filter(l => !l.startsWith('=====')).join(' ')
  .replace(/(\w)- (\w)/g, '$1$2').replace(/\s+/g, ' ');
const squash = s => String(s).replace(/(\w)- (\w)/g, '$1$2').replace(/\s+/g, ' ').trim();
const EXTRA = path.join(WORK, 'facts_extra');
let nExtra = 0;
if (fs.existsSync(EXTRA)) {
  for (const fn of fs.readdirSync(EXTRA).filter(n => n.endsWith('.json')).sort()) {
    const ex = js(path.join(EXTRA, fn));
    for (const [k, e] of Object.entries(ex)) {
      const q = squash(e.quote), val = String(e.v).replace(/,/g, '');
      if (k in data.facts) throw new Error(`${fn}: fact ${k} already exists`);
      if (!PAPER_TXT.includes(q)) throw new Error(`${fn}: quote for ${k} is not in the paper: ${q}`);
      if (!q.replace(/,/g, '').includes(val)) throw new Error(`${fn}: value ${val} of ${k} is not in its quote`);
      data.facts[k] = e.v;
      nExtra++;
    }
  }
}
console.log(`  extra facts: ${nExtra}, each quote found verbatim in the paper`);

const f = data.facts;
const trio = data.tables[4].rows.slice(2, 5).map(r => r.vals[ml4]);
data.derived = {
  safety_mean: Math.round(trio.reduce((a, b) => a + b, 0) / 3 * 100) / 100,
  f17_saved_us: f['f17.before'] - f['f17.after'],
  kv_frac: 1 / f['kv.reduction'],
};
console.log(`  figure reads: exact ${fr.exact.join(' ')}; approximate ${rasters.join(' ') || 'none'}`);
console.log(`  ML4 ranks first on ${top} of ${rowsAll} rows (${topOut} outright)`);
for (const t of data.tables) for (const r of t.rows)
  console.log(`   T${t.n} ${r.label.slice(0, 40).padEnd(40)} rank ${r.rank[ml4]} of ${r.n}` + (r.tied ? '  tied' : ''));
console.log('  derived', JSON.stringify(data.derived));

// the page carries no dash characters at all: a missing table entry becomes 'n/a',
// and a range like '0\u20132' or '03/26\u201305/26' reads 'to'
const out = 'window.ML4 = ' + JSON.stringify(data).replace(/"\u2014"/g, '"n/a"')
  .replace(/\s*\u2013\s*/g, ' to ').replace(/\s*\u2014\s*/g, ', ') + ';\n';
// several writers may run this at once; a rename never leaves a half-written file
const tmp = path.join(HERE, `.data.js.${process.pid}`);
fs.writeFileSync(tmp, out);
fs.renameSync(tmp, path.join(HERE, 'data.js'));
console.log(`wrote data.js (${out.length.toLocaleString()} bytes)`);
