#!/usr/bin/env node
/* Assemble the Mistral Large 4 visual guide into one self-contained index.html.

     tokens.css   design system, shared with the Shieldstral and Robostral guides
     beats.html   the numbered beats, text only
     kit.js       the 2D SVG drawing vocabulary
     engine.js    scroll engine and scene registry
     scenes/*.js  the figures
     data.js      window.ML4, generated from the paper by parse_ml4.py + emit_data_js.py

   Written in node so it runs anywhere, including the login node.
     node build.js                                   -> index.html
     BEATS=_check/a2.beats OUT=_check/a2.html node build.js   -> a partial page for checks */
const fs = require('fs'), path = require('path');
const HERE = __dirname;
const rd = f => fs.readFileSync(path.join(HERE, f), 'utf8');

const tokens = rd('tokens.css');
const beats = fs.readFileSync(process.env.BEATS ? path.resolve(HERE, process.env.BEATS) : path.join(HERE, 'beats.html'), 'utf8');
const engine = rd('engine.js'), kit = rd('kit.js'), data = rd('data.js');
const extra = fs.existsSync(path.join(HERE, 'scene.css')) ? rd('scene.css') : '';
const ML4 = JSON.parse(data.replace(/^window\.ML4 = /, '').replace(/;\s*$/, ''));
const F = ML4.facts;

// SCENES_ONLY=a4_01,a4_02 builds a check page with just those scene files (plus 00_chart),
// so one half-written scene cannot break another writer's check
const only = process.env.SCENES_ONLY ? ['00_chart', ...process.env.SCENES_ONLY.split(',')] : null;
const scenes = [], files = [];
for (const f of fs.readdirSync(path.join(HERE, 'scenes')).filter(f => f.endsWith('.js')).sort()) {
  if (only && !only.some(p => f.startsWith(p))) continue;
  scenes.push(`/* ---- ${f.replace(/\.js$/, '')} ---- */\n` + rd('scenes/' + f).trim());
  files.push(f.replace(/\.js$/, ''));
}
// SCENES_EXTRA=dir adds every scene file in dir, for checking a draft outside scenes/
if (process.env.SCENES_EXTRA) {
  const xd = path.resolve(HERE, process.env.SCENES_EXTRA);
  for (const f of fs.readdirSync(xd).filter(f => f.endsWith('.js')).sort()) {
    scenes.push(`/* ---- extra ${f.replace(/\.js$/, '')} ---- */\n` + fs.readFileSync(path.join(xd, f), 'utf8').trim());
    files.push('extra:' + f.replace(/\.js$/, ''));
  }
}
const js = scenes.join('\n');
const wanted = [...beats.matchAll(/data-s="(\w+)"/g)].map(m => m[1]);
const have = new Set([...js.matchAll(/SCENES\[['"](\w+)['"]\]/g), ...js.matchAll(/SCENES\.(\w+)\s*=/g)].map(m => m[1]));
const missing = wanted.filter(w => !have.has(w));

const TITLE = 'Mistral Large 4, taken apart';
const DESC = 'A trillion-parameter mixture of experts, built for agentic work, whose attention was ' +
  'operated on in the middle of training. A scrollable walk through the technical report, with ' +
  'figures you can click.';
const URL = 'https://avisoori1x.github.io/mistral_large_4/';
const PAPER = 'https://www.alphaxiv.org/pdf/2611.mistral-large-4';

const HERO = `
<div class="hero">
  <div class="eyebrow">a visual guide &nbsp;&middot;&nbsp; paper results, live figures</div>
  <h1>Mistral Large 4,<br>taken <em>apart</em></h1>
  <p class="sub">Mistral Large 4 is a mixture-of-experts model with <b>${F['size.total_t']}T parameters</b>,
  <b>${F['size.active_b']}B</b> of them active on any one token, built for reasoning and long-horizon
  agentic work. Its technical report is unusually candid: the attention design picked to make it cheap
  to serve turned out to hurt exactly that work, and the model was <b>operated on in the middle of
  training</b> to fix it.</p>
  <p class="sub" style="margin-top:14px">The figures respond to clicks. Their measurements come from the paper, and when a
  figure is a schematic or does its own arithmetic, it says so.</p>

  <div class="metastrip">
    <span class="chip"><a href="${PAPER}">technical report</a></span>
    <span class="chip">FrontierCode 2.0 <b>${F['intro.frontiercode']}%</b></span>
    <span class="chip">Terminal-Bench 4.0 <b>${F['intro.tb4']}%</b></span>
    <span class="chip">AA Cyber Index <b>${F['intro.aacyber']}%</b></span>
    <span class="chip"><b>${F['q.gb']} GB</b> on ${F['cp.gpus']}&times;GB300</span>
  </div>
  <div class="toc">
    <div class="a1"><h4>The model</h4>
      <p>What was built, what it was fed, and how its settings were predicted rather than searched.</p></div>
    <div class="a2"><h4>The attention U-turn</h4>
      <p>Three tricks that shrank the KV cache, the agentic penalty they caused, and the surgery that
      undid most of them between the two pretraining phases.</p></div>
    <div class="a3"><h4>Agents write the tasks</h4>
      <p>Taxonomies, agent pipelines and verifiers that generate training environments, then
      midtraining and SFT.</p></div>
    <div class="a4"><h4>Reinforcement learning at scale</h4>
      <p>An asynchronous RL system, the escapes and reward hacks it caught, and three reasoning
      efforts from one student.</p></div>
    <div class="a5"><h4>How it scores, how it is served</h4>
      <p>Six tables against six models, and the serving work that fits it on four GPUs.</p></div>
  </div>
  <div class="scrollcue"><i></i> scroll &nbsp;&middot;&nbsp; the figure on the right is live</div>
</div>
`;

const OUTRO = `
<div class="outro">
  <h2>Notes</h2>
  <p>The measurements on this page come from the technical report,
  <a href="${PAPER}">Mistral Large 4</a>. The figures take table values and quoted numbers from data
  parsed out of the paper, and each quoted number is checked against the paper's text when the page
  is built. Some figures do simple arithmetic on those numbers, such as a ratio or a sum, and say so.
  Figures 6, 13, 15 and 17 are vector graphics, so their values
  are extracted exactly. Figures 2, 4, 5, 7, 8, 11, 14 and 16 are raster images; their values were read
  from the pixels of a 600 dpi render, are approximate, and the figures that use them say so.</p>
  <p>Where a figure here is a schematic, such as the layer diagrams or the token routing, it
  illustrates a mechanism the paper describes, its values are illustrative unless it says otherwise,
  and it is labelled as a schematic. The prose is my reading of the
  paper, not the paper's own words, so check anything that matters against the source.</p>
  <p>If a figure here disagrees with the paper, the paper is right. Tell me and I will fix it.</p>
</div>
`;

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${TITLE}</title>
<meta name="description" content="${DESC}">
<link rel="canonical" href="${URL}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="Avinash Sooriyarachchi">
<meta property="og:title" content="${TITLE}">
<meta property="og:description" content="${DESC}">
<meta property="og:url" content="${URL}">
<meta property="og:locale" content="en_US">
<meta property="og:image" content="${URL}og.png">
<meta property="og:image:secure_url" content="${URL}og.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="${TITLE}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@aviTwit3">
<meta name="twitter:creator" content="@aviTwit3">
<meta name="twitter:title" content="${TITLE}">
<meta name="twitter:description" content="${DESC}">
<meta name="twitter:image" content="${URL}og.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect width=%22100%22 height=%22100%22 fill=%22%23141922%22/><circle cx=%2250%22 cy=%2250%22 r=%2216%22 fill=%22none%22 stroke=%22%23b4551a%22 stroke-width=%228%22/></svg>">
<style>
${tokens}
${extra}
</style>
</head>
<body>
${HERO}

<div id="stage"><div id="art"></div></div>
<div class="railwrap"><div id="rail"></div></div>
<div id="actlab"></div>

<div id="beats">
${beats}
</div>

${OUTRO}

<script>
${data}
</script>
<script>
${kit}
window.SCENES = window.SCENES || {};
${js}
</script>
<script>
${engine}
</script>
</body>
</html>
`;

const out = process.env.OUT ? path.resolve(HERE, process.env.OUT) : path.join(HERE, 'index.html');
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, html);
console.log(`assembled ${path.relative(HERE, out)} (${html.length.toLocaleString()} bytes)`);
console.log(`scene files: ${files.join(', ') || 'none'}`);
console.log(`scenes wired: ${wanted.length - missing.length}/${wanted.length}`);
if (missing.length) console.log('MISSING SCENES: ' + missing.join(', '));
