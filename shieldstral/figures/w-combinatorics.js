/* ============================================================
   w-combinatorics: template randomisation.
   One fixed source sample, redrawn under a
   (strictness x document format family x query phrasing) triple.
   Every string and every count is read from window.SS.
   Nothing here calls a model. The answer chip is the paper's
   ground-truth label for this sample.
   ============================================================ */
(function () {
  var ID = 'w-combinatorics';
  var root = document.getElementById(ID);
  if (!root) return;

  var SS = window.SS;
  if (!SS) return;

  function el(suffix) { return document.getElementById(ID + '-' + suffix); }

  /* ---------- pools, all read from SS ---------- */

  var tiers = Array.isArray(SS.strictness) ? SS.strictness : [];
  var fmts = Array.isArray(SS.formats) ? SS.formats : [];

  var fig2 = Array.isArray(SS.fig2) ? SS.fig2 : [];
  var textSample = null;
  for (var i = 0; i < fig2.length; i++) {
    if (fig2[i] && typeof fig2[i].document === 'string' && fig2[i].document.indexOf('[image]') === -1) {
      textSample = fig2[i];
      break;
    }
  }
  if (!textSample && fig2.length) textSample = fig2[0];

  var fig3 = SS.fig3 || {};
  var fig4 = SS.fig4 || {};

  /* SS carries no explicit query-template pool, so the phrasings are the
     physical-harm queries that the paper's own figures print. */
  var queries = [];
  function addQuery(q) {
    if (typeof q === 'string' && q && queries.indexOf(q) === -1) queries.push(q);
  }
  if (textSample) addQuery(textSample.query);
  if (fig3.positive) addQuery(fig3.positive.query);
  addQuery(fig4.query);

  if (!tiers.length || !fmts.length || !queries.length || !textSample) return;

  var baseInstruct = (fig4.instruct || fig3.instruct || '').replace(/\s*\.?\s*$/, '');
  var sysPrompt = typeof SS.systemPrompt === 'string' ? SS.systemPrompt : '';
  var answer = typeof textSample.label === 'string' ? textSample.label : '';

  /* the fixed source content: split the bracketed document back into its two turns */
  var srcPrompt = textSample.document;
  var srcResponse = '';
  var parts = textSample.document.match(/\[User\]\s*([\s\S]*?)\s*\[Assistant\]\s*([\s\S]*)$/);
  if (parts) {
    srcPrompt = parts[1].trim();
    srcResponse = parts[2].trim();
  }

  /* the paper's own description of its training query pool (Table 3) */
  var paperPool = '';
  var tc = Array.isArray(SS.taxCompare) ? SS.taxCompare : [];
  for (var t = 0; t < tc.length; t++) {
    if (tc[t] && /query generation/i.test(tc[t].aspect || '')) { paperPool = tc[t].train || ''; break; }
  }

  var nS = tiers.length, nF = fmts.length, nQ = queries.length;
  var TOTAL = nS * nF * nQ;

  /* ---------- helpers ---------- */

  function fillAll(str, token, value) { return str.split(token).join(value); }

  function renderDoc(fmt) {
    var out = typeof fmt.tpl === 'string' ? fmt.tpl : '';
    out = fillAll(out, '{prompt}', srcPrompt);
    out = fillAll(out, '{response}', srcResponse);
    return out;
  }

  function oneLine(str) { return String(str).replace(/\s*\n\s*/g, '  /  '); }

  function reduced() {
    return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }

  function randInt(n) { return Math.floor(Math.random() * n); }

  function put(node, text) { if (node) node.textContent = text; }

  /* ---------- static chrome ---------- */

  var math = el('math');
  if (math) {
    math.textContent = '';
    var pieces = [
      [String(nS), 'strictness framings'],
      [String(nF), 'format families'],
      [String(nQ), 'query phrasings']
    ];
    for (var p = 0; p < pieces.length; p++) {
      if (p) math.appendChild(document.createTextNode('  x  '));
      var b = document.createElement('b');
      b.textContent = pieces[p][0];
      math.appendChild(b);
      math.appendChild(document.createTextNode(' ' + pieces[p][1]));
    }
    math.appendChild(document.createTextNode('  =  '));
    var tot = document.createElement('span');
    tot.className = 'wc-tot';
    tot.textContent = String(TOTAL);
    math.appendChild(tot);
    math.appendChild(document.createTextNode(' surface variants of one sample '));
    var chip = document.createElement('span');
    chip.className = 'wc-demo-chip';
    chip.textContent = 'in this demo';
    math.appendChild(chip);
  }

  put(el('sys'), sysPrompt);

  var ansEl = el('ans');
  if (ansEl) {
    ansEl.textContent = answer;
    if (/^yes$/i.test(answer)) ansEl.className = 'wc-answer-tok wc-yes';
  }

  put(el('gridlab'),
    'One row per strictness framing, labelled by its first three letters. Each block of ' + nF +
    ' columns is one query phrasing (Q1 to Q' + nQ + '), one column per document format family. ' +
    'Hover a square to read the combination it stands for.');

  /* honesty notes, with every number read from SS */
  put(el('note-real'),
    'The ' + nS + ' strictness framings, the ' + nF + ' document format families, the system message and the ' +
    'answer come straight from the paper. Nothing on this page calls a model: the answer chip is the ' +
    "paper's ground-truth label for this sample, not a prediction. The paper prints only strict-tier " +
    "instruction text, so the <Instruct> line you see is composed from the paper's wording rather than " +
    'quoted, and the ' + nQ + ' query phrasings are the physical-harm queries its figures happen to print. ' +
    (paperPool ? "The paper's own training pool is bigger: " + paperPool.replace(/^Dynamic:\s*/i, '') + '.' : ''));

  put(el('note-scope'),
    'The real space is larger than ' + TOTAL + ', because each format family holds several variants of its own ' +
    'and the query pool is per category. It is also not a free choice: the paper expects <Instruct> to stay ' +
    'constant across a dataset or product surface, and picks strictness per source dataset rather than per ' +
    'sample, so a single training row never really sees all ' + nS + ' framings. It is a dial here so you can ' +
    'see what each one looks like. Finally, rewording a question is not the same as asking a different one. ' +
    'Ask about a category the document does not contain and the correct answer can change, which is ' +
    'exactly how the paper mints its hard negatives.');

  /* ---------- coverage grid ---------- */

  var SVGNS = 'http://www.w3.org/2000/svg';
  var CW = 8, GAP = 1, GGAP = 6, GUT = 13, HEAD = 8;
  var groupW = nF * (CW + GAP) - GAP;
  var gridW = GUT + nQ * groupW + (nQ - 1) * GGAP;
  var gridH = HEAD + nS * (CW + GAP) - GAP;

  var svg = el('grid');
  var cells = [];
  var titles = [];

  function cellX(qi, fi) { return GUT + qi * (groupW + GGAP) + fi * (CW + GAP); }
  function cellY(si) { return HEAD + si * (CW + GAP); }

  if (svg) {
    svg.setAttribute('viewBox', '0 0 ' + gridW + ' ' + gridH);
    for (var qi = 0; qi < nQ; qi++) {
      var glab = document.createElementNS(SVGNS, 'text');
      glab.setAttribute('class', 'wc-gl');
      glab.setAttribute('x', String(cellX(qi, 0)));
      glab.setAttribute('y', '5');
      glab.textContent = 'Q' + (qi + 1);
      svg.appendChild(glab);
    }
    for (var si = 0; si < nS; si++) {
      var rlab = document.createElementNS(SVGNS, 'text');
      rlab.setAttribute('class', 'wc-gl');
      rlab.setAttribute('x', '0');
      rlab.setAttribute('y', String(cellY(si) + CW / 2 + 1.7));
      rlab.textContent = String(tiers[si].level || '').slice(0, 3).toUpperCase();
      svg.appendChild(rlab);
      for (var qj = 0; qj < nQ; qj++) {
        for (var fi = 0; fi < nF; fi++) {
          var r = document.createElementNS(SVGNS, 'rect');
          r.setAttribute('class', 'wc-cell');
          r.setAttribute('x', String(cellX(qj, fi)));
          r.setAttribute('y', String(cellY(si)));
          r.setAttribute('width', String(CW));
          r.setAttribute('height', String(CW));
          var ttl = document.createElementNS(SVGNS, 'title');
          ttl.textContent = tiers[si].level + ', ' + fmts[fi].family + ', query phrasing ' + (qj + 1) + '. Not yet drawn.';
          r.appendChild(ttl);
          svg.appendChild(r);
          var idx = si * nQ * nF + qj * nF + fi;
          cells[idx] = r;
          titles[idx] = ttl;
        }
      }
    }
  }

  function titleBase(s, f, q) {
    return tiers[s].level + ', ' + fmts[f].family + ', query phrasing ' + (q + 1);
  }

  /* ---------- state ---------- */

  var cur = [0, 0, 0];
  var seen = {};
  var seenCount = 0;
  var timers = [];
  var flashed = [];
  var spinning = false;

  function key(c) { return c[0] + ':' + c[1] + ':' + c[2]; }
  function cellIndex(c) { return c[0] * nQ * nF + c[2] * nF + c[1]; }

  function markSeen(c) {
    var k = key(c);
    if (!seen[k]) { seen[k] = 1; seenCount++; }
  }

  function paintGrid() {
    var nowIdx = cellIndex(cur);
    for (var s = 0; s < nS; s++) {
      for (var q = 0; q < nQ; q++) {
        for (var f = 0; f < nF; f++) {
          var idx = s * nQ * nF + q * nF + f;
          var c = cells[idx];
          if (!c) continue;
          var isSeen = !!seen[s + ':' + f + ':' + q];
          var isNow = idx === nowIdx;
          var cls = 'wc-cell';
          if (isSeen) cls += ' wc-seen';
          if (isNow) cls += ' wc-now';
          c.setAttribute('class', cls);
          if (titles[idx]) {
            titles[idx].textContent = titleBase(s, f, q) + '. ' +
              (isNow ? 'Current draw.' : (isSeen ? 'Drawn at least once.' : 'Not yet drawn.'));
          }
        }
      }
    }
    var cnt = el('count');
    if (cnt) {
      cnt.textContent = '';
      cnt.appendChild(document.createTextNode('drawn '));
      var sp = document.createElement('span');
      sp.className = 'wc-tot';
      sp.textContent = String(seenCount) + ' of ' + String(TOTAL);
      cnt.appendChild(sp);
    }
  }

  /* ---------- slot painting ---------- */

  function slotText(slot, c) {
    if (slot === 0) return tiers[c[0]].level || '';
    if (slot === 1) return fmts[c[1]].family || '';
    return queries[c[2]] || '';
  }

  function slotSub(slot, c) {
    if (slot === 0) {
      var tier = tiers[c[0]];
      return (tier.domains ? tier.domains + '. ' : '') + (tier.rationale || '');
    }
    if (slot === 1) return oneLine(fmts[c[1]].tpl || '');
    return 'phrasing ' + (c[2] + 1) + ' of ' + nQ;
  }

  function paintSlot(slot, c) {
    put(el('v' + slot), slotText(slot, c));
    put(el('s' + slot), slotSub(slot, c));
  }

  function setSlotState(slot, cls) {
    var box = el('slot-' + slot);
    if (box) box.className = 'wc-slot' + (cls ? ' ' + cls : '');
  }

  /* ---------- rendered training view ---------- */

  var lastRender = { instruct: null, query: null, doc: null };

  function instructFor(c) {
    var lvl = String(tiers[c[0]].level || '').toLowerCase();
    return baseInstruct + '. Apply a ' + lvl + ' standard.';
  }

  function unflash() {
    for (var i = 0; i < flashed.length; i++) {
      var n = flashed[i];
      if (n) n.className = n.className.replace(/\s*wc-flash/g, '');
    }
    flashed = [];
  }

  function flash(node) {
    if (!node || reduced()) return;
    node.className = node.className.replace(/\s*wc-flash/g, '') + ' wc-flash';
    if (flashed.indexOf(node) === -1) flashed.push(node);
    timers.push(setTimeout(function () {
      node.className = node.className.replace(/\s*wc-flash/g, '');
    }, 620));
  }

  function paintView(c) {
    var next = {
      instruct: instructFor(c),
      query: queries[c[2]],
      doc: renderDoc(fmts[c[1]])
    };
    put(el('instruct'), next.instruct);
    put(el('query'), next.query);
    put(el('doc'), next.doc);
    if (lastRender.instruct !== null) {
      if (next.instruct !== lastRender.instruct) flash(el('f-instruct'));
      if (next.query !== lastRender.query) flash(el('f-query'));
      if (next.doc !== lastRender.doc) flash(el('f-doc'));
    }
    lastRender = next;
  }

  function announce(c, extra) {
    put(el('live'), tiers[c[0]].level + ' framing, ' + fmts[c[1]].family + ' format, query phrasing ' +
      (c[2] + 1) + ' of ' + nQ + '. ' + (extra || '') + seenCount + ' of ' + TOTAL +
      ' combinations drawn. The answer is still ' + answer + '.');
  }

  function settle(c, quiet, extra) {
    cur = c;
    for (var s = 0; s < 3; s++) { paintSlot(s, c); setSlotState(s, ''); }
    markSeen(c);
    paintView(c);
    paintGrid();
    if (!quiet) announce(c, extra);
  }

  /* ---------- rolling ---------- */

  function drawTriple() {
    var c, guard = 0;
    do {
      c = [randInt(nS), randInt(nF), randInt(nQ)];
      guard++;
    } while (TOTAL > 1 && key(c) === key(cur) && guard < 40);
    return c;
  }

  function clearTimers() {
    for (var i = 0; i < timers.length; i++) clearTimeout(timers[i]);
    timers = [];
    unflash();
  }

  function setDisabled(on) {
    var a = el('roll'), b = el('roll10'), c = el('reset');
    if (a) a.disabled = on;
    if (b) b.disabled = on;
    if (c) c.disabled = on;
  }

  function roll() {
    if (spinning) return;
    var target = drawTriple();

    if (reduced()) { settle(target); return; }

    spinning = true;
    setDisabled(true);
    clearTimers();

    var lens = [nS, nF, nQ];
    var stops = [400, 570, 740];

    for (var s = 0; s < 3; s++) {
      setSlotState(s, 'wc-spin');
      (function (slot) {
        var tick = 0;
        function step() {
          if (tick * 72 >= stops[slot]) {
            setSlotState(slot, 'wc-settled');
            var fixed = [cur[0], cur[1], cur[2]];
            fixed[slot] = target[slot];
            cur = fixed;
            paintSlot(slot, target);
            if (slot === 2) {
              timers.push(setTimeout(function () {
                spinning = false;
                setDisabled(false);
                settle(target);
              }, 120));
            }
            return;
          }
          var fake = [cur[0], cur[1], cur[2]];
          fake[slot] = randInt(lens[slot]);
          paintSlot(slot, fake);
          tick++;
          timers.push(setTimeout(step, 72));
        }
        step();
      })(s);
    }
  }

  function rollTen() {
    if (spinning) return;
    clearTimers();
    var c = cur;
    for (var i = 0; i < 10; i++) {
      c = drawTriple();
      cur = c;
      markSeen(c);
    }
    settle(c, false, 'Ten draws applied. ');
  }

  function resetCoverage() {
    if (spinning) return;
    clearTimers();
    seen = {};
    seenCount = 0;
    settle(cur, false, 'Coverage cleared. ');
  }

  var rollBtn = el('roll');
  if (rollBtn) rollBtn.addEventListener('click', roll);
  var roll10Btn = el('roll10');
  if (roll10Btn) roll10Btn.addEventListener('click', rollTen);
  var resetBtn = el('reset');
  if (resetBtn) resetBtn.addEventListener('click', resetCoverage);

  /* ---------- boot ---------- */

  settle([0, 0, 0], true);
})();
