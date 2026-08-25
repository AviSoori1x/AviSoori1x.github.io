/* ============================================================
   w-pipeline: the four data stages and the corpus they produce
   ============================================================ */
(function () {
  var ID = 'w-pipeline';
  var root = document.getElementById(ID);
  if (!root) { return; }

  var SS = (typeof window !== 'undefined' && window.SS) ? window.SS : null;
  if (!SS || !SS.headline) { return; }

  var H = SS.headline;
  var strictness = SS.strictness || [];
  var formats = SS.formats || [];
  var f4 = SS.fig4 || {};

  function q(sel) { return root.querySelector(sel); }
  function esc(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
  function num(v, fallback) {
    var n = parseFloat(v);
    return isFinite(n) ? n : fallback;
  }

  /* ---------- corpus buckets, all values read from SS ---------- */

  var buckets = [
    { name: 'Open-source text', v: num(H.openSourceText, 0), sw: 1 },
    { name: 'Synthetic contrastive text', v: num(H.syntheticText, 0), sw: 2 },
    { name: 'Multimodal', v: num(H.multimodalSamples, 0), sw: 3 }
  ];
  var partsSum = buckets.reduce(function (a, b) { return a + b.v; }, 0);
  var total = num(H.totalSamples, partsSum) || partsSum;
  if (!(total > 0)) { total = 1; }

  /* which bucket each stage writes into */
  var STAGE_BUCKET = [0, 0, 1, 2];

  /* ---------- panel bodies ---------- */

  function chip(inner) { return '<span class="pl-chip">' + inner + '</span>'; }

  function rowsHtml(rows) {
    return '<div class="pl-rows">' + rows.map(function (r) {
      return '<div class="pl-row"><span class="pl-lab">' + r[0] + '</span><div>' + r[1] + '</div></div>';
    }).join('') + '</div>';
  }

  var caret =
    '<svg class="pl-caret" viewBox="0 0 11 7" preserveAspectRatio="xMidYMid meet" aria-hidden="true">' +
    '<path d="M0.6 0.6 L5.5 5.6 L10.4 0.6" fill="none" style="stroke:currentColor;stroke-width:1.1"/></svg>';

  function panel0() {
    var proc = ['labelling logic', 'category mappings', 'instruction templates']
      .map(function (t) { return chip(esc(t)); }).join('');

    var tiers = strictness.map(function (s) {
      return chip('<b>' + esc(s.level) + '</b> ' + esc(s.domains));
    }).join('');

    var fams = formats.map(function (f) { return chip(esc(f.family)); }).join('');

    return {
      lead: 'Each public dataset gets its own hand-written processor. The processor encodes what that dataset\'s labels actually mean, then emits pools of phrasings rather than one fixed wording.',
      body: rowsHtml([
        ['Per-dataset processor', '<div class="pl-chips">' + proc + '</div>'],
        ['Strictness tiers', '<div class="pl-chips">' + tiers + '</div>'],
        ['Document formats <span class="pl-leg-v">' + formats.length + '</span>',
          '<div class="pl-chips">' + fams + '</div>'],
        ['Result', '<p class="pl-note-line">Incompatible taxonomies collapse into one shape: <b>Instruct</b>, <b>Query</b>, <b>Document</b>. Every later stage writes into those same three fields.</p>']
      ])
    };
  }

  function panel1() {
    var pos = ['coarse binary', 'category-specific', 'target-group-specific']
      .map(function (t) { return chip(esc(t)); }).join('');
    var neg = ['category-based hard negatives', 'demographic-based negatives', 'safe-content negatives']
      .map(function (t) { return chip(esc(t)); }).join('');

    return {
      lead: 'The content is held fixed and the query moves. That is what produces the case a naive harm detector fails: an unsafe document whose correct answer is still no.',
      body: rowsHtml([
        ['Positive queries', '<div class="pl-chips">' + pos + '</div>'],
        ['Negative queries', '<div class="pl-chips">' + neg + '</div>'],
        ['Class balancing', '<p class="pl-note-line">Absent categories make cheap negatives, so negatives pile up. Each positive is duplicated <b>k</b> times, with the instruction and the query paraphrased independently per copy.</p>'],
        ['Label filtering', '<p class="pl-note-line">An open-source LLM cross-validates every sample and drops it when the dataset label disagrees at both the binary and the per-category level.</p>']
      ])
    };
  }

  function panel2() {
    var tax = chip('<b>' + esc(H.trainSupers) + '</b> super classes') +
              chip('<b>' + esc(H.trainLeaves) + '</b> leaf categories');

    var flow =
      '<div class="pl-flow">' +
        '<div class="pl-flow-step"><span class="pl-flow-k">In</span><span>safe source text</span></div>' +
        caret +
        '<div class="pl-flow-step"><span class="pl-flow-k">Rewrite</span><span>exhibit <em>' +
          esc(f4.category || '') + '</em>, avoid <em>' + esc(f4.sibling || '') + '</em></span></div>' +
        caret +
        '<div class="pl-flow-step"><span class="pl-flow-k">Out</span><span class="pl-chips">' +
          '<span class="pl-chip pl-chip-yes">query on target &rarr; ' +
            esc((f4.positive && f4.positive.label) || '') + '</span>' +
          '<span class="pl-chip pl-chip-no">query on sibling &rarr; ' +
            esc((f4.negative && f4.negative.label) || '') + '</span>' +
        '</span></div>' +
      '</div>';

    return {
      lead: 'Public data holds almost no pairs of near-identical texts that differ only in which sibling category they violate. Those are rare in the wild and costly to annotate, so an LLM writes them.',
      body: rowsHtml([
        ['Training taxonomy', '<div class="pl-chips">' + tax + '</div>'],
        ['One call', flow],
        ['Yield', '<p class="pl-note-line">Two training rows per generation, and the document is identical across both. Only the question changes.</p>']
      ])
    };
  }

  function panel3() {
    var pool = chip('<b>' + esc(H.imageQueryPhrasings) + '</b> query phrasings') +
               chip('<b>' + esc(H.imageSubcats) + '</b> visual subcategories') +
               chip('<b>' + esc(H.inversePct) + '%</b> inverse formulations');

    return {
      lead: 'You can ask an LLM to rewrite safe text into unsafe text. You cannot ask it for unsafe images, so the supply of positives is whatever already exists and diversity moves onto the query side.',
      body: rowsHtml([
        ['Query pool', '<div class="pl-chips">' + pool + '</div>'],
        ['Filtering', '<p class="pl-note-line">A vision-language reranker scores every image and query pair, lenient on the scarce positives and strict on the abundant negatives.</p>'],
        ['Elsewhere', '<p class="pl-note-line">This stage has a figure of its own further down the page.</p>']
      ])
    };
  }

  var STAGES = [
    { n: '01', title: 'Template-based unification', build: panel0 },
    { n: '02', title: 'Contrastive sample curation', build: panel1 },
    { n: '03', title: 'Contrastive sample generation', build: panel2 },
    { n: '04', title: 'Image data processing', build: panel3 }
  ];

  /* ---------- static wiring ---------- */

  var tabs = [];
  for (var i = 0; i < STAGES.length; i++) {
    var t = document.getElementById(ID + '-tab-' + i);
    if (t) { tabs.push(t); }
    var fe = document.getElementById(ID + '-feeds-' + i);
    if (fe) { fe.textContent = 'feeds ' + buckets[STAGE_BUCKET[i]].name.toLowerCase(); }
  }
  if (!tabs.length) { return; }

  var panelEl = document.getElementById(ID + '-panel');
  var barEl = document.getElementById(ID + '-bar');
  var barLab = document.getElementById(ID + '-barlab');
  var legendEl = document.getElementById(ID + '-legend');
  var totalEl = document.getElementById(ID + '-total');
  var ratioEl = document.getElementById(ID + '-ratio');

  function fmtPct(v) { return (v / total * 100).toFixed(1) + '%'; }

  if (totalEl) {
    totalEl.innerHTML = '<b>' + esc(H.totalSamples) + '</b>M samples';
  }

  if (ratioEl) {
    var textTotal = buckets[0].v + buckets[1].v;
    var mm = buckets[2].v;
    var ratio = mm > 0 ? textTotal / mm : 0;
    var ratioTxt = ratio >= 10 ? Math.round(ratio) : Math.round(ratio * 10) / 10;
    ratioEl.innerHTML = 'Text comes to <b>' + textTotal.toFixed(1) + 'M</b> rows, about <b>' +
      ratioTxt + 'x</b> the <b>' + mm.toFixed(1) + 'M</b> multimodal rows. The image supply is the ' +
      'binding constraint on the multimodal share.';
  }

  /* ---------- legend ---------- */

  var legItems = [];
  if (legendEl) {
    legendEl.innerHTML = buckets.map(function (b, k) {
      return '<li class="pl-leg" id="' + ID + '-leg-' + k + '">' +
        '<span class="pl-leg-sw pl-leg-sw-' + b.sw + '" aria-hidden="true"></span>' +
        '<span class="pl-leg-txt">' +
          '<span class="pl-leg-n">' + esc(b.name) + '</span>' +
          '<span class="pl-leg-v">' + b.v.toFixed(1) + 'M <span>' + fmtPct(b.v) + '</span></span>' +
          '<span class="pl-leg-tag" id="' + ID + '-legtag-' + k + '"></span>' +
        '</span></li>';
    }).join('');
    for (var k = 0; k < buckets.length; k++) {
      legItems.push({
        li: document.getElementById(ID + '-leg-' + k),
        tag: document.getElementById(ID + '-legtag-' + k)
      });
    }
  }

  if (barLab) {
    barLab.textContent = 'Corpus split. ' + buckets.map(function (b) {
      return b.name + ' ' + b.v.toFixed(1) + ' million, ' + fmtPct(b.v);
    }).join('. ') + '.';
  }

  /* ---------- proportional stacked bar ---------- */

  var active = 0;

  function renderBar() {
    if (!barEl) { return; }
    var w = 0;
    if (typeof barEl.getBoundingClientRect === 'function') {
      w = Math.round(barEl.getBoundingClientRect().width || 0);
    }
    if (!w) { w = Math.round(barEl.clientWidth || 0); }
    if (!w && barEl.parentNode) { w = Math.round(barEl.parentNode.clientWidth || 0); }
    if (!w) { w = 640; }
    var h = 46;
    var barY = 2, barH = 28, ruleY = barY + barH + 5;

    var defs =
      '<defs>' +
      '<pattern id="' + ID + '-hatch-2" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">' +
        '<rect width="6" height="6" style="fill:var(--paper)"/>' +
        '<line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--ink-2);stroke-width:2.4"/></pattern>' +
      '<pattern id="' + ID + '-hatch-3" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">' +
        '<rect width="7" height="7" style="fill:var(--paper)"/>' +
        '<line x1="0" y1="0" x2="0" y2="7" style="stroke:var(--ink-2);stroke-width:1.5"/></pattern>' +
      '</defs>';

    var out = '';
    var seps = '';
    var x = 0;
    var activeBucket = STAGE_BUCKET[active];

    for (var b = 0; b < buckets.length; b++) {
      var isLast = (b === buckets.length - 1);
      var segW = isLast ? (w - x) : Math.max(2, (buckets[b].v / total) * w);
      var fill = b === 0 ? 'var(--ink-2)' : 'url(#' + ID + '-hatch-' + buckets[b].sw + ')';
      var dim = (b === activeBucket) ? '' : ';opacity:.4';

      out += '<rect x="' + x.toFixed(2) + '" y="' + barY + '" width="' + segW.toFixed(2) +
        '" height="' + barH + '" style="fill:' + fill + dim + '" shape-rendering="crispEdges"/>';
      out += '<rect x="' + x.toFixed(2) + '" y="' + barY + '" width="' + segW.toFixed(2) +
        '" height="' + barH + '" style="fill:none;stroke:var(--ink-2);stroke-width:.9' + dim +
        '" shape-rendering="crispEdges"/>';

      if (segW >= 44) {
        var lblFill = (b === 0) ? 'var(--paper)' : 'var(--ink)';
        out += '<text x="' + (x + segW / 2).toFixed(2) + '" y="' + (barY + barH / 2 + 3.6).toFixed(2) +
          '" text-anchor="middle" style="font-family:var(--mono);font-size:10px;fill:' + lblFill + dim +
          '">' + esc(fmtPct(buckets[b].v)) + '</text>';
      }

      if (b === activeBucket) {
        out += '<rect x="' + x.toFixed(2) + '" y="' + ruleY + '" width="' + segW.toFixed(2) +
          '" height="2.5" style="fill:var(--accent)" shape-rendering="crispEdges"/>';
      }
      x += segW;
      if (!isLast) {
        seps += '<rect x="' + (x - 1.5).toFixed(2) + '" y="' + (barY - 1) + '" width="3" height="' +
          (barH + 2) + '" style="fill:var(--paper)" shape-rendering="crispEdges"/>';
      }
    }

    barEl.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    barEl.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    barEl.innerHTML = defs + out + seps;
  }

  /* ---------- selection ---------- */

  function select(idx, focus) {
    active = idx;
    for (var i = 0; i < tabs.length; i++) {
      var on = (i === idx);
      tabs[i].setAttribute('aria-selected', on ? 'true' : 'false');
      tabs[i].setAttribute('tabindex', on ? '0' : '-1');
    }
    var st = STAGES[idx];
    var content = st.build();
    if (panelEl) {
      panelEl.innerHTML =
        '<div class="pl-p-head">' +
          '<span class="pl-p-num">' + esc(st.n) + '</span>' +
          '<span class="pl-p-title">' + esc(st.title) + '</span>' +
        '</div>' +
        '<p class="pl-p-lead">' + content.lead + '</p>' +
        content.body;
      panelEl.setAttribute('aria-labelledby', ID + '-tab-' + idx);
    }
    var ab = STAGE_BUCKET[idx];
    for (var k = 0; k < legItems.length; k++) {
      if (!legItems[k].li) { continue; }
      if (k === ab) {
        legItems[k].li.classList.remove('pl-dim');
        legItems[k].tag.textContent = 'from stage ' + st.n;
      } else {
        legItems[k].li.classList.add('pl-dim');
        legItems[k].tag.textContent = '';
      }
    }
    renderBar();
    if (focus) { tabs[idx].focus(); }
  }

  for (var j = 0; j < tabs.length; j++) {
    (function (idx) {
      tabs[idx].addEventListener('click', function () { select(idx, false); });
      tabs[idx].addEventListener('keydown', function (ev) {
        var key = ev.key, next = -1;
        if (key === 'ArrowRight' || key === 'ArrowDown') { next = (idx + 1) % tabs.length; }
        else if (key === 'ArrowLeft' || key === 'ArrowUp') { next = (idx - 1 + tabs.length) % tabs.length; }
        else if (key === 'Home') { next = 0; }
        else if (key === 'End') { next = tabs.length - 1; }
        if (next >= 0) { ev.preventDefault(); select(next, true); }
      });
    })(j);
  }

  /* ---------- responsive redraw ---------- */

  var pending = false;
  function scheduleRedraw() {
    if (pending) { return; }
    pending = true;
    var raf = (typeof window.requestAnimationFrame === 'function')
      ? window.requestAnimationFrame
      : function (fn) { return setTimeout(fn, 16); };
    raf(function () { pending = false; renderBar(); });
  }

  if (typeof window.ResizeObserver === 'function' && barEl) {
    new window.ResizeObserver(scheduleRedraw).observe(barEl);
  } else {
    window.addEventListener('resize', scheduleRedraw);
  }

  select(0, false);
})();
