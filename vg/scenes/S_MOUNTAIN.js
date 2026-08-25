window.SCENES = window.SCENES || {};

/* Act II, scene 08. The mountain.
   54.1M training rows as a unit field. One mark is a fixed number of rows, so
   the three buckets sit at true area proportion and the imbalance is visible
   before any label is read. Every quantity comes from window.SS.headline at
   runtime, including the text to multimodal ratio, which is computed here
   rather than asserted. */
window.SCENES['S_MOUNTAIN'] = function (root, api) {
  var SS = api.SS || {};
  var H = SS.headline || {};

  var totalRaw = (H.totalSamples == null) ? '' : String(H.totalSamples);
  var total = Number(totalRaw);

  /* millions of rows, straight from the paper's headline block */
  var BUCKETS = [
    { key: 'pub', cls: 'g0', shape: 'disc', name: 'public datasets, text', v: Number(H.openSourceText) },
    { key: 'syn', cls: 'g1', shape: 'ring', name: 'synthetic contrastive text', v: Number(H.syntheticText) },
    { key: 'mm', cls: 'g2', shape: 'sq', name: 'multimodal, image plus text', v: Number(H.multimodalSamples) }
  ];

  var ok = BUCKETS.every(function (b) { return isFinite(b.v) && b.v > 0; }) && isFinite(total);
  if (!ok) {
    root.classList.add('sc-s_mountain');
    root.appendChild(api.frag('<div class="wrap"><p class="miss">'
      + 'headline sample counts are not present in the data file, so this field cannot be drawn'
      + '</p></div>'));
    return null;
  }

  var sum = BUCKETS.reduce(function (a, b) { return a + b.v; }, 0);
  var textM = BUCKETS[0].v + BUCKETS[1].v;
  var mmM = BUCKETS[2].v;
  var ratioAll = textM / mmM;
  var ratioPub = BUCKETS[0].v / mmM;
  var sumsMatch = Math.abs(sum - total) < 0.05;

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function group(n) {
    var s = String(Math.round(n)), out = '', i, c = 0;
    for (i = s.length - 1; i >= 0; i--) {
      out = s.charAt(i) + out;
      if (++c % 3 === 0 && i > 0) out = ',' + out;
    }
    return out;
  }

  /* ---------------- the field ---------------- */

  var ASPECT = 2.5;    /* width over height of the dot block */
  var C = 10;          /* cell size in svg units */

  /* grains are expressed in millions of rows per mark */
  var GRAINS = [
    { g: 0.1, label: group(0.1 * 1e6) },
    { g: 0.02, label: group(0.02 * 1e6) }
  ];
  var grainIx = 0;

  /* the 1 to 3 rectangles covering a contiguous row major run of marks */
  function beds(a, b, cols) {
    var r0 = Math.floor(a / cols), c0 = a % cols;
    var r1 = Math.floor((b - 1) / cols), c1 = ((b - 1) % cols) + 1;
    var out = [], p = 0.6;
    function push(row, from, to, span) {
      out.push('<rect class="bed" x="' + (from * C + p).toFixed(1)
        + '" y="' + (row * C + p).toFixed(1)
        + '" width="' + ((to - from) * C - 2 * p).toFixed(1)
        + '" height="' + (span * C - 2 * p).toFixed(1) + '" rx="2"></rect>');
    }
    if (r0 === r1) { push(r0, c0, c1, 1); return out.join(''); }
    push(r0, c0, cols, 1);
    if (r1 > r0 + 1) push(r0 + 1, 0, cols, r1 - r0 - 1);
    push(r1, 0, c1, 1);
    return out.join('');
  }

  function mark(shape, cx, cy) {
    if (shape === 'ring') return '<circle class="m ring" cx="' + cx + '" cy="' + cy + '" r="2.8"></circle>';
    if (shape === 'sq') return '<rect class="m sq" x="' + (cx - 3.1) + '" y="' + (cy - 3.1)
      + '" width="6.2" height="6.2" rx=".8"></rect>';
    return '<circle class="m disc" cx="' + cx + '" cy="' + cy + '" r="3.5"></circle>';
  }

  var F = null;  /* current field geometry */

  function buildField() {
    var grain = GRAINS[grainIx].g;
    var counts = BUCKETS.map(function (b) { return Math.max(1, Math.round(b.v / grain)); });
    var N = counts.reduce(function (a, b) { return a + b; }, 0);
    var cols = Math.max(8, Math.round(Math.sqrt(N * ASPECT)));
    var rows = Math.ceil(N / cols);
    var W = cols * C, Hh = rows * C;

    var body = '', at = 0, i, j, cx, cy, ends = [];
    for (i = 0; i < BUCKETS.length; i++) {
      var n = counts[i], marks = '';
      for (j = at; j < at + n; j++) {
        cx = (j % cols) * C + C / 2;
        cy = Math.floor(j / cols) * C + C / 2;
        marks += mark(BUCKETS[i].shape, cx, cy);
      }
      body += '<g class="grp ' + BUCKETS[i].cls + '">' + beds(at, at + n, cols) + marks + '</g>';
      at += n;
      ends.push(at);
    }

    var lab = BUCKETS.map(function (b, k) {
      return esc(b.name) + ' takes ' + counts[k] + ' of them';
    }).join(', ');

    F = { N: N, cols: cols, rows: rows, W: W, H: Hh, counts: counts, ends: ends, grain: grain };

    return '<svg class="field" viewBox="0 0 ' + W + ' ' + Hh + '" role="img" aria-label="'
      + 'Unit field of ' + N + ' marks, one mark for every ' + esc(GRAINS[grainIx].label)
      + ' training rows. ' + lab + '.">'
      + '<defs><clipPath id="S_MOUNTAIN-clip" clipPathUnits="userSpaceOnUse">'
      + '<polygon id="S_MOUNTAIN-poly" points="0,0"></polygon></clipPath></defs>'
      + '<g clip-path="url(#S_MOUNTAIN-clip)">' + body + '</g>'
      + '</svg>';
  }

  /* ---------------- legend ---------------- */

  function glyph(shape) {
    var inner = shape === 'ring'
      ? '<circle class="ring" cx="8" cy="8" r="5"></circle>'
      : (shape === 'sq'
        ? '<rect class="sq" x="2.4" y="2.4" width="11.2" height="11.2" rx="1.4"></rect>'
        : '<circle class="disc" cx="8" cy="8" r="6.2"></circle>');
    return '<svg class="gly" viewBox="0 0 16 16" aria-hidden="true">' + inner + '</svg>';
  }

  function legendRow(b, k) {
    return '<button type="button" class="row ' + b.cls + '" id="S_MOUNTAIN-row-' + b.key + '"'
      + ' aria-pressed="false" data-ix="' + k + '">'
      + '<span class="top">'
      +   glyph(b.shape)
      +   '<span class="nm">' + esc(b.name) + '</span>'
      +   '<span class="val">' + api.num(b.v, 1) + '<i>M</i></span>'
      + '</span>'
      + '<span class="meta" id="S_MOUNTAIN-meta-' + b.key + '"></span>'
      + '</button>';
  }

  /* ---------------- shell ---------------- */

  root.classList.add('sc-s_mountain');
  root.appendChild(api.frag(
    '<div class="wrap" id="S_MOUNTAIN-wrap">'

    + '<div class="hd">'
    +   '<span class="eyeb">the training mix</span>'
    +   '<span class="ctrls">'
    +     '<span class="seg" role="group" aria-label="marks per row count">'
    +       '<button type="button" class="sg sel" id="S_MOUNTAIN-grain-0" aria-pressed="true">1 mark = '
    +          esc(GRAINS[0].label) + '</button>'
    +       '<button type="button" class="sg" id="S_MOUNTAIN-grain-1" aria-pressed="false">1 mark = '
    +          esc(GRAINS[1].label) + '</button>'
    +     '</span>'
    +     '<button type="button" class="rep" id="S_MOUNTAIN-replay" aria-label="replay the fill">'
    +       '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M13 8a5 5 0 1 1-1.6-3.6"'
    +       ' fill="none" stroke="currentColor" stroke-width="1.6"></path>'
    +       '<path d="M13.4 1.6V5H10" fill="none" stroke="currentColor" stroke-width="1.6"></path></svg>'
    +       '<span>replay</span>'
    +     '</button>'
    +   '</span>'
    + '</div>'

    + '<div class="big">'
    +   '<span class="numeral" id="S_MOUNTAIN-count" aria-live="off">0.0</span>'
    +   '<span class="unit">M</span>'
    +   '<span class="bigside">'
    +     '<b>samples in the Shieldstral training mix</b>'
    +     '<span class="ratio" id="S_MOUNTAIN-ratio"></span>'
    +   '</span>'
    + '</div>'

    + '<div class="fieldbox" id="S_MOUNTAIN-fieldbox"></div>'

    + '<div class="lgd" id="S_MOUNTAIN-lgd">'
    +   BUCKETS.map(legendRow).join('')
    +   '<p class="tot" id="S_MOUNTAIN-tot"></p>'
    + '</div>'

    + '<p class="foot">Bucket sizes are the paper\'s reported counts. The grid is a layout choice,'
    + ' and a mark stands for a fixed row count, not for any particular rows.</p>'

    + '</div>'
  ));

  var wrap = root.querySelector('#S_MOUNTAIN-wrap');
  var fieldbox = root.querySelector('#S_MOUNTAIN-fieldbox');
  var countEl = root.querySelector('#S_MOUNTAIN-count');
  var ratioEl = root.querySelector('#S_MOUNTAIN-ratio');
  var totEl = root.querySelector('#S_MOUNTAIN-tot');
  var rowEls = BUCKETS.map(function (b) { return root.querySelector('#S_MOUNTAIN-row-' + b.key); });
  var poly = null;

  ratioEl.innerHTML =
    '<span class="rline"><i>all text rows to multimodal</i><b>' + api.num(ratioAll, 1) + ' : 1</b></span>'
    + '<span class="rline dim"><i>public text alone to multimodal</i><b>'
    + api.num(ratioPub, 1) + ' : 1</b></span>';

  totEl.innerHTML = BUCKETS.map(function (b) { return api.num(b.v, 1); }).join(' + ')
    + ' = ' + api.num(sum, 1) + 'M, '
    + (sumsMatch
      ? 'which is the headline total of ' + esc(totalRaw) + 'M'
      : '<span class="warn">the headline total reads ' + esc(totalRaw) + 'M</span>');

  function paintLegend() {
    BUCKETS.forEach(function (b, k) {
      var share = (b.v / sum) * 100;
      root.querySelector('#S_MOUNTAIN-meta-' + b.key).textContent =
        F.counts[k] + ' marks at ' + GRAINS[grainIx].label + ' rows each, '
        + api.num(share, 1) + '% of the mix';
    });
  }

  /* ---------------- reveal ---------------- */

  var DUR = 2.7;
  var t0 = null, playing = false, at1 = false;

  function render(p) {
    var k = p * F.N;
    var row = Math.floor(k / F.cols);
    var pts;
    if (row >= F.rows) {
      pts = '0,0 ' + F.W + ',0 ' + F.W + ',' + F.H + ' 0,' + F.H;
    } else {
      var x = ((k - row * F.cols) / F.cols) * F.W;
      var yt = row * C, yb = (row + 1) * C;
      pts = '0,0 ' + F.W + ',0 ' + F.W + ',' + yt.toFixed(1) + ' '
        + x.toFixed(1) + ',' + yt.toFixed(1) + ' '
        + x.toFixed(1) + ',' + yb.toFixed(1) + ' 0,' + yb.toFixed(1);
    }
    if (poly) poly.setAttribute('points', pts);
    countEl.textContent = (p >= 1) ? totalRaw : api.num(total * p, 1);
    for (var i = 0; i < BUCKETS.length; i++) {
      rowEls[i].classList.toggle('in', k >= F.ends[i] - 0.5);
    }
    at1 = (p >= 1);
  }

  function mount(play) {
    fieldbox.innerHTML = buildField();
    poly = root.querySelector('#S_MOUNTAIN-poly');
    paintLegend();
    if (api.reduce || !play) { render(1); playing = false; }
    else { t0 = null; playing = true; render(0); }
  }

  mount(!api.reduce);

  /* ---------------- controls ---------------- */

  function setGrain(ix) {
    if (ix === grainIx) return;
    grainIx = ix;
    for (var i = 0; i < GRAINS.length; i++) {
      var b = root.querySelector('#S_MOUNTAIN-grain-' + i);
      b.classList.toggle('sel', i === ix);
      b.setAttribute('aria-pressed', i === ix ? 'true' : 'false');
    }
    mount(!api.reduce);
  }
  for (var gi = 0; gi < GRAINS.length; gi++) {
    (function (i) {
      root.querySelector('#S_MOUNTAIN-grain-' + i)
        .addEventListener('click', function () { setGrain(i); });
    })(gi);
  }

  var rep = root.querySelector('#S_MOUNTAIN-replay');
  if (api.reduce) rep.style.display = 'none';
  rep.addEventListener('click', function () { t0 = null; playing = true; render(0); });

  /* isolate one bucket on hover, pin it on click */
  var pinned = -1;
  function iso(ix) {
    for (var i = 0; i < BUCKETS.length; i++) wrap.classList.toggle('iso-' + i, i === ix);
    wrap.classList.toggle('iso', ix >= 0);
  }
  rowEls.forEach(function (btn, k) {
    btn.addEventListener('mouseenter', function () { if (pinned < 0) iso(k); });
    btn.addEventListener('mouseleave', function () { if (pinned < 0) iso(-1); });
    btn.addEventListener('focus', function () { if (pinned < 0) iso(k); });
    btn.addEventListener('blur', function () { if (pinned < 0) iso(-1); });
    btn.addEventListener('click', function () {
      pinned = (pinned === k) ? -1 : k;
      rowEls.forEach(function (o, j) { o.setAttribute('aria-pressed', j === pinned ? 'true' : 'false'); });
      iso(pinned < 0 ? -1 : pinned);
    });
  });

  return {
    start: function () { if (!at1) { t0 = null; playing = true; } },
    stop: function () { playing = false; },
    tick: function (s) {
      if (!playing) return;
      if (t0 == null) t0 = s;
      var p = (s - t0) / DUR;
      if (p >= 1) { p = 1; playing = false; }
      render(p);
    }
  };
};
