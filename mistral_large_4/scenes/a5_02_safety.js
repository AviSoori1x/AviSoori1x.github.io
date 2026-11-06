/* S_SAFETY: Table 5, the internal safety evaluations (§5), and the introduction's
   96.6% cyber safety headline taken apart into the three cyber-subset scores it
   averages. Every value comes from window.ML4 (tables[4], facts, derived). */
window.SCENES.S_SAFETY = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts, D = api.D;
  // the introduction's headline, built once from facts and reused in every string below
  var HEAD = F['intro.safety'], HS = CH.fmt(HEAD, 1) + '%';
  var svg = K.board(host, { alt: 'Table 5 safety evaluations, and the ' + HS + ' cyber safety headline ' +
    'split into the three cyber-subset scores it averages' });
  K.head(svg, 'Taking apart the ' + HS, 'Table 5 · safety, run internally · higher is better on every row');
  var T = (D.tables || []).filter(function (t) { return t.n === 5; })[0];
  if (!T) { K.mono(svg, 0, 40, 'Table 5 values are not available.', { size: 14 }); return; }

  var ML = D.ml4, MUSE = D.short.indexOf('Muse Spark Open');
  function row(re) { return T.rows.filter(function (r) { return re.test(r.name); })[0]; }
  var KORA = row(/^KORA/), B3 = row(/^B3/), JBB = row(/^JailbreakBench/),
    SR = row(/^StrongREJECT/), AH = row(/^AgentHarm/);
  // the three parts of the headline, in the order the paper lists them
  var PARTS = [
    { r: JBB, name: 'JailbreakBench' },
    { r: SR, name: 'StrongREJECT' },
    { r: AH, name: 'AgentHarm' }
  ];
  // derived: the unweighted mean of ML4's three cyber-subset scores (paper: 96.6%)
  var MEAN = PARTS.reduce(function (s, p) { return s + p.r.vals[ML]; }, 0) / PARTS.length;
  function num(v) { return v === 100 ? '100' : CH.fmt(v, 1); }
  function ord(k) { return k + (k === 1 ? 'st' : k === 2 ? 'nd' : k === 3 ? 'rd' : 'th'); }
  // fade-and-slide entry, so a state change reads as motion rather than a swap
  function enter(el, dy, delay) {
    try {
      el.animate([{ opacity: 0, transform: 'translate(0px,' + dy + 'px)' },
        { opacity: 1, transform: 'translate(0px,0px)' }],
        { duration: 420, delay: delay || 0, easing: 'cubic-bezier(.2,.7,.2,1)', fill: 'backwards' });
    } catch (e) {}
  }

  /* ---------- top: the headline number, a button ---------- */
  var top = CH.body(svg);
  K.label(top, 0, 14, 'the introduction\'s cyber safety number');
  var btn = K.n('g', { tabindex: 0, role: 'button', 'aria-label': 'take the ' + HS + ' apart',
    style: 'cursor:pointer;outline:none' });
  var face = K.n('rect', { x: 0, y: 26, width: 238, height: 92, rx: 14,
    fill: K.TINT.amber, stroke: C.amber, 'stroke-width': 1.6 });
  btn.appendChild(face);
  // the browser outline is off, so keyboard focus shows as a heavier face stroke
  btn.addEventListener('focus', function () { face.setAttribute('stroke-width', 3.2); });
  btn.addEventListener('blur', function () { face.setAttribute('stroke-width', 1.6); });
  K.big(btn, 119, 96, HS, { size: 66, anchor: 'middle', color: C.ink });
  top.appendChild(btn);
  var line1 = K.mono(top, 266, 62, '', { size: 14, color: C.ink, weight: 600 });
  var line2 = K.mono(top, 266, 90, '', { size: 13, color: C.ink3 });

  /* ---------- all of Table 5 as dot rows, the headline's three parts first ---------- */
  var SX0 = 206, SX1 = 494;
  var PX = CH.scale(60, 100, SX0, SX1);      // the four percentage rows share one axis
  var KX = CH.scale(1.3, 1.9, SX0, SX1);     // KORA is a 0 to 2 mean score, on its own axis
  var ROWS = [
    { r: JBB, name: 'JailbreakBench, cyber', y: 232, sx: PX, pct: true, part: true },
    { r: SR, name: 'StrongREJECT, cyber', y: 268, sx: PX, pct: true, part: true },
    { r: AH, name: 'AgentHarm, cyber', y: 304, sx: PX, pct: true, part: true },
    { r: B3, name: 'B3 agentic security', y: 344, sx: PX, pct: true },
    { r: KORA, name: 'KORA, 0 to 2 scale', y: 432, sx: KX, pct: false }
  ];
  K.label(svg, 0, 172, 'Table 5');
  // a dot-only key (CH.key draws a line swatch, which suits lines, not dot rows)
  function keyDot(x, name, col, r) {
    svg.appendChild(K.n('circle', { cx: x + 6, cy: 167.5, r: r, fill: col, stroke: '#fff', 'stroke-width': 1.4 }));
    K.mono(svg, x + 18, 172, name, { size: 12.5, color: C.ink2 });
  }
  keyDot(100, 'Mistral Large 4', C.amber, 6);
  keyDot(250, 'Muse Spark Open', C.blue, 5.2);
  keyDot(400, 'the other five models', CH.GREY2, 4.6);

  // gridlines and tick labels, drawn once
  var grid = CH.body(svg);
  [60, 70, 80, 90, 100].forEach(function (t) {
    grid.appendChild(K.n('line', { x1: PX(t), x2: PX(t), y1: 214, y2: 358, stroke: C.line2 }));
    if (t % 20 === 0) K.mono(grid, PX(t), 380, t + '%', { size: 12.5, anchor: 'middle', color: C.ink3 });
  });
  grid.appendChild(K.n('line', { x1: 0, x2: 640, y1: 400, y2: 400, stroke: C.line }));
  [1.4, 1.6, 1.8].forEach(function (t) {
    grid.appendChild(K.n('line', { x1: KX(t), x2: KX(t), y1: 418, y2: 446, stroke: C.line2 }));
    K.mono(grid, KX(t), 474, CH.fmt(t, 1), { size: 12.5, anchor: 'middle', color: C.ink3 });
  });

  var rowsG = CH.body(svg), split = false, MX = PX(MEAN);
  function fmtV(R, v) { return R.pct ? num(v) + '%' : CH.fmt(v, 3); }
  // stack equal or near-equal scores, dot-plot style, so no model hides behind another:
  // ML4 stays on the row line, Muse tries below first, the grey models try above first
  function stack(R) {
    var v = R.r.vals, off = [], placed = [];
    function free(x, dy) {
      return placed.every(function (p) { return Math.hypot(p.x - x, p.dy - dy) >= 8.5; });
    }
    var order = [ML, MUSE].concat(v.map(function (_, j) { return j; })
      .filter(function (j) { return j !== ML && j !== MUSE; }));
    order.forEach(function (j) {
      if (v[j] == null) return;
      var x = R.sx(v[j]), tries = j === ML ? [0] : j === MUSE ? [0, 9, -9, 18] : [0, -9, 9, -18, 18];
      var dy = tries.filter(function (t) { return free(x, t); })[0];
      off[j] = dy == null ? 0 : dy;
      placed.push({ x: x, dy: off[j] });
    });
    return off;
  }
  function slide(el, dx, delay) {
    try {
      el.animate([{ transform: 'translate(' + dx + 'px,0px)' }, { transform: 'translate(0px,0px)' }],
        { duration: 520, delay: delay, easing: 'cubic-bezier(.2,.7,.2,1)', fill: 'backwards' });
    } catch (e) {}
  }
  function draw() {
    CH.clear(rowsG);
    face.setAttribute('fill', split ? '#fff' : K.TINT.amber);
    if (split) face.setAttribute('stroke-dasharray', '5 4'); else face.removeAttribute('stroke-dasharray');
    btn.setAttribute('aria-label', split ? 'put the ' + HS + ' back together' : 'take the ' + HS + ' apart');
    btn.setAttribute('aria-pressed', split ? 'true' : 'false');
    if (split) {
      // (97.3 + 92.6 + 100) / 3, built from the table values rather than typed in
      line1.textContent = '(' + PARTS.map(function (p) { return num(p.r.vals[ML]); }).join(' + ') +
        ') / 3 = ' + CH.fmt(MEAN, 2);
      line2.textContent = 'a plain average; click to put it back';
      // the three rows the headline averages, banded, with their mean as a rule
      var band = K.n('rect', { x: 0, y: 212, width: 640, height: 112, rx: 9,
        fill: K.TINT.amber, stroke: C.amber, 'stroke-opacity': .45 });
      rowsG.appendChild(band);
      enter(band, 0, 0);
      var rule = K.n('path', { d: 'M' + MX + ' 212 V324', stroke: C.ink, 'stroke-width': 1.5,
        'stroke-dasharray': '4 3' });
      rowsG.appendChild(rule);
      K.mono(rowsG, MX - 7, 206, 'mean ' + CH.fmt(MEAN, 2), { size: 13, color: C.ink, weight: 600, anchor: 'end' });
    } else {
      line1.textContent = 'one number, three cyber-subset scores';
      line2.textContent = 'click it to take it apart';
    }
    ROWS.forEach(function (R, ri) {
      var dim = split && !R.part, on = split && R.part;
      var g = K.n('g', { opacity: dim ? 0.4 : 1 }), y = R.y, v = R.r.vals;
      rowsG.appendChild(g);
      // inset 12, so the labels clear the band edge in the split state
      K.mono(g, 12, y + 4.6, R.name, { size: 13, color: on ? C.ink : C.ink2, weight: on ? 600 : 400 });
      g.appendChild(K.n('line', { x1: SX0, x2: SX1, y1: y, y2: y, stroke: C.line }));
      // grey context first, then the comparison, then ML4 on top; ties are stacked
      var off = stack(R);
      v.forEach(function (x, j) {
        if (j === ML || j === MUSE || x == null) return;
        g.appendChild(K.n('circle', { cx: R.sx(x), cy: y + off[j], r: 4.6, fill: CH.GREY2, stroke: '#fff', 'stroke-width': 1.4 }));
      });
      g.appendChild(K.n('circle', { cx: R.sx(v[MUSE]), cy: y + off[MUSE], r: 5.4, fill: C.blue, stroke: '#fff', 'stroke-width': 1.6 }));
      var dx = MX - R.sx(v[ML]);
      if (on) {
        // how far this part sits from the mean: the stem grows out of the rule
        var stem = K.n('path', { d: 'M' + MX + ' ' + y + ' H' + R.sx(v[ML]), stroke: 'rgba(180,85,26,.55)',
          'stroke-width': 3.2, 'stroke-linecap': 'round' });
        g.appendChild(stem);
        enter(stem, 0, 120 + 60 * ri);
      }
      var dot = K.n('circle', { cx: R.sx(v[ML]), cy: y, r: 6.6, fill: C.amber, stroke: '#fff', 'stroke-width': 2 });
      g.appendChild(dot);
      if (on) slide(dot, dx, 60 * ri);
      // ML4's place among the seven columns, ranked from the table values (data.js rank array)
      var ties = v.filter(function (x) { return x === v[ML]; }).length;
      var place = (ties > 1 ? 'tied ' : '') + ord(R.r.rank[ML]);
      K.mono(g, 628, y + 4.6, fmtV(R, v[ML]) + ', ' + place, { size: 13.5, color: C.ink, weight: 600, anchor: 'end' });
      // where ML4 is not first, name the leader and give its score just under its (blue) dot,
      // starting at the dot so it stays clear of the dashed mean rule and the axis ticks
      if (R.r.rank[MUSE] === 1 && R.r.rank[ML] !== 1) {
        K.mono(g, R.sx(v[MUSE]) - 6, y + off[MUSE] + 21, 'Muse ' + fmtV(R, v[MUSE]),
          { size: 12.5, color: C.ink2, anchor: 'start' });
      }
    });
  }

  function toggle() { split = !split; draw(); }
  btn.addEventListener('click', toggle);
  btn.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
  });
  draw();
  K.foot(svg, 'Table 5, internal runs. The mean ' + CH.fmt(MEAN, 2) + ' is computed here from the three ' +
    'table values; the introduction reports it as ' + HS + '. KORA, a mean score from 0 to 2, keeps its ' +
    'own axis. Equal or near-equal scores are stacked so every model shows.');
};
