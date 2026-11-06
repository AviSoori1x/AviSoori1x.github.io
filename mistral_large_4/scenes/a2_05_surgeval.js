/* S_SURGEVAL: Figure 6, exact vector reads (window.ML4.figread.fig6). Two stacked
   panels on one midtraining-step axis: SWE-bench Verified on top, training
   perplexity below. The phase-2 surgery runs are amber, the full-attention
   baseline blue, the rest context. Hover either panel: the nearest point gets a
   ring and a readout. Early perplexity sits above the paper's plotted range, so
   those lines enter at the top edge, as in the paper. */
window.SCENES.S_SURGEVAL = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts, S = (api.D.figread || {}).fig6;
  var SUP = { 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  var sup = function (v) { return String(v).split('').map(function (d) { return SUP[d]; }).join(''); };
  var svg = K.board(host, { alt: 'Figure 6: SWE-bench Verified and training perplexity during midtraining for six surgery and baseline runs' });
  K.head(svg, 'Loss came close, SWE-bench did not', 'Figure 6 · surgery tested at 8×10' +
    sup(F['surg.small_exp']) + ' FLOPs · then midtraining');
  var body = CH.body(svg);
  if (!S || !S.length) {
    K.mono(body, 0, 100, 'Figure 6 values are not available.', { size: 14, color: C.ink2 });
    return;
  }
  var get = function (nm) { return S.filter(function (s) { return s.name === nm; })[0]; };
  var ORIG = 'Original baseline', FULL = 'Full-attention baseline', MID = 'Midtraining only: YaRN',
    VAN = 'Phase 2 + midtraining: vanilla', YARN = 'Phase 2 + midtraining: YaRN',
    LER = 'Phase 2 + midtraining: YaRN + LeRoPE + SSMax';
  // the phase-2 surgery runs are the focus (amber); full attention is the comparison (blue).
  // Dashes are long enough to stay open under the round line caps.
  var DASH = '7 5', ST = {};
  ST[ORIG] = { c: CH.GREY, w: 2 };
  ST[FULL] = { c: C.blue, w: 2.4 };
  ST[MID] = { c: CH.GREY, w: 2, dash: DASH };
  ST[VAN] = { c: C.amber, w: 2.4 };
  ST[YARN] = { c: C.amber, w: 2.4, dash: DASH };
  ST[LER] = { c: CH.GREY2, w: 2 };
  var DRAW = [LER, MID, ORIG, YARN, VAN, FULL];

  // key, short names (the readout gives the paper's full name): the focus and the
  // comparison on the first row, context on the second. A local 30px swatch, because
  // CH.key's 18px round-capped line hides a dash: solid runs show their marker, dashed
  // runs show the dash with square ends and no marker over the gaps.
  var SHORT = {};
  SHORT[VAN] = 'phase 2: vanilla'; SHORT[YARN] = 'phase 2: YaRN'; SHORT[FULL] = 'full-attention baseline';
  SHORT[ORIG] = 'original baseline'; SHORT[MID] = 'midtraining only'; SHORT[LER] = 'phase 2: YaRN, LeRoPE, SSMax';
  [[VAN, 0, 8], [YARN, 184, 8], [FULL, 344, 8], [ORIG, 0, 30], [MID, 184, 30], [LER, 368, 30]].forEach(function (k) {
    var st = ST[k[0]], x = k[1], y = k[2] - 4.5;
    body.appendChild(K.n('path', { d: 'M' + x + ' ' + y + ' h30', stroke: st.c, 'stroke-width': 2.5,
      'stroke-dasharray': st.dash || null }));
    if (!st.dash) body.appendChild(K.n('circle', { cx: x + 15, cy: y, r: 3.6, fill: st.c, stroke: '#fff',
      'stroke-width': 1.2 }));
    K.mono(body, x + 37, k[2], SHORT[k[0]], { size: 12.5, color: C.ink2 });
  });

  // SWE-bench from 0, as in the paper
  var X = { x0: 0, x1: 3200 }, TK = [0, 1000, 2000, 3000];
  var top = CH.axes(body, 54, 72, 586, 150, { x0: X.x0, x1: X.x1, y0: 0, y1: 60,
    yt: [0, 10, 20, 30, 40, 50, 60], yl: 'SWE-bench Verified pass@1 (%), higher is better' });
  // the top panel shares the bottom panel's step axis: gridlines only, labels below
  TK.forEach(function (t) {
    body.insertBefore(K.n('line', { x1: top.sx(t), x2: top.sx(t), y1: top.y, y2: top.y + top.h, stroke: C.line2 }),
      top.g.nextSibling);
  });
  // the paper clips early perplexity at its axis top, so the largest read is that top
  var PMAX = Math.max.apply(null, S.map(function (s) {
    return Math.max.apply(null, s.ppl.map(function (q) { return q.ppl; }));
  }));
  var bot = CH.axes(body, 54, 260, 586, 90, { x0: X.x0, x1: X.x1, y0: 1.48, y1: PMAX,
    xt: TK, yt: [1.50, 1.55, 1.60, 1.65], xf: function (v) { return v ? v / 1000 + 'k' : '0'; },
    yf: function (v) { return v.toFixed(2); }, xl: 'midtraining step',
    yl: 'training perplexity in midtraining, lower is better' });

  DRAW.forEach(function (nm) {
    var s = get(nm), st = ST[nm];
    if (!s) return;
    var sw = s.swe.map(function (q) { return [q.step, q.swe]; });
    CH.line(body, top, sw, { color: st.c, w: st.w, dash: st.dash });
    CH.dots(body, top, sw, { color: st.c, r: 3.2 });
    CH.line(body, bot, s.ppl.map(function (q) { return [q.step, q.ppl]; }), { color: st.c, w: st.w - 0.6,
      dash: st.dash });
  });

  // hover: a ring on the point, the series redrawn heavier on top, one readout line
  var fx = K.n('g', {});
  body.appendChild(fx);
  var out = CH.readout(body, 0, 420, { size: 13 });
  // SWE-bench checkpoints sit one step short of round numbers (1499, 2999): show them
  // round, but keep a step that is genuinely off the grid (the last, 3071) as it is
  var snap = function (s) { var r = Math.round(s / 50) * 50; return Math.abs(r - s) <= 2 ? r : s; };
  function focus(nm, ax, step, v, isPpl) {
    while (fx.firstChild) fx.removeChild(fx.firstChild);
    var s = get(nm), st = ST[nm];
    var pts = isPpl ? s.ppl.map(function (q) { return [q.step, q.ppl]; }) :
      s.swe.map(function (q) { return [q.step, q.swe]; });
    CH.line(fx, ax, pts, { color: st.c === CH.GREY2 ? CH.GREY : st.c, w: st.w + 1.4, dash: st.dash });
    fx.appendChild(K.n('circle', { cx: ax.sx(step), cy: ax.sy(v), r: 7, fill: 'none', stroke: C.ink, 'stroke-width': 2 }));
    out.set(nm + ' · step ' + (isPpl ? step : snap(step)) + ' · ' +
      (isPpl ? 'perplexity ' + CH.fmt(v, 3) : 'SWE-bench ' + CH.fmt(v, 1) + '%'));
  }

  // targets: one per panel. The nearest point under the pointer wins, so markers that
  // crowd together at the last checkpoint all stay reachable.
  var hits = K.n('g', {});
  function nearest(ax, isPpl, e) {
    if (e.clientX == null) return;
    var q = CH.toLocal(hits, e), best = null, bd = 1e9;
    DRAW.forEach(function (nm) {
      (isPpl ? get(nm).ppl : get(nm).swe).forEach(function (p) {
        var v = isPpl ? p.ppl : p.swe, dx = ax.sx(p.step) - q.x, dy = ax.sy(v) - q.y, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = [nm, p.step, v]; }
      });
    });
    if (best) focus(best[0], ax, best[1], best[2], isPpl);
  }
  [[top, false, 'SWE-bench panel'], [bot, true, 'perplexity panel']].forEach(function (P) {
    var ax = P[0], r = CH.hot(hits, 0, 0, 0, function (e) { nearest(ax, P[1], e); },
      { rect: [ax.x - 6, ax.y - 8, ax.w + 6, ax.h + 16], label: P[2] });
    r.addEventListener('mousemove', function (e) { nearest(ax, P[1], e); });
  });
  svg.appendChild(hits);
  var full = get(FULL), fl = full.swe[full.swe.length - 1];
  focus(FULL, top, fl.step, fl.swe, false);

  // the finding, read at the last checkpoint of each run
  var end = function (nm, k) { var a = get(nm)[k]; return a[a.length - 1][k]; };
  var p3 = function (nm) { return CH.fmt(end(nm, 'ppl'), 3); }, s1 = function (nm) { return CH.fmt(end(nm, 'swe'), 1); };
  var notBetter = Math.max(end(VAN, 'swe'), end(YARN, 'swe')) <= end(MID, 'swe');
  K.callout(body, 0, 434, 640, 'Surgery before phase 2 (amber) pulled perplexity to ' + p3(VAN) + ' with vanilla ' +
    'and ' + p3(YARN) + ' with YaRN, near full attention\'s ' + p3(FULL) + ' (blue). Yet SWE-bench ended at ' +
    s1(VAN) + ' (vanilla) and ' + s1(YARN) + ' (YaRN): well below full attention\'s ' + s1(FULL) + ', and ' +
    (notBetter ? 'no better than' : 'close to') + ' midtraining-only surgery at the end (' + s1(MID) + ').',
    { cols: 74 });

  K.foot(svg, 'Values are exact reads of the vector Figure 6. Early perplexity values sit above the plotted ' +
    'range, as in the paper, so the lines start at the top edge. The YaRN + LeRoPE + SSMax run is the next ' +
    'beat\'s subject.');
};
