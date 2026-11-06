/* S_FRONTIER: Figure 16, the decode trade-off, one optimization at a time. The
   points are read from the raster figure (window.ML4.figread.fig16) and are
   approximate. The caption gives hardware for two curves only: the FP8 baseline
   on 8 GB300s and the NVFP4 curve on 4. Every point's per-GPU value is its
   concurrency times its interactivity over 8 GPUs (4 for NVFP4); for example
   64 x 24.4 / 8 = 195, read 192 on +PDD. The paper does not say whether prefill
   runs on extra GPUs once it is disaggregated, so the +PDD headline is the
   interactivity ratio, which does not depend on counting GPUs. At 8 requests the
   NVFP4 curve runs each request at the DSpark speed, so its doubling per GPU is
   the same output on half the GPUs. The y axis is log scale so the low curves
   stay readable. */
window.SCENES.S_FRONTIER = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts, n = K.n;
  var FR = (api.D.figread || {}).fig16;
  var svg = K.board(host, { alt: 'Figure 16: decode interactivity against output throughput per GPU for five serving setups' });
  K.head(svg, 'Each optimization moves the curve', 'Figure 16 · decode at ' + F['fig16.input_k'] + 'K input · ' +
    'baseline on ' + F['fig16.base_gpus'] + '×GB300, NVFP4 on ' + F['cp.gpus'] + '×GB300');
  if (!FR || !FR.curves || FR.curves.length !== 5) {
    K.mono(svg, 0, 100, 'Figure 16 values are not available.', { size: 14, color: C.ink2 });
    return;
  }
  var CV = FR.curves, CONC = FR.concurrency;
  var I64 = CONC.indexOf(64), I8 = CONC.indexOf(8), I1 = CONC.indexOf(1);
  var NAME = ['FP8 CP8', '+PDD', '+DCP', '+DSpark', '+NVFP4 CP4'];
  var r0 = function (v) { return (+v).toFixed(0); };
  var P = function (k, i) { return CV[k].pts[i]; };

  /* the step's headline number: the paper's own figure where it gives one (exact),
     else a ratio of read values (approximate, so it says "about"):
     +PDD is interactivity at 64 requests, 24.4 / 5.6 = 4.4, which does not depend
     on how many GPUs run prefill; +NVFP4 is per GPU at 8 requests, 235 / 118 = 2.0 */
  var BIG = [
    null,
    [CH.fmt(P(1, I64)[0] / P(0, I64)[0], 1) + '×', 'interactivity at 64 requests', true],
    ['+' + F['cp.dcp_gain'] + '%', 'interactivity at 8 requests', false],
    ['+' + F['spec.gain'] + '%', 'interactivity at 8 requests', false],
    [CH.fmt(P(4, I8)[1] / P(3, I8)[1], 1) + '×', 'per GPU at 8 requests', true]
  ];
  var SAY = [
    'The baseline: the FP8 model on an off-the-shelf context-parallel setup across ' +
      CV[0].gpus + ' GB300s. Each curve runs from 64 concurrent requests on the left down to 1.',
    'Prefill-decode disaggregation stops prefill from interrupting decode. At 64 requests each ' +
      'request goes from about ' + r0(P(0, I64)[0]) + ' to ' + r0(P(1, I64)[0]) + ' tok/s. Per GPU ' +
      'counts ' + CV[1].gpus + ' GPUs; the paper does not say if prefill runs on extra ones.',
    'Custom kernels for decode context parallelism (Figure 17): +' + F['cp.dcp_gain'] +
      '% interactivity at 8 requests, about ' + r0(P(1, I8)[0]) + ' to ' + r0(P(2, I8)[0]) + ' tok/s per request.',
    'The DSpark speculator drafts tokens that ML4 verifies in blocks: +' + F['spec.gain'] +
      '% interactivity at 8 requests, about ' + r0(P(2, I8)[0]) + ' to ' + r0(P(3, I8)[0]) + ' tok/s per request.',
    'NVFP4 fits ML4 on ' + CV[4].gpus + ' GB300s instead of ' + CV[3].gpus + '. Each request runs at the ' +
      'same speed (' + r0(P(3, I8)[0]) + ' and ' + r0(P(4, I8)[0]) + ' tok/s at 8 requests), so output per GPU ' +
      'doubles, ' + r0(P(3, I8)[1]) + ' to ' + r0(P(4, I8)[1]) + ': the same work on half the GPUs.'
  ];

  // the plot ends at x 620 so the centred '200' tick label stays inside 640
  var X0 = 54, XW = 566, XE = X0 + XW;
  var ax = CH.axes(svg, X0, 62, XW, 258, { x0: 0, x1: 200, y0: 5, y1: 1000, logy: true,
    xt: [0, 50, 100, 150, 200], yt: [10, 100, 1000], yf: function (v) { return CH.comma(v); },
    xl: 'decode interactivity, tok/s per request', yl: 'output throughput, tok/s per GPU · log scale' });
  // unlabelled minor gridlines, so the log scale reads as one
  [20, 30, 50, 200, 300, 500].forEach(function (v) {
    ax.g.appendChild(n('line', { x1: X0, x2: XE, y1: ax.sy(v), y2: ax.sy(v), stroke: C.line2,
      'stroke-dasharray': '2 4' }));
  });
  var swG = CH.body(svg);       // the switcher goes first in the DOM, the curves draw over the grid
  var body = CH.body(svg);
  var keys = CH.body(svg);
  var ring = n('circle', { r: 9, fill: 'none', stroke: C.ink, 'stroke-width': 1.6, opacity: 0 });
  svg.appendChild(ring);
  var S = 0, out = null, sel = -1;
  function px(q) { return [ax.sx(q[0]), ax.sy(q[1])]; }
  // read one point of the amber curve into the readout line and ring it
  function read(i) {
    var q = CV[S].pts[i], p = px(q), cc = CONC[i];
    sel = i;
    ring.setAttribute('cx', p[0]);
    ring.setAttribute('cy', p[1]);
    ring.setAttribute('opacity', 1);
    out.set(NAME[S] + ' · ' + cc + (cc === 1 ? ' request' : ' requests') + ': about ' + r0(q[0]) +
      ' tok/s per request, ' + r0(q[1]) + ' tok/s per GPU');
  }

  function draw(s) {
    S = s;
    sel = -1;
    ring.setAttribute('opacity', 0);
    CH.clear(body);
    CH.clear(keys);
    var role = function (k) { return k === s ? 0 : k === s - 1 ? 1 : 2; };
    var COL = [C.amber, C.blue, CH.GREY], WID = [3, 2.2, 1.6];
    // grey context first, then the comparison, then the focus on top
    [2, 1, 0].forEach(function (rk) {
      for (var k = 0; k <= s; k++) {
        if (role(k) !== rk) continue;
        CH.line(body, ax, CV[k].pts, { color: COL[rk], w: WID[rk] });
        CH.dots(body, ax, CV[k].pts, { color: COL[rk], r: rk ? 3.4 : 4.6 });
      }
    });
    // the key; hardware only where the caption states it (the baseline and NVFP4)
    NAME.forEach(function (nm, k) {
      var on = k <= s, x = (k % 3) * 214, y = 392 + Math.floor(k / 3) * 22;
      var hw = k === 0 || k === 4 ? ' · ' + CV[k].gpus + ' GPUs' : '';
      CH.key(keys, x, y, nm + hw, { color: on ? COL[role(k)] : CH.GREY2,
        dash: on ? null : '2 3', dot: on, text: on ? C.ink2 : C.ink3 });
    });
    // concurrency labels on the focus curve
    var cur = CV[s].pts, a = px(cur[I64]), b = px(cur[I8]), c = px(cur[I1]);
    if (a[0] > 100) K.mono(body, a[0] - 10, a[1] + 4.5, '64', { size: 12.5, anchor: 'end', color: C.ink2 });
    else K.mono(body, a[0], a[1] - 11, '64', { size: 12.5, anchor: 'middle', color: C.ink2 });
    K.mono(body, b[0] + 12, b[1] - 9, '8', { size: 12.5, color: C.ink2 });
    K.mono(body, c[0] + 10, c[1] + 4.5, '1', { size: 12.5, color: C.ink2 });
    // the step's gain, from the previous curve to this one
    if (s > 0) {
      var at = s === 1 ? I64 : I8, p0 = px(CV[s - 1].pts[at]), p1 = px(cur[at]);
      var dx = p1[0] - p0[0], dy = p1[1] - p0[1], L = Math.sqrt(dx * dx + dy * dy), u = 8 / L;
      K.arrow(body, p0[0] + dx * u, p0[1] + dy * u, p1[0] - dx * u, p1[1] - dy * u,
        { color: C.ink, w: 1.6, headSize: 7 });
      // local helper (K.big has no prefix): a small 'about' before a read ratio
      var bt = n('text', { x: XE, y: 104, 'text-anchor': 'end', 'font-family': "'Sora',system-ui,sans-serif",
        'font-weight': 700, fill: C.ink });
      if (BIG[s][2]) {
        var ab = n('tspan', { 'font-size': 17, 'font-weight': 500, fill: C.ink2 });
        ab.textContent = 'about ';
        bt.appendChild(ab);
      }
      var bn = n('tspan', { 'font-size': 40, 'letter-spacing': '-.03em' });
      bn.textContent = BIG[s][0];
      bt.appendChild(bn);
      body.appendChild(bt);
      K.mono(body, XE, 126, BIG[s][1], { size: 12.5, anchor: 'end', color: C.ink2 });
    }
    out = CH.readout(body, 0, 446, { size: 13, text: 'Hover the chart to read the nearest amber point.' });
    K.callout(body, 0, 460, 640, SAY[s], { cols: 74 });
  }

  /* Local helper (the kit's hot target is a fixed circle or rect): one hover area
     over the whole plot that reads the amber point nearest the pointer. It stays
     put while the curves change, so it never goes stale. Arrow keys step along. */
  var pad = n('rect', { x: X0, y: 62, width: XW, height: 258, fill: 'transparent', tabindex: 0,
    'data-hover': '1', 'aria-label': 'read the amber curve, point by point', style: 'cursor:crosshair;outline:none' });
  function near(evt) {
    var m = pad.getScreenCTM();
    if (!m) return;
    var pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    var l = pt.matrixTransform(m.inverse()), best = 0, bd = 1e9;
    CV[S].pts.forEach(function (q, i) {
      var p = px(q), d = (p[0] - l.x) * (p[0] - l.x) + (p[1] - l.y) * (p[1] - l.y);
      if (d < bd) { bd = d; best = i; }
    });
    if (best !== sel) read(best);
  }
  ['mouseenter', 'mousemove', 'click'].forEach(function (ev) { pad.addEventListener(ev, near); });
  pad.addEventListener('focus', function () { read(I8); });
  pad.addEventListener('keydown', function (e) {
    var d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    read(Math.max(0, Math.min(CONC.length - 1, (sel < 0 ? I8 : sel) + d)));
  });

  K.switcher(swG, 0, 0, NAME, draw, { tint: 'amber' });
  svg.appendChild(pad);           // last, so nothing covers it
  K.foot(svg, 'Values are read from the raster figure and are approximate, to about 1 tok/s per request ' +
    'and 3 tok/s per GPU. Numbers on the amber curve are concurrent requests.');
};
