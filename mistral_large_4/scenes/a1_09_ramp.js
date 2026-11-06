/* S_RAMP: the batch-size ramp, 40M then 80M then 160M tokens per batch. The last
   stage's 50% and the one before's 25% are the paper's; with three stages the
   first must take the remaining 25%, which is my inference and is labelled. */
window.SCENES.S_RAMP = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var B = [F['batch.b1_m'], F['batch.b2_m'], F['batch.b3_m']];
  var last = F['batch.last_share'], pen = F['batch.penult_share'], first = 100 - last - pen;
  var EDGE = [0, first, first + pen, 100];
  var svg = K.board(host, { alt: 'Batch size against training progress: ' + B.join(', ') + ' million tokens' });
  K.head(svg, 'Small batches early, big late', 'batch-size ramp · ' + F['batch.chosen_from'] + '× to ' +
    F['batch.chosen_to'] + '× the predicted batch');

  var ax = CH.axes(svg, 54, 20, 570, 236, { x0: 0, x1: 100, y0: 0, y1: 180,
    xt: [0, 25, 50, 75, 100], yt: [0, B[0], B[1], B[2]], xf: function (v) { return v + '%'; },
    yf: function (v) { return v ? v + 'M' : '0'; }, xl: 'share of training tokens', yl: 'batch, tokens' });
  var pts = [];
  B.forEach(function (b, i) { pts.push([EDGE[i], b], [EDGE[i + 1], b]); });
  CH.line(svg, ax, pts, { color: C.amber, w: 3 });
  var SH = [first, pen, last];
  B.forEach(function (b, i) {
    var mid = (EDGE[i] + EDGE[i + 1]) / 2;
    K.mono(svg, ax.sx(mid), ax.sy(b) - 12, SH[i] + '% of tokens' + (i === 0 ? '*' : ''),
      { size: 13, anchor: 'middle', color: C.ink2 });
  });

  // cursor, driven by the slider
  var cur = K.n('path', { d: '', stroke: C.ink, 'stroke-width': 1.2, 'stroke-dasharray': '3 3' });
  svg.appendChild(cur);
  var dot = K.n('circle', { r: 7, fill: C.amber, stroke: '#fff', 'stroke-width': 2.2 });
  svg.appendChild(dot);
  var out = CH.readout(svg, 54, 360, { size: 14 });
  function at(v) {
    var s = v < EDGE[1] ? 0 : v < EDGE[2] ? 1 : 2, b = B[s];
    // the cursor stops at the step, so it never runs through the share labels above it
    cur.setAttribute('d', 'M' + ax.sx(v) + ' ' + (ax.y + ax.h) + ' V' + ax.sy(b));
    dot.setAttribute('cx', ax.sx(v));
    dot.setAttribute('cy', ax.sy(b));
    out.set(Math.round(v) + '% of the way through: ' + b + 'M tokens per batch, ' +
      (b / B[0]) * F['batch.chosen_from'] + '× the predicted size');
  }
  CH.slider(svg, 54, 324, 570, { min: 0, max: 100, step: 1, value: 37, label: 'training progress',
    fmt: function (v) { return v + '% of training tokens'; }, onChange: at });

  // the trade: best for optimization against chosen for wall clock
  var mx = CH.scale(Math.log2(0.25), Math.log2(4), 210, 630);
  var m = function (v) { return mx(Math.log2(v)); };
  K.label(svg, 0, 412, 'ramp, as multiples of the predicted batch');
  [0.25, 0.5, 1, 2, 4].forEach(function (v) {
    svg.appendChild(K.n('path', { d: 'M' + m(v) + ' 424 V496', stroke: C.line2 }));
    K.mono(svg, m(v), 514, v + '×', { size: 12.5, anchor: 'middle', color: C.ink3 });
  });
  var ROW = [['best for optimization', F['batch.opt_from'], F['batch.opt_to'], C.blue],
    ['chosen, for wall clock', F['batch.chosen_from'], F['batch.chosen_to'], C.amber]];
  ROW.forEach(function (r, i) {
    var y = 442 + i * 36;
    K.mono(svg, 0, y + 5, r[0], { size: 13, color: C.ink, weight: 600 });
    svg.appendChild(K.n('path', { d: 'M' + m(r[1]) + ' ' + y + ' H' + m(r[2]), stroke: r[3],
      'stroke-width': 9, 'stroke-linecap': 'round' }));
  });

  K.foot(svg, '*The paper gives 50% for the last stage and 25% for the one before it; with three ' +
    'stages, the first takes the remaining 25%. Bigger batches run at higher MFU, which is why ' +
    'the chosen ramp starts higher.');
};
