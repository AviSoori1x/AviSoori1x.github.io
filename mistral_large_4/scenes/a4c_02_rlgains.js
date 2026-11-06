/* S_RLGAINS: Figure 13, six evaluations across SFT and then CISPO. Exact values,
   extracted from the vector figure (window.ML4.figread.fig13). As in the paper the
   two phases share one axis: SFT 0% to 100%, then CISPO 0% to 100%. */
window.SCENES.S_RLGAINS = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts, FR = (api.D.figread || {}).fig13;
  var svg = K.board(host, { alt: 'Figure 13: six evaluation scores through SFT and then CISPO training' });
  K.head(svg, 'What RL bought', 'Figure 13 · ' + F['a4c.six'] + ' evaluations · SFT, then the CISPO phase');
  if (!FR || !FR.length) { K.mono(svg, 0, 40, 'Figure 13 values are not available.', { size: 14 }); return; }
  var SHORT = ['Knowledge work', 'Instruction following', 'Scientific chart QA',
    'Terminal tasks', 'Competitive code', 'Agentic CAD QA'];
  var CAD = 5;   // the one panel whose domain is absent from the reward mixture (§4.4.4)
  var r1 = function (v) { return Math.round(v * 10) / 10; };

  // one axis for both phases: SFT at 0..100, CISPO at 100..200
  var P = FR.map(function (p) {
    var sft = [], rl = [];
    p.points.forEach(function (q) {
      if (q.phase === 'SFT') sft.push([q.frac * 100, q.v]); else rl.push([100 + q.frac * 100, q.v]);
    });
    var rawA = sft[sft.length - 1][1], rawB = rl[rl.length - 1][1];
    // the gain is taken between the unrounded end values (last SFT checkpoint to the
    // end of CISPO), so it can differ by 0.1 from the difference of the shown ends
    return { p: p, sft: sft, rl: rl, a: r1(rawA), b: r1(rawB), gain: r1(rawB - rawA),
      all: sft.concat(rl).map(function (q) { return q[1]; }) };
  });

  function series(g, ax, d, o) {
    o = o || {};
    CH.line(g, ax, d.sft, { color: CH.GREY, w: o.w || 2 });
    // the paper draws the hand-over from the last SFT point in the CISPO colour
    CH.line(g, ax, [d.sft[d.sft.length - 1]].concat(d.rl), { color: C.amber, w: o.w || 2.2 });
    if (o.dots) {
      CH.dots(g, ax, d.sft, { color: CH.GREY, r: 3.6 });
      CH.dots(g, ax, d.rl, { color: C.amber, r: 3.8 });
    }
  }
  function range(d, padFrac) {
    var lo = Math.min.apply(null, d.all), hi = Math.max.apply(null, d.all), pad = (hi - lo) * padFrac;
    return [lo - pad, hi + pad];
  }

  // six tiles: the overview, and the panel picker
  var TW = 200, TH = 78, tiles = [];
  P.forEach(function (d, i) {
    var x = (i % 3) * (TW + 20), y = Math.floor(i / 3) * (TH + 14);
    var g = K.n('g', { tabindex: 0, role: 'button', 'aria-label': d.p.title, style: 'cursor:pointer;outline:none' });
    svg.appendChild(g);
    var bg = K.n('rect', { x: x, y: y, width: TW, height: TH, rx: 8, fill: '#fff', stroke: C.line });
    g.appendChild(bg);
    K.mono(g, x + 12, y + 21, SHORT[i], { size: 13, weight: 600, color: C.ink });
    var rg = range(d, 0.08);
    var ax = { sx: CH.scale(0, 200, x + 12, x + TW - 62), sy: CH.scale(rg[0], rg[1], y + TH - 10, y + 32) };
    g.appendChild(K.n('path', { d: 'M' + ax.sx(100) + ' ' + (y + 30) + ' V' + (y + TH - 8),
      stroke: C.ink3, 'stroke-width': 1, 'stroke-dasharray': '2 3', opacity: 0.6 }));
    series(g, ax, d, { w: 1.7 });
    K.mono(g, x + TW - 10, y + 58, (d.gain >= 0 ? '+' : '') + CH.fmt(d.gain, 1), { size: 13.5,
      anchor: 'end', weight: 600, color: C.ink });
    function go() { tiles.forEach(function (t, j) {
      t.bg.setAttribute('fill', j === i ? K.TINT.amber : '#fff');
      t.bg.setAttribute('stroke', j === i ? C.amber : C.line);
      t.bg.setAttribute('stroke-width', j === i ? 2 : 1);
    }); draw(i); }
    g.addEventListener('click', go);
    g.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); }
    });
    tiles.push({ g: g, bg: bg, go: go });
  });

  // the selected panel, large, with every point
  var body = CH.body(svg);
  // reserve the tallest callout, so the foot (placed once) never lands on it
  svg.appendChild(K.n('rect', { x: 0, y: 474, width: 640, height: 3 * 19 + 22, fill: 'none' }));
  // a background-coloured stroke behind a label, so lines passing under it stay legible
  function halo(t) {
    t.setAttribute('stroke', C.bg); t.setAttribute('stroke-width', 4);
    t.setAttribute('paint-order', 'stroke'); t.setAttribute('stroke-linejoin', 'round');
    return t;
  }
  function thin(ticks, lo, hi) {
    var ok = ticks.filter(function (t) { return t >= lo && t <= hi; });
    var k = Math.ceil(ok.length / 4);
    return ok.filter(function (t, j) { return j % k === 0; });
  }
  /* Centre an end value directly above or below its own dot, whichever side the
     drawn lines and other dots leave free (a pixel test on the line samples). */
  function endLabel(ax, d, e, s, col) {
    var px = ax.sx(e[0]), py = ax.sy(e[1]), w = s.length * 13 * 0.62 + 4;
    var line = d.sft.concat(d.rl).map(function (q) { return [ax.sx(q[0]), ax.sy(q[1])]; });
    function cost(base) {
      var x0 = px - w / 2, x1 = px + w / 2, y0 = base - 12, y1 = base + 4, c = 0;
      for (var k = 1; k < line.length; k++) {
        for (var t = 0; t <= 1; t += 0.05) {
          var qx = line[k - 1][0] + t * (line[k][0] - line[k - 1][0]);
          var qy = line[k - 1][1] + t * (line[k][1] - line[k - 1][1]);
          if (qx > x0 && qx < x1 && qy > y0 && qy < y1) c++;
        }
      }
      if (y1 > ax.y + ax.h + 6) c += 50;   // keep clear of the x tick labels
      return c;
    }
    var up = py - 12, dn = py + 23, base = cost(up) <= cost(dn) ? up : dn;
    halo(K.mono(body, px, base, s, { size: 13, anchor: 'middle', weight: 600, color: col }));
  }
  function draw(i) {
    CH.clear(body);
    var d = P[i], rg = range(d, 0.1);
    // 570 wide, so the last tick label ("100%", centred on x = 620) ends inside 640
    var ax = CH.axes(body, 50, 230, 570, 178, { x0: 0, x1: 200, y0: rg[0], y1: rg[1],
      xt: [0, 50, 100, 150, 200], yt: thin(d.p.ticks, rg[0], rg[1]),
      xf: function (v) { return (v > 100 ? v - 100 : v) + '%'; }, yl: d.p.metric });
    body.appendChild(K.n('path', { d: 'M' + ax.sx(100) + ' 230 V408', stroke: C.ink3,
      'stroke-width': 1.2, 'stroke-dasharray': '3 4' }));
    series(body, ax, d, { w: 2.4, dots: true });
    var e1 = d.sft[d.sft.length - 1], e2 = d.rl[d.rl.length - 1];
    // the two end points the numbers refer to are drawn larger than the rest
    CH.dots(body, ax, [e1], { color: CH.GREY, r: 5.6 });
    CH.dots(body, ax, [e2], { color: C.amber, r: 5.6 });
    endLabel(ax, d, e1, CH.fmt(d.a, 1), C.ink2);
    endLabel(ax, d, e2, CH.fmt(d.b, 1), C.ink);
    CH.key(body, ax.sx(50) - 22, 452, 'SFT', { color: CH.GREY });
    CH.key(body, ax.sx(150) - 30, 452, 'CISPO', { color: C.amber });
    var unit = d.p.metric.indexOf('%') >= 0 ? '%' : '';
    // the last plotted SFT checkpoint is at 100% of SFT except in the CAD panel (90%)
    var at = 'at the last SFT checkpoint' + (e1[0] < 99.5 ? ' (' + Math.round(e1[0]) + '% of SFT)' : '');
    var line = i === CAD ?
      d.p.title + ' is not in the reward mixture at all, yet it rose from ' + CH.fmt(d.a, 1) + unit +
        ' ' + at + ' to ' + CH.fmt(d.b, 1) + unit + ' at the end of CISPO.' :
      d.p.title + ': ' + CH.fmt(d.a, 1) + unit + ' ' + at + ', ' + CH.fmt(d.b, 1) + unit +
        ' at the end of CISPO, a gain of ' + CH.fmt(d.gain, 1) + ' points.';
    K.callout(body, 0, 474, 640, line, { cols: 74 });
  }

  tiles[CAD].go();
  // two lines at the foot's 88 columns, so it clears the page footer bar. No line may
  // start with "T": the board is fitted before the web fonts load, and the fallback
  // font's T overhangs x = 0 by about a unit, which widened the board.
  K.foot(svg, 'Exact values from the vector figure; tile gains use unrounded values, from the last SFT ' +
    'checkpoint to the end of CISPO. Each panel has its own y scale, as in the paper.');
};
