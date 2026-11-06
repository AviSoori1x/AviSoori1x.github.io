/* S_YOCOKEPT: Figure 8, the conservative test at 8x10^23 FLOPs: surgery, then
   midtraining only. "SWA and CLA surgery" is the run that KEEPS YOCO, the one
   adopted for the full model, so it is amber; "SWA, CLA, and YOCO surgery"
   removes YOCO as well and is the comparison, blue; the unmodified baseline is
   grey. Raster read, approximate (figread.fig8). */
window.SCENES.S_YOCOKEPT = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts, FR = (api.D.figread || {}).fig8;
  var SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  var sup = function (v) { return String(v).split('').map(function (c) { return SUP[+c]; }).join(''); };
  var svg = K.board(host, { alt: 'Figure 8: SWE-bench Verified against training completion for two surgery runs and the baseline' });
  K.head(svg, 'Keeping YOCO did better', 'Figure 8 · surgery, then midtraining only · 8×10' +
    sup(F['surg.big_exp']) + ' FLOPs');
  if (!FR || !FR.series) {
    K.mono(svg, 0, 100, 'Figure 8 values are not available.', { size: 14, color: C.ink2 });
    return;
  }
  var by = function (name) { return FR.series.filter(function (s) { return s.name === name; })[0]; };
  var RUNS = [
    { s: by('SWA and CLA surgery'), col: C.amber, tag: 'keeps YOCO' },
    { s: by('SWA, CLA, and YOCO surgery'), col: C.blue, tag: 'removes YOCO' },
    { s: by('Baseline'), col: CH.GREY, tag: 'baseline' }
  ];
  var P = RUNS.map(function (r) { return r.s.pts.map(function (q) { return [q.x, q.y]; }); });
  var last = function (i) { return P[i][P[i].length - 1]; };
  var at = function (i, x) { var h = P[i].filter(function (q) { return q[0] === x; })[0]; return h ? h[1] : null; };
  var lowest = function (i, xmax) { return P[i].filter(function (q) { return q[0] <= xmax; })
    .reduce(function (a, q) { return q[1] < a[1] ? q : a; }); };
  var highest = function (i, xmax) { return P[i].filter(function (q) { return q[0] <= xmax; })
    .reduce(function (a, q) { return q[1] > a[1] ? q : a; }); };
  // from 25% on, at checkpoints both surgery runs share: how often keeping YOCO is ahead
  var shared = P[0].filter(function (q) { return q[0] >= 25 && at(1, q[0]) != null; });
  var ahead = shared.filter(function (q) { return q[1] > at(1, q[0]); }).length;

  var ax = CH.axes(svg, 46, 72, 578, 280, { x0: 0, x1: 100, y0: 15, y1: 65,
    xt: [0, 20, 40, 60, 80, 100], yt: [20, 30, 40, 50, 60], xf: function (v) { return v + '%'; },
    xl: 'training completion', yl: 'SWE-bench Verified, %' });
  var plot = CH.body(svg), hots = CH.body(svg), note = CH.body(svg);
  var HINT = 'Hover the chart to read every run at a checkpoint.';
  var out = CH.readout(svg, 0, 424, { size: 13.5, text: HINT });

  var END = [[10, 4, 'start'], [10, 4, 'start'], [0, 24, 'end']];
  var MSG = [
    function () {
      var lo = lowest(0, 20), e = last(0);
      return 'SWA and CLA surgery keeps YOCO. It dips to ' + CH.fmt(lo[1], 1) + ' at ' + lo[0] +
        '%, then is ahead at ' + ahead + ' of ' + shared.length + ' shared checkpoints from 25% on and ' +
        'ends highest, ' + CH.fmt(e[1], 1) + ' at ' + e[0] + '%. The full model kept YOCO.';
    },
    function () {
      var hi = highest(1, 15), e = last(1), k = at(0, hi[0]);
      return 'SWA, CLA, and YOCO surgery removes YOCO as well. It leads early, ' + CH.fmt(hi[1], 1) +
        ' at ' + hi[0] + '% against ' + CH.fmt(k, 1) + ' for keeping YOCO, but from 25% on it is ahead ' +
        'at only ' + (shared.length - ahead) + ' of ' + shared.length + ' shared checkpoints, and it stops at ' + CH.fmt(e[1], 1) + ' at ' + e[0] + '%.';
    },
    function () {
      // the last surgery checkpoint: the later of the two surgery runs' final points
      var e = last(2), stop = Math.max(last(0)[0], last(1)[0]), m = at(2, stop);
      var lowerThanBoth = m != null && m < last(0)[1] && m < last(1)[1];
      return 'The baseline is the unmodified model, the only run drawn to ' + e[0] + '%. At ' + stop +
        '%, where the last surgery run stops, it is at ' + CH.fmt(m, 1) + (lowerThanBoth ?
        ', lower than either surgery run\'s last point. ' : '. ') +
        'At ' + e[0] + '% it is at ' + CH.fmt(e[1], 1) + '.';
    }
  ];

  function draw(f) {
    CH.clear(plot); CH.clear(note);
    if (guide) CH.clear(guide);
    [0, 1, 2].filter(function (i) { return i !== f; }).concat([f]).forEach(function (i) {
      var on = i === f;
      CH.line(plot, ax, P[i], { color: RUNS[i].col, w: on ? 3 : 1.8, opacity: on ? 1 : 0.5 });
      CH.dots(plot, ax, P[i], { color: RUNS[i].col, r: on ? 4.4 : 3, opacity: on ? 1 : 0.5 });
    });
    RUNS.forEach(function (r, i) {
      var e = last(i), d = END[i];
      K.mono(plot, ax.sx(e[0]) + d[0] * (d[2] === 'end' ? 0 : 1), ax.sy(e[1]) + d[1],
        r.tag + ' · ' + CH.fmt(e[1], 1), { size: 13, anchor: d[2], weight: i === f ? 600 : 400,
          color: i === f ? C.ink : C.ink3 });
    });
    out.set(HINT, C.ink);
    K.callout(note, 0, 442, 640, MSG[f](), { cols: 74 });
  }

  var sw = K.switcher(svg, 0, 0, RUNS.map(function (r) { return r.s.name; }), draw, { tint: 'amber' });
  // the run pills first in document order, ahead of the point hovers
  svg.insertBefore(sw.g, plot);
  // one hover column per checkpoint: it reads out every run that has a point there
  var XS = [];
  P.forEach(function (pp) { pp.forEach(function (q) { if (XS.indexOf(q[0]) < 0) XS.push(q[0]); }); });
  XS.sort(function (u, v) { return u - v; });
  var guide = CH.body(svg);
  hots.parentNode.insertBefore(guide, hots);
  XS.forEach(function (x, k) {
    var lo = k ? (XS[k - 1] + x) / 2 : x - 2.5, hi = k < XS.length - 1 ? (x + XS[k + 1]) / 2 : Math.min(100, x + 2.5);
    CH.hot(hots, 0, 0, 0, function () {
      CH.clear(guide);
      guide.appendChild(K.n('path', { d: 'M' + ax.sx(x) + ' 72 V352', stroke: C.ink3, 'stroke-dasharray': '2 3' }));
      var parts = [];
      RUNS.forEach(function (r, i) {
        var y = at(i, x);
        if (y == null) return;
        parts.push(r.tag + ' ' + CH.fmt(y, 1));
        guide.appendChild(K.n('circle', { cx: ax.sx(x), cy: ax.sy(y), r: 7, fill: 'none', stroke: r.col, 'stroke-width': 2 }));
      });
      out.set(x + '% of training · ' + parts.join(' · '), C.ink);
    }, { rect: [ax.sx(lo), 72, ax.sx(hi) - ax.sx(lo), 280], label: x + '% of training' });
  });
  K.foot(svg, 'Figure 8 is a raster image, so values are read from pixels and approximate, to about ' +
    '±0.1 points. The x axis is the figure\'s own training completion; the surgery runs stop near 70%.');
};
