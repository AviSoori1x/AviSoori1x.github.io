/* S_EQUALLOSS: Figure 5, two panels. Left: MMLU Redux against Monorepo perplexity in
   pretraining, where all five designs fall on one trend. Right: SWE-bench Verified
   after midtraining against validation perplexity, where they split, and where the
   2x sliding-window run only gets further along its own line. Every dot and fit
   is a pixel read (window.ML4.figread.fig5), so approximate. MLA = full attention. */
window.SCENES.S_EQUALLOSS = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, FR = (api.D.figread || {}).fig5;
  var svg = K.board(host, { alt: 'Figure 5: benchmark score against perplexity for five attention designs, in pretraining and after midtraining' });
  K.head(svg, 'Same loss, different agent', 'Figure 5 · one dot per checkpoint · a fitted line per design');
  var body = CH.body(svg);
  if (!FR || !FR.left) {
    K.mono(body, 0, 100, 'Figure 5 values are not available.', { size: 14, color: C.ink2 });
    return;
  }
  var byName = function (P, nm) { return P.series.filter(function (s) { return s.name === nm; })[0]; };
  var onFit = function (P, nm, x) {
    return P.y_on_fit_at_equal_perplexity[nm].filter(function (q) { return q.x === x; })[0].y;
  };
  // one panel per state: data, axis ranges, and the perplexity where the fits are compared
  var PANEL = [
    { P: FR.left, tag: 'pretraining', at: 2.4, d: 1,
      o: { x0: 2.15, x1: 2.65, y0: 50, y1: 80, xt: [2.2, 2.3, 2.4, 2.5, 2.6], yt: [50, 60, 70, 80] } },
    { P: FR.right, tag: 'after midtraining', at: 1.4, d: 2,
      o: { x0: 1.365, x1: 1.47, y0: 20, y1: 60, xt: [1.38, 1.40, 1.42, 1.44, 1.46], yt: [20, 30, 40, 50, 60] } }
  ];
  // MLA is full attention, the reference: amber. SWA alone is the comparison: blue.
  var STYLE = {
    'MLA': { c: C.amber, w: 2.6, r: 4 }, 'SWA': { c: C.blue, w: 2.6, r: 4 },
    'MLA (0.5x)': { c: C.amber, w: 2, r: 3.6, dash: '6 4', hollow: true },
    'SWA (2x)': { c: C.blue, w: 2, r: 3.6, dash: '6 4', hollow: true }
  };
  var GREYS = { c: CH.GREY, w: 1.5, r: 3.2, dot: CH.GREY2 };
  var ORDER = ['SWA x YOCO', 'SWA x CLA', 'SWA x CLA x YOCO', 'SWA (2x)', 'MLA (0.5x)', 'SWA', 'MLA'];

  // A key swatch drawn like the plot: a 30px line in the series' dash with its marker in
  // the middle (hollow for the dashed runs). CH.key's 18px line is too short to show a dash.
  function swatch(x, y, name, st) {
    body.appendChild(K.n('path', { d: 'M' + x + ' ' + (y - 4.5) + ' h30', stroke: st.c, 'stroke-width': 2.5,
      'stroke-dasharray': st.dash || null }));
    body.appendChild(K.n('circle', { cx: x + 15, cy: y - 4.5, r: 3.6, fill: st.hollow ? '#fff' : (st.dot || st.c),
      stroke: st.hollow ? st.c : '#fff', 'stroke-width': st.hollow ? 2 : 1.2 }));
    K.mono(body, x + 37, y, name, { size: 12.5, color: C.ink2 });
  }

  // the probe reads and the callout for each panel, computed up front so the callout
  // slot can be sized to the taller of the two
  function info(k) {
    var Q = PANEL[k], P = Q.P;
    var names = Object.keys(P.y_on_fit_at_equal_perplexity).filter(function (nm) {
      return k === 0 || nm === 'MLA' || nm === 'SWA';
    });
    var vals = names.map(function (nm) { return onFit(P, nm, Q.at); });
    if (k === 0) {
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
      return { names: names, vals: vals, msg: 'Pretraining, a knowledge test: at perplexity ' + Q.at.toFixed(2) +
        ' the five fitted lines read ' + CH.fmt(lo, 1) + ' to ' + CH.fmt(hi, 1) + ' on MMLU Redux. ' +
        'At equal loss, the knowledge scores stay close together, whatever the attention design.' };
    }
    // the best checkpoint of each longer or shorter run: the highest dot it reached
    var best = function (nm) {
      return byName(P, nm).pts.reduce(function (a, q) { return q.y > a.y ? q : a; });
    };
    var b2 = best('SWA (2x)'), bh = best('MLA (0.5x)');
    return { names: names, vals: vals, b2: b2, bh: bh,
      // the gap is the difference of the two fits at the probe
      msg: 'After midtraining, an agentic test: at perplexity ' + Q.at.toFixed(2) + ', full attention\'s ' +
        'line reads ' + CH.fmt(vals[0], 1) + ' and sliding windows\' ' + CH.fmt(vals[1], 1) + ', ' +
        CH.fmt(vals[0] - vals[1], 1) + ' apart. SWA (2×) runs further along a line no higher than ' +
        'SWA\'s; its best (' + CH.fmt(b2.y, 1) + ') only draws level with MLA (0.5×)\'s best (' +
        CH.fmt(bh.y, 1) + ').' };
  }
  var COLS = Math.min(74, Math.floor((640 - 32) / (13.5 * 0.625)));
  var SLOT = Math.max(K.wrap(info(0).msg, COLS).length, K.wrap(info(1).msg, COLS).length) * 19 + 22;

  function draw(k) {
    CH.clear(body);
    var Q = PANEL[k], P = Q.P;
    var o = Q.o;
    o.xf = function (v) { return v.toFixed(Q.d); };
    o.xl = P.x_label.toLowerCase();
    o.yl = P.y_label;
    var ax = CH.axes(body, 54, 70, 586, 220, o);
    K.label(body, 640, 56, Q.tag, { anchor: 'end' });

    ORDER.forEach(function (nm) {
      var s = byName(P, nm);
      if (!s) return;
      var st = STYLE[nm] || GREYS;
      CH.line(body, ax, s.fit, { color: st.c, w: st.w, dash: st.dash });
      CH.dots(body, ax, s.pts.map(function (q) { return [q.x, q.y]; }),
        { color: st.dot || st.c, r: st.r, hollow: st.hollow });
    });

    // later checkpoints sit at lower perplexity, so training runs right to left
    K.mono(body, ax.x + 10, ax.y + ax.h - 12, '← later in training',
      { size: 12.5, color: C.ink3 });

    // the equal-perplexity probe
    var px = ax.sx(Q.at);
    body.appendChild(K.n('path', { d: 'M' + px + ' ' + ax.y + ' V' + (ax.y + ax.h), stroke: C.ink,
      'stroke-width': 1.2, 'stroke-dasharray': '4 4' }));
    K.mono(body, px + 8, ax.y + 16, 'same perplexity, ' + Q.at.toFixed(2), { size: 13, color: C.ink, weight: 600 });
    var I = info(k), names = I.names, vals = I.vals;
    if (k === 1) {
      // a bracket between the two fits at the probe, no text: the numbers are in the callout
      body.appendChild(K.n('path', { d: 'M' + (px - 7) + ' ' + ax.sy(vals[0]) + ' h-5 V' + ax.sy(vals[1]) + ' h5',
        fill: 'none', stroke: C.ink, 'stroke-width': 1.4 }));
    }
    names.forEach(function (nm, i) {
      var st = STYLE[nm] || GREYS;
      body.appendChild(K.n('circle', { cx: px, cy: ax.sy(vals[i]), r: st === GREYS ? 4 : 6,
        fill: st === GREYS ? CH.GREY : st.c, stroke: '#fff', 'stroke-width': 2 }));
    });

    if (k === 1) {
      var b2 = I.b2, bh = I.bh;
      [[b2, C.blue], [bh, C.amber]].forEach(function (b) {
        var cx = ax.sx(b[0].x), cy = ax.sy(b[0].y);
        body.appendChild(K.n('circle', { cx: cx, cy: cy, r: 9, fill: 'none', stroke: b[1], 'stroke-width': 2 }));
      });
      K.mono(body, ax.sx(b2.x), ax.sy(b2.y) - 16, 'best ' + CH.fmt(b2.y, 1), { size: 13, anchor: 'middle',
        color: C.ink, weight: 600 });
      K.mono(body, ax.sx(bh.x) + 14, ax.sy(bh.y) + 4.5, 'best ' + CH.fmt(bh.y, 1), { size: 13,
        color: C.ink, weight: 600 });
    }

    // key: the reference, the comparison, the context; the dashed runs on the right panel only
    var KY = 362;
    swatch(0, KY, 'MLA, full attention', STYLE['MLA']);
    swatch(232, KY, 'SWA, sliding windows', STYLE['SWA']);
    swatch(448, KY, 'other SWA variants', GREYS);
    if (k === 1) {
      swatch(0, KY + 22, 'MLA (0.5×), shorter run', STYLE['MLA (0.5x)']);
      swatch(232, KY + 22, 'SWA (2×), longer run', STYLE['SWA (2x)']);
    }
    // both states share one callout slot, sized to the taller message, so the foot
    // (placed once, under the first state) never collides
    K.callout(body, 0, 400, 640, I.msg, { cols: 74, color: k === 1 ? C.blue : C.amber,
      tint: k === 1 ? K.TINT.blue : K.TINT.amber });
    body.appendChild(K.n('rect', { x: 0, y: 400, width: 1, height: SLOT, fill: 'none' }));
  }

  K.switcher(svg, 0, 0, ['knowledge: MMLU Redux', 'agentic: SWE-bench Verified'], draw, { tint: 'ink' });

  K.foot(svg, 'Figure 5 is raster: dots and fitted lines are pixel reads, approximate. As in the paper, ' +
    'each run\'s first two checkpoints are left out. The left axis says Monorepo perplexity, the ' +
    'caption validation perplexity. The paper does not define 0.5× and 2×; we read them as run ' +
    'lengths, since it says longer training softens the penalty.');
};
