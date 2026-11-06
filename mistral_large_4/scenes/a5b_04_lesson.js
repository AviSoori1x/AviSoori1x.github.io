/* S_LESSON: the conclusions. Top, the paper's main lesson on two strips read off
   the Figure 5 fit lines (window.ML4.figread.fig5): at one perplexity the five
   attention variants sit close together on MMLU Redux after pretraining and far
   apart on SWE-bench Verified after midtraining. Both strips share one scale, so
   the spreads compare honestly. The left panel's x axis is monorepo perplexity.
   MLA, full attention, is the reference design (amber); SWA x CLA x YOCO was the
   first ML4 design (blue). Bottom, the conclusions as five hoverable tiles. */
window.SCENES.S_LESSON = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts, n = K.n;
  var FR = (api.D.figread || {}).fig5;
  var svg = K.board(host, { alt: 'Equal perplexity, unequal agentic scores, and the paper\'s conclusions' });
  K.head(svg, 'Equal loss is not equal ability', 'conclusions · Figure 5 fit lines · five attention variants');
  if (!FR || !FR.left || !FR.right) {
    K.mono(svg, 0, 100, 'Figure 5 values are not available.', { size: 14, color: C.ink2 });
    return;
  }

  var SER = ['MLA', 'SWA', 'SWA x YOCO', 'SWA x CLA', 'SWA x CLA x YOCO'];
  var REF = 'MLA', FIRST = 'SWA x CLA x YOCO';
  function at(panel, px) {
    var tab = panel.y_on_fit_at_equal_perplexity;
    return SER.map(function (s) {
      var e = (tab[s] || []).filter(function (q) { return q.x === px && !q.extrapolated; })[0];
      return e ? { s: s, y: e.y } : null;
    }).filter(Boolean);
  }
  // one scale for both strips: 12 points of score across the same width
  var WIN = 12, X0 = 40, X1 = 600;
  var STRIP = [
    { head: 'MMLU Redux · pretraining · monorepo perplexity 2.4', v: at(FR.left, 2.4), lo: 59, ticks: [60, 65, 70] },
    { head: 'SWE-bench Verified · midtraining · validation perplexity 1.41', v: at(FR.right, 1.41), lo: 36,
      ticks: [40, 45] }
  ];
  K.label(svg, 0, 12, 'same loss, two exams');
  var spread = [];
  STRIP.forEach(function (st, i) {
    var y0 = 44 + i * 92, yd = y0 + 26, sx = CH.scale(st.lo, st.lo + WIN, X0, X1);
    var ys = st.v.map(function (d) { return d.y; });
    var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
    spread.push(hi - lo);
    K.mono(svg, 0, y0, st.head, { size: 13, color: C.ink, weight: 600 });
    K.mono(svg, 640, y0, 'spread ' + CH.fmt(hi - lo, 1) + ' pts', { size: 13, anchor: 'end', color: C.ink, weight: 600 });
    svg.appendChild(n('path', { d: 'M' + X0 + ' ' + yd + ' H' + X1, stroke: C.line2, 'stroke-width': 1.2 }));
    st.ticks.forEach(function (t) {
      svg.appendChild(n('path', { d: 'M' + sx(t) + ' ' + (yd - 5) + ' V' + (yd + 5), stroke: C.ink3 }));
      K.mono(svg, sx(t), yd + 36, t + '%', { size: 12.5, anchor: 'middle', color: C.ink3 });
    });
    // the spread, as a bracket under the dots
    svg.appendChild(n('path', { d: 'M' + sx(lo) + ' ' + (yd + 10) + ' V' + (yd + 15) + ' H' + sx(hi) + ' V' + (yd + 10),
      fill: 'none', stroke: C.ink2, 'stroke-width': 1.3 }));
    /* Local dodge (a small beeswarm): amber and blue sit on the line; a grey dot
       that would touch an already placed dot lifts 12px above the line, so all
       five variants stay visible. The lift is position only, not a value. */
    var placed = [];
    st.v.filter(function (d) { return d.s === REF || d.s === FIRST; })
      .concat(st.v.filter(function (d) { return d.s !== REF && d.s !== FIRST; })
        .sort(function (a, b) { return a.y - b.y; }))
      .forEach(function (d) {
        var big = d.s === REF || d.s === FIRST, r = big ? 7 : 5.5, x = sx(d.y), dy = 0;
        [0, -12].some(function (o) {
          dy = o;
          return placed.every(function (q) {
            var ex = q.x - x, ey = q.dy - o;
            return Math.sqrt(ex * ex + ey * ey) >= q.r + r + 1.5;
          });
        });
        placed.push({ x: x, dy: dy, r: r, s: d.s });
      });
    // context first, then the first design, then the reference on top
    placed.sort(function (a, b) {
      var rk = function (d) { return d.s === REF ? 2 : d.s === FIRST ? 1 : 0; };
      return rk(a) - rk(b);
    }).forEach(function (d) {
      var col = d.s === REF ? C.amber : d.s === FIRST ? C.blue : CH.GREY;
      svg.appendChild(n('circle', { cx: d.x, cy: yd + d.dy, r: d.r, fill: col, stroke: '#fff', 'stroke-width': 1.8 }));
    });
  });
  CH.key(svg, 0, 236, 'MLA, full attention', { color: C.amber });
  CH.key(svg, 190, 236, 'SWA × CLA × YOCO, first design', { color: C.blue });
  CH.key(svg, 462, 236, 'other SWA variants', { color: CH.GREY });

  K.callout(svg, 0, 254, 640, 'At one perplexity the variants sit ' + CH.fmt(spread[0], 1) + ' points apart on ' +
    'MMLU Redux and ' + CH.fmt(spread[1], 1) + ' apart on SWE-bench Verified. Comparable loss does not ' +
    'imply comparable long-horizon ability, and better proxies must be found.', { cols: 74 });

  // the conclusions, one tile each
  var CHAP = [
    ['architecture', 'Designed to cut the KV cache ' + F['kv.reduction'] + '×, which cost it on agentic tasks.',
      'Surgery moved it to almost full attention (YOCO kept), then to DSA.'],
    ['post-training', 'Agents author verifiable environments, and large-scale RL trains on them,',
      'at peak ' + F['rlinf.rollouts_k'] + 'k concurrent rollouts on about ' + F['rlinf.sandboxes_k'] + 'k sandboxes.'],
    ['serving', 'A ' + F['q.gb'] + ' GB mixed-precision checkpoint deploys on ' + F['cp.gpus'] + ' GB300 GPUs.',
      'Multi-teacher distillation gives high, medium and low reasoning effort.'],
    // the main lesson is the callout above; this tile adds its two qualifiers (§3.1)
    ['the lesson', 'Knowledge benchmarks did not show the same gap as agentic ones,',
      'and the ablations suggest longer training can shrink the penalty.'],
    ['what is next', 'Sparse, linear and hybrid attention, with agentic evals guiding design.',
      'The aim: train each model on environments authored by the one before.']
  ];
  K.label(svg, 0, 370, 'the conclusions · hover a tile');
  var TW = (640 - 4 * 8) / 5, tiles = [];
  CHAP.forEach(function (c, i) {
    var x = i * (TW + 8);
    var r = n('rect', { x: x, y: 382, width: TW, height: 42, rx: 8, fill: '#fff', stroke: C.line });
    svg.appendChild(r);
    var t = K.mono(svg, x + TW / 2, 407.5, c[0], { size: 13, anchor: 'middle', color: C.ink2 });
    tiles.push([r, t]);
  });
  var o1 = CH.readout(svg, 0, 454, { size: 13, color: C.ink });
  var o2 = CH.readout(svg, 0, 474, { size: 13, color: C.ink });
  function show(i) {
    tiles.forEach(function (tt, j) {
      var on = j === i;
      tt[0].setAttribute('fill', on ? K.TINT.amber : '#fff');
      tt[0].setAttribute('stroke', on ? C.amber : C.line);
      tt[0].setAttribute('stroke-width', on ? 1.8 : 1);
      tt[1].setAttribute('fill', on ? C.ink : C.ink2);
      tt[1].setAttribute('font-weight', on ? 600 : 400);
    });
    o1.set(CHAP[i][1]);
    o2.set(CHAP[i][2]);
  }
  // hover targets last, so nothing covers them
  CHAP.forEach(function (c, i) {
    CH.hot(svg, 0, 0, 0, function () { show(i); }, { rect: [i * (TW + 8), 382, TW, 42], label: c[0] });
  });
  show(0);

  K.foot(svg, 'Strip values are read off the Figure 5 fit lines in the raster figure and are approximate. ' +
    'Both strips share one scale, 12 points wide. A grey dot drawn above its line is only lifted so it ' +
    'stays visible.');
};
