/* S_SFTLAD: §4.2.2 and Figure 11, the SFT learning rate predicted by a scaling ladder.
   (a) the ladder: optimal SFT learning rate against pretraining compute, a law fitted on
   the smaller rungs, the held-out largest model, and the final mixture's shifted law.
   (b) the ablation at ML4 scale: validation loss above its minimum against peak learning
   rate, with the ladder's prediction. Every value is an approximate raster read of the
   figure (window.ML4.figread.fig11); both axes are per 8M-token batch (sftlad.batch_m). */
window.SCENES.S_SFTLAD = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var FR = (api.D.figread || {}).fig11, BM = F['sftlad.batch_m'];
  var svg = K.board(host, { alt: 'Figure 11: the SFT learning rate predicted by a scaling ladder, ' +
    'and a learning-rate ablation at Mistral Large 4 scale' });
  K.head(svg, 'A ladder sets the SFT rate', 'Figure 11 · per ' + BM + 'M-token batch · ' +
    'values read from the figure');
  if (!FR || !FR.a || !FR.b) { K.mono(svg, 0, 40, 'Figure 11 values are not available.', { size: 14 }); return; }

  var X = 48, W = 582, TOP = 122, H = 230;
  // "6.1 × 10^−6": mantissa to d places; rich() raises whatever follows a caret
  function sci(v, d) {
    var e = Math.floor(Math.log10(v) + 1e-9), m = v / Math.pow(10, e);
    return CH.fmt(m, d == null ? 1 : d) + ' × 10^' + String(e).replace('-', '−');
  }
  // set a text's content, turning each ^exponent into a raised tspan (a local kit addition)
  function rich(el, s) {
    while (el.firstChild) el.removeChild(el.firstChild);
    String(s).split(/\^([−0-9]+)/).forEach(function (part, i) {
      if (!part) return;
      if (i % 2) {
        var t = K.n('tspan', { 'baseline-shift': 'super', 'font-size': 12.5 });
        t.textContent = part;
        el.appendChild(t);
      } else el.appendChild(document.createTextNode(part));
    });
    return el;
  }
  // a tick label with a real superscript: base text plus a raised tspan (one text leaf)
  function pow(p, x, y, base, exp) {
    var t = K.mono(p, x, y, base, { size: 14, anchor: 'middle', color: C.ink3 });
    var ts = K.n('tspan', { dy: -6, 'font-size': 12.5 });
    ts.textContent = exp;
    t.appendChild(ts);
    return t;
  }
  // x gridline and superscript label at value v
  function xtick(p, ax, v, base, exp) {
    p.appendChild(K.n('line', { x1: ax.sx(v), x2: ax.sx(v), y1: TOP, y2: TOP + H, stroke: C.line2 }));
    pow(p, ax.sx(v), TOP + H + 23, base, exp);
  }
  // legend swatch: line, dash, dot, focus (big amber dot) or tri, then the name
  function sw(p, x, y, kind, col, name) {
    var cy = y - 4.5;
    if (kind === 'line' || kind === 'dash') {
      p.appendChild(K.n('path', { d: 'M' + x + ' ' + cy + ' h20', stroke: col, 'stroke-width': 2.4,
        'stroke-dasharray': kind === 'dash' ? '5 3' : null, 'stroke-linecap': 'round' }));
    } else if (kind === 'tri') {
      p.appendChild(K.n('path', { d: tri(x + 10, cy + 1, 6.5), fill: col, stroke: '#fff', 'stroke-width': 1.4 }));
    } else {
      p.appendChild(K.n('circle', { cx: x + 10, cy: cy, r: kind === 'focus' ? 6 : 4.4, fill: col,
        stroke: '#fff', 'stroke-width': 1.6 }));
    }
    return K.mono(p, x + 28, y, name, { size: 12.5, color: C.ink2 });
  }
  // an upward triangle centred on (x, y)
  function tri(x, y, r) {
    return 'M' + x + ' ' + (y - r) + ' L' + (x + r * 0.95) + ' ' + (y + r * 0.62) + ' L' +
      (x - r * 0.95) + ' ' + (y + r * 0.62) + ' Z';
  }

  // the switcher's group comes first, so its two buttons lead the tab order
  var ctl = CH.body(svg), body = CH.body(svg), hots = CH.body(svg), out, note;

  // a ring around the point being read; it ignores the mouse, so the hit targets stay on top
  function mkRing() {
    var r = K.n('circle', { r: 9.5, fill: 'none', stroke: C.ink, 'stroke-width': 1.6,
      'pointer-events': 'none', display: 'none' });
    hots.appendChild(r);
    return { at: function (x, y, rad) {
      if (x == null) { r.setAttribute('display', 'none'); return; }
      r.setAttribute('cx', x); r.setAttribute('cy', y); r.setAttribute('r', rad || 9.5);
      r.removeAttribute('display');
    } };
  }

  /* ---------- (a) the scaling ladder ---------- */
  function panelA() {
    var A = FR.a, L = A.law_params;
    // the drawn law: log10(lr) = a + b * log10(FLOPs), read from the figure
    function law(p, f) { return Math.pow(10, p.a + p.b * Math.log10(f)); }
    var ax = CH.axes(body, X, TOP, W, H, { x0: 4e21, x1: 1.6e24, y0: 1.1e-5, y1: 9.5e-5,
      logx: true, logy: true, xt: [], yt: [2e-5, 4e-5, 6e-5, 8e-5],
      yf: function (v) { return String(Math.round(v * 1e5)); },
      // K.label prints capitals, and "FLOPS" would read as per second: say total FLOP instead
      xl: 'pretraining compute, total FLOP' });
    rich(K.label(body, X, TOP - 14, ''), 'OPTIMAL SFT LEARNING RATE, ×10^−5');
    xtick(body, ax, 1e22, '10', '22');
    xtick(body, ax, 1e23, '10', '23');
    xtick(body, ax, 1e24, '10', '24');

    // the key, one row per mixture
    K.label(body, 0, 58, 'initial mixture');
    sw(body, 166, 58, 'line', C.blue, 'fitted law');
    sw(body, 340, 58, 'dot', C.blue, 'measured');
    sw(body, 486, 58, 'focus', C.amber, 'held out');
    K.label(body, 0, 84, 'final mixture');
    sw(body, 166, 84, 'dash', CH.GREY, 'recalibrated law');
    sw(body, 340, 84, 'dot', CH.GREY, 'measured');

    // both laws over the compute range the figure draws them on
    var f0 = A.initial_law[0][0], f1 = A.initial_law[1][0];
    CH.line(body, ax, [[f0, law(L.final, f0)], [f1, law(L.final, f1)]],
      { color: CH.GREY, w: 2, dash: '7 5' });
    CH.line(body, ax, [[f0, law(L.initial, f0)], [f1, law(L.initial, f1)]], { color: C.blue, w: 2.4 });
    // name the shift on the chart: a label in the band between the two laws, past the
    // last fitted rung and short of the held-out one
    var ga = 1.1e23, gb = 5e23, gx = ax.sx(ga), gy = ax.sy(law(L.final, ga));
    var ang = Math.atan2(ax.sy(law(L.final, gb)) - gy, ax.sx(gb) - gx);
    var tx = gx + Math.sin(ang) * 8, ty = gy - Math.cos(ang) * 8;
    K.mono(body, tx, ty, 'shifted by a constant', { size: 12.5, color: C.ink3 })
      .setAttribute('transform', 'rotate(' + (ang * 180 / Math.PI).toFixed(2) + ' ' + tx.toFixed(1) +
        ' ' + ty.toFixed(1) + ')');
    function xy(q) { return [q.flops, q.lr]; }
    CH.dots(body, ax, A.final_measured.map(xy), { color: CH.GREY, r: 4.8 });
    CH.dots(body, ax, A.initial_measured.map(xy), { color: C.blue, r: 4.8 });

    // the held-out rung, and how far it sits past the largest fitted one
    var ho = A.held_out, last = A.initial_measured[A.initial_measured.length - 1];
    var hx = ax.sx(ho.flops), hy = ax.sy(ho.lr), lx = ax.sx(last.flops), ly = ax.sy(last.lr);
    var ratio = Math.round(ho.flops / last.flops);   // derived: held-out over largest fitted compute
    var SY = 176;
    body.appendChild(K.n('path', { d: 'M' + lx + ' ' + (SY - 6) + ' v12 M' + lx + ' ' + SY + ' H' + hx +
      ' M' + hx + ' ' + (SY - 6) + ' v12', stroke: C.ink3, 'stroke-width': 1.3, fill: 'none' }));
    body.appendChild(K.n('path', { d: 'M' + lx + ' ' + (SY + 6) + ' V' + (ly - 9) + ' M' + hx + ' ' +
      (SY + 6) + ' V' + (hy - 12), stroke: C.ink3, 'stroke-width': 1, 'stroke-dasharray': '2 3', fill: 'none' }));
    K.mono(body, (lx + hx) / 2, SY - 11, 'held out at about ' + ratio + '× the compute',
      { size: 13, anchor: 'middle', color: C.ink });
    body.appendChild(K.n('circle', { cx: hx, cy: hy, r: 13, fill: C.amber, opacity: 0.14 }));
    body.appendChild(K.n('circle', { cx: hx, cy: hy, r: 6.4, fill: C.amber, stroke: '#fff', 'stroke-width': 2 }));

    // hover any point to read it
    var base = 'held out at ' + sci(ho.flops) + ' FLOPs: measured ' + sci(ho.lr, 2) +
      ', law predicts ' + sci(A.initial_law_at_held_out_flops, 2);
    out.set(base);
    var ring = mkRing();
    function hov(q, who) {
      return function () {
        ring.at(ax.sx(q.flops), ax.sy(q.lr));
        out.set(who + ' at ' + sci(q.flops) + ' FLOPs: measured optimum ' + sci(q.lr, 2));
      };
    }
    A.initial_measured.forEach(function (q) {
      CH.hot(hots, ax.sx(q.flops), ax.sy(q.lr), 12, hov(q, 'initial mixture'), { label: 'initial mixture rung' });
    });
    A.final_measured.forEach(function (q) {
      CH.hot(hots, ax.sx(q.flops), ax.sy(q.lr), 12, hov(q, 'final mixture'), { label: 'final mixture rung' });
    });
    CH.hot(hots, hx, hy, 14, function () { ring.at(hx, hy, 13); out.set(base); }, { label: 'held-out rung' });
    K.callout(note, 0, 444, 640, 'The law fitted on the smaller rungs predicts the held-out rung, at ' +
      'about ' + ratio + '× the compute, and its measured optimum lands just below the line. For the ' +
      'final mixture the same law is shifted down by a constant.', { cols: 74 });
  }

  /* ---------- (b) the learning-rate ablation at ML4 scale ---------- */
  function panelB() {
    var B = FR.b, opt = B.min.lr, pred = B.ladder_prediction_lr;
    var gap = Math.round((pred / opt - 1) * 100);   // derived: prediction over measured optimum
    var ax = CH.axes(body, X, TOP, W, H, { x0: 1.5e-6, x1: 7e-5, y0: -0.0015, y1: 0.031, logx: true,
      xt: [], yt: [0, 0.01, 0.02, 0.03], yf: function (v) { return CH.fmt(v, 2); },
      yl: 'validation loss minus its minimum', xl: 'peak learning rate' });
    xtick(body, ax, 2e-6, '2 × 10', '−6');
    xtick(body, ax, 1e-5, '10', '−5');
    xtick(body, ax, 5e-5, '5 × 10', '−5');

    sw(body, 0, 58, 'dot', CH.GREY, 'full-scale SFT runs');
    sw(body, 300, 58, 'line', CH.GREY2, 'parabola fit');
    sw(body, 0, 84, 'tri', C.amber, 'measured optimum');
    sw(body, 300, 84, 'dash', C.blue, 'ladder prediction');

    CH.line(body, ax, B.curve, { color: CH.GREY2, w: 3 });
    var px = ax.sx(pred), ox = ax.sx(opt), oy = ax.sy(0);
    var pl = K.n('path', { d: 'M' + px + ' ' + TOP + ' V' + (TOP + H), stroke: C.blue,
      'stroke-width': 2.2, 'stroke-dasharray': '6 4' });
    body.appendChild(pl);
    function thick(on) { pl.setAttribute('stroke-width', on ? 4 : 2.2); }
    CH.dots(body, ax, B.points.map(function (q) { return [q.lr, q.dloss]; }), { color: CH.GREY, r: 5 });
    body.appendChild(K.n('circle', { cx: ox, cy: oy - 1, r: 14, fill: C.amber, opacity: 0.14 }));
    body.appendChild(K.n('path', { d: tri(ox, oy - 1, 8.5), fill: C.amber, stroke: '#fff', 'stroke-width': 1.6 }));
    // the gap is along the x axis, a learning rate, so say "rate" rather than "above"
    K.mono(body, px + 9, TOP + 16, 'prediction: about ' + gap + '% higher rate',
      { size: 13, color: C.ink });

    var base = 'measured optimum ' + sci(opt, 2) + ', ladder prediction ' + sci(pred, 2);
    out.set(base);
    var ring = mkRing();
    B.points.forEach(function (q) {
      CH.hot(hots, ax.sx(q.lr), ax.sy(q.dloss), 12, function () {
        ring.at(ax.sx(q.lr), ax.sy(q.dloss)); thick(false);
        out.set('run at peak rate ' + sci(q.lr, 2) + ': loss ' + CH.fmt(q.dloss, 4) + ' above the minimum');
      }, { label: 'full-scale SFT run' });
    });
    CH.hot(hots, ox, oy - 1, 11, function () {
      ring.at(ox, oy - 1, 13); thick(false);
      out.set('measured optimum at peak rate ' + sci(opt, 2));
    }, { label: 'measured optimum' });
    CH.hot(hots, 0, 0, 0, function () {
      ring.at(); thick(true);
      out.set('ladder prediction ' + sci(pred, 2) + ', about ' + gap + '% above the optimum rate');
    }, { label: 'ladder prediction', rect: [px - 7, TOP, 14, H - 36] });
    K.callout(note, 0, 444, 640, 'Early in training, the sweep at ML4 scale bottoms out close to the ' +
      'ladder\'s prediction (dashed). The sweep is a check: the rate itself came from the ladder, ' +
      'not from a search at this scale.', { cols: 74 });
  }

  // one readout line; its exponents are raised tspans, so the line never grows extra nodes
  var ro = K.mono(svg, 0, 424, '', { size: 13, color: C.ink, weight: 500 });
  out = { set: function (s) { rich(ro, s); } };
  note = CH.body(svg);
  // reserve the callout's tallest state, so the foot (placed once) never lands on it
  svg.appendChild(K.n('rect', { x: 0, y: 444, width: 640, height: 3 * 19 + 22, fill: 'none' }));
  function draw(i) {
    CH.clear(body); CH.clear(hots); CH.clear(note);
    if (i) panelB(); else panelA();
  }
  K.switcher(ctl, 0, 0, ['(a) scaling ladder', '(b) ablation at ML4 scale'], draw, { tint: 'amber' });
  K.foot(svg, 'Values are approximate, read from the raster figure; hover a point to read it. The ' +
    'compute ratio and the gap between prediction and optimum are derived from those reads. Both ' +
    'learning-rate axes are per ' + BM + 'M-token batch.');
};
