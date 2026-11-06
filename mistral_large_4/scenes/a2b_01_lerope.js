/* S_LEROPE: Figure 7. Top: the learned rotary frequency over fixed YaRN, per
   band, as in the figure's left panel (raster read, approximate; the left panel
   is authoritative, see figread.fig7 notes). Middle: every band's YaRN
   wavelength on a log ruler against the 4K training context. Bottom: the picked
   band drawn as a wave, under YaRN and as learned. The learned wavelength is
   the YaRN wavelength divided by the left-panel ratio, which reproduces the
   paper's 35% and 44%. */
window.SCENES.S_LEROPE = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts, FR = (api.D.figread || {}).fig7;
  var svg = K.board(host, { alt: 'Figure 7: learned rotary frequency over YaRN for 32 bands, with one band drawn as a wave' });
  K.head(svg, 'Two bands retuned, most moved less', 'Figure 7 · learned frequencies, intermediate phase-2 checkpoint');
  if (!FR || !FR.ratio) {
    K.mono(svg, 0, 100, 'Figure 7 values are not available.', { size: 14, color: C.ink2 });
    return;
  }
  var R = FR.ratio, WL = FR.wavelengths, CTX = FR.context, HL = FR.highlight, NB = R.length;
  // the printed labels (1.53x, 1.77x) where the figure gives them, the pixel read elsewhere
  var ratioOf = function (b) { return HL[b] != null ? HL[b] : R[b]; };
  var lamY = function (b) { return WL[b].fixed; };
  var lamL = function (b) { return WL[b].fixed / ratioOf(b); };
  // how far a band's wavelength moved, in whole percent, as the callout prints it
  var movePct = function (b) { return Math.round(Math.abs(1 - 1 / ratioOf(b)) * 100); };
  // bands whose wavelength is within 10% of YaRN's, counted with that same rounding
  var calm = R.filter(function (r, b) { return movePct(b) <= 10; }).length;
  // bands whose learned frequency sits below YaRN (left-panel ratio under 1)
  var below = R.filter(function (r) { return r < 1; }).length;
  // wavelengths are read to about 1%: two significant figures below 10 tokens (a step
  // of 0.1 is about 1% there), three from 10 up; YaRN and learned share one format
  var wl = function (v) {
    var s = v < 10 ? 2 : 3, e = Math.floor(Math.log10(v)) - (s - 1), r = Math.round(v / Math.pow(10, e)) * Math.pow(10, e);
    return v >= 100 ? CH.comma(r) : String(+r.toFixed(Math.max(0, -e)));
  };

  var X0 = 62, XW = 568;
  // ---- ratio per band (Figure 7, left) ----
  var selG = CH.body(svg);   // the picked band's column, behind the series
  var ax = CH.axes(svg, X0, 74, XW, 136, { x0: 0, x1: NB - 1, y0: 0.85, y1: 1.85,
    xt: [0, 16, NB - 1], yt: [1, 1.25, 1.5, 1.75], yf: function (v) { return v.toFixed(2); },
    yl: 'learned ÷ YaRN frequency' });
  K.label(svg, 0, 230, 'band');
  svg.appendChild(K.n('path', { d: 'M' + X0 + ' ' + ax.sy(1) + ' H' + (X0 + XW), stroke: C.blue,
    'stroke-width': 1.6, 'stroke-dasharray': '6 4' }));
  // the key for the dashed line sits in the plot's empty upper left, clear of the markers
  CH.key(svg, X0 + 10, ax.sy(1.38), 'YaRN: ratio 1, no change', { color: C.blue, dash: '6 4', dot: false, w: 1.6 });
  var pts = R.map(function (r, b) { return [b, r]; });
  CH.line(svg, ax, pts, { color: C.amber, w: 2 });
  CH.dots(svg, ax, pts, { color: C.amber, r: 3.4 });
  Object.keys(HL).forEach(function (k) {
    var b = +k, left = b < 16;
    K.mono(svg, ax.sx(b) + (left ? -14 : 14), ax.sy(R[b]) + 4, HL[k].toFixed(2) + '×',
      { size: 13.5, weight: 600, color: C.ink, anchor: left ? 'end' : 'start' });
  });

  // ---- every band's YaRN wavelength against the 4K context, log scale ----
  var RY = 290, lg = CH.scale(Math.pow(10, 0.5), 1e6, X0, X0 + XW, true);
  K.label(svg, X0, 262, 'YaRN wavelength per band, log scale');
  svg.appendChild(K.n('path', { d: 'M' + X0 + ' ' + RY + ' H' + (X0 + XW), stroke: C.ink3 }));
  [10, 1e3, 1e5].forEach(function (v) {
    svg.appendChild(K.n('path', { d: 'M' + lg(v) + ' ' + RY + ' v5', stroke: C.ink3 }));
    K.mono(svg, lg(v), RY + 22, v >= 1e3 ? (v / 1e3) + 'K' : String(v), { size: 12.5, anchor: 'middle', color: C.ink3 });
  });
  // one grey dot per band, at its YaRN wavelength: data, not axis ticks
  WL.forEach(function (w) {
    svg.appendChild(K.n('circle', { cx: lg(w.fixed), cy: RY, r: 2.6, fill: CH.GREY, stroke: '#fff', 'stroke-width': 0.8 }));
  });
  svg.appendChild(K.n('path', { d: 'M' + lg(CTX) + ' ' + (RY - 22) + ' V' + (RY + 8), stroke: C.ink2,
    'stroke-width': 1.4, 'stroke-dasharray': '2 3' }));
  K.mono(svg, lg(CTX) + 6, 262, (CTX / 1024) + 'K training context', { size: 12.5, color: C.ink2 });
  var rulG = CH.body(svg);

  // ---- the picked band as a wave ----
  var WY = 391, AMP = 30;
  var waveG = CH.body(svg);
  var note = CH.body(svg);

  function paintWave(b) {
    CH.clear(waveG);
    var span = 2 * lamY(b), sx = CH.scale(0, span, X0, X0 + XW);
    K.label(waveG, 0, 340, 'band ' + b);
    CH.key(waveG, 92, 340, 'YaRN · ' + wl(lamY(b)) + ' tokens', { color: C.blue });
    CH.key(waveG, 360, 340, 'learned · ' + wl(lamL(b)) + ' tokens', { color: C.amber });
    waveG.appendChild(K.n('path', { d: 'M' + X0 + ' ' + WY + ' H' + (X0 + XW), stroke: C.line }));
    [1, 2].forEach(function (k) {
      waveG.appendChild(K.n('path', { d: 'M' + sx(k * lamY(b)) + ' ' + (WY + AMP + 4) + ' v6', stroke: C.ink3 }));
      K.mono(waveG, sx(k * lamY(b)), WY + AMP + 26, wl(k * lamY(b)), { size: 12.5,
        anchor: k === 2 ? 'end' : 'middle', color: C.ink3 });
    });
    var f = CTX / span;
    if (f > 0.05 && f < 0.95) {
      waveG.appendChild(K.n('path', { d: 'M' + sx(CTX) + ' ' + (WY - AMP - 6) + ' V' + (WY + AMP + 10),
        stroke: C.ink2, 'stroke-width': 1.4, 'stroke-dasharray': '2 3' }));
      K.mono(waveG, sx(CTX), WY + AMP + 26, (CTX / 1024) + 'K', { size: 12.5, anchor: 'middle', color: C.ink2 });
    }
    [[lamY(b), C.blue, 2], [lamL(b), C.amber, 2.6]].forEach(function (s) {
      var d = '';
      for (var i = 0; i <= 360; i++) {
        var t = span * i / 360;
        d += (i ? 'L' : 'M') + sx(t).toFixed(1) + ' ' + (WY - AMP * Math.cos(2 * Math.PI * t / s[0])).toFixed(1);
      }
      waveG.appendChild(K.n('path', { d: d, fill: 'none', stroke: s[1], 'stroke-width': s[2],
        'stroke-linejoin': 'round', opacity: 0.92 }));
    });
  }

  function say(b) {
    CH.clear(note);
    var r = ratioOf(b), ap = movePct(b);
    var said = ap === 0 ? 'its wavelength is unchanged' :
      'its wavelength is ' + ap + '% ' + (r > 1 ? 'shorter' : 'longer');
    var s = 'Band ' + b + ': frequency ' + r.toFixed(2) + '× YaRN, so ' + said + '. ';
    if (HL[b] != null) s += 'One of the two big moves, both beside the ' + (CTX / 1024) + 'K training context.';
    else if (ap <= 10) s += 'Like most, it moved little: ' + calm + ' of ' + NB +
      ' wavelengths are within 10% of YaRN\'s.';
    else if (r < 1) s += 'A modest move the other way: one of ' + below + ' bands that sit below YaRN.';
    else s += 'A smaller move, next to the two big ones.';
    K.callout(note, 0, 466, 640, s, { cols: 74 });
  }

  var CHIPS = [4, 15, 16], sw = null;
  function select(b) {
    CH.clear(selG);
    selG.appendChild(K.n('rect', { x: ax.sx(b) - 9, y: 74, width: 18, height: 136, rx: 4, fill: K.TINT.amber }));
    CH.clear(rulG);
    var xa = lg(lamY(b)), xb = lg(lamL(b));
    rulG.appendChild(K.n('circle', { cx: xa, cy: RY, r: 7, fill: '#fff', stroke: C.blue, 'stroke-width': 2.4 }));
    rulG.appendChild(K.n('circle', { cx: xb, cy: RY, r: 5.5, fill: C.amber, stroke: '#fff', 'stroke-width': 1.6 }));
    rulG.appendChild(K.n('circle', { cx: ax.sx(b), cy: ax.sy(R[b]), r: 7.5, fill: 'none', stroke: C.amber, 'stroke-width': 2 }));
    paintWave(b);
    say(b);
    // keep the band chips honest when a band is picked on the chart instead
    if (sw) sw.btns.forEach(function (g, i) {
      var on = CHIPS[i] === b;
      g.firstChild.setAttribute('fill', on ? K.TINT.amber : 'none');
      g.firstChild.setAttribute('stroke', on ? C.amber : C.line);
      g.lastChild.setAttribute('fill', on ? C.amber : C.ink3);
    });
  }

  sw = K.switcher(svg, 0, 0, CHIPS.map(function (b) { return 'band ' + b; }),
    function (i) { select(CHIPS[i]); }, { tint: 'amber' });
  K.mono(svg, 640, 18, 'or hover any band', { size: 12.5, anchor: 'end', color: C.ink3 });
  for (var b = 0; b < NB; b++) (function (b) {
    CH.hot(svg, 0, 0, 0, function () { select(b); }, { rect: [ax.sx(b) - 9, 70, 18, 144], label: 'band ' + b });
  })(b);
  sw.pick(2);

  K.foot(svg, 'Figure 7 is a raster image, so values are read from pixels and approximate. Learned ' +
    'wavelength is the YaRN wavelength divided by the frequency ratio. SSMax, the other learned ' +
    'piece, scales each head\'s queries and is not in this figure.');
};
