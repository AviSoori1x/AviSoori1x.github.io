/* S_PENALTY: Figure 4, SWE-bench Verified through midtraining (up to 100%) and
   on into SFT (above 100%) for five attention designs. MLA is the full-attention
   reference and is amber; the picked design is blue. Figure 4 is raster, so the
   values are pixel reads (window.ML4.figread.fig4) and approximate. */
window.SCENES.S_PENALTY = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, FR = (api.D.figread || {}).fig4;
  var svg = K.board(host, { alt: 'Figure 4: SWE-bench Verified during midtraining and SFT for five attention designs' });
  K.head(svg, 'The agentic penalty', 'Figure 4 · SWE-bench Verified · five attention designs, same token budget');
  var body = CH.body(svg);
  if (!FR || !FR.series) {
    K.mono(body, 0, 100, 'Figure 4 values are not available.', { size: 14, color: C.ink2 });
    return;
  }
  var S = FR.series.slice(), NAME = ['MLA', 'SWA', 'SWA × YOCO', 'SWA × CLA', 'SWA × CLA × YOCO'];
  var at = function (s, x) { return s.pts.filter(function (q) { return q.x === x; })[0].y; };
  var endOf = function (s) { return s.pts[s.pts.length - 1]; };
  var B = FR.boundary;   // 100: end of midtraining, SFT above it
  // Up to 100%, MLA and SWA x CLA x YOCO are the runs that Figure 6 draws as its full-attention
  // and original baselines: all 13 midtraining checkpoints agree to within 0.15 points. Figure 6
  // is vector, so use its exact values there (rounded to 0.1), and the two beats agree.
  var F6 = (api.D.figread || {}).fig6 || [];
  function exact(s, nm) {
    var e = F6.filter(function (r) { return r.name === nm; })[0];
    var mid = s.pts.filter(function (q) { return q.x <= B; });
    if (!e || e.swe.length !== mid.length) return s;
    if (!mid.every(function (q, i) { return Math.abs(q.y - e.swe[i].swe) < 0.3; })) return s;
    var k = 0;
    return { name: s.name, pts: s.pts.map(function (q) {
      return q.x <= B ? { x: q.x, y: Math.round(e.swe[k++].swe * 10) / 10 } : q;
    }) };
  }
  S[0] = exact(S[0], 'Full-attention baseline');
  S[4] = exact(S[4], 'Original baseline');
  var swapped = S[0] !== FR.series[0] && S[4] !== FR.series[4];
  var full = S[0], f100 = at(full, B), fEnd = endOf(full).y;

  var ax;
  function frame() {
    ax = CH.axes(body, 40, 70, 600, 290, { x0: 0, x1: 200, y0: 10, y1: 70, xt: [0, 50, 100, 150],
      yt: [20, 40, 60], xf: function (v) { return v + '%'; },
      xl: 'training tokens, % of the midtraining horizon', yl: 'SWE-bench Verified (%)' });
    // SFT continues above 100%: shade it and draw the boundary
    body.insertBefore(K.n('rect', { x: ax.sx(B), y: ax.y, width: ax.sx(200) - ax.sx(B), height: ax.h,
      fill: K.TINT.ink }), ax.g);
    body.appendChild(K.n('path', { d: 'M' + ax.sx(B) + ' ' + ax.y + ' V' + (ax.y + ax.h),
      stroke: C.ink3, 'stroke-dasharray': '5 4', 'stroke-width': 1.2 }));
    // phase names sit low in the plot, where no curve reaches
    K.label(body, ax.sx(B / 2), ax.y + ax.h - 12, 'midtraining', { anchor: 'middle' });
    K.label(body, ax.sx(150), ax.y + ax.h - 12, 'SFT', { anchor: 'middle' });
  }

  function draw(sel) {
    CH.clear(body);
    frame();
    var pts = function (s) { return s.pts.map(function (q) { return [q.x, q.y]; }); };
    // YOCO and CLA are measured against sliding windows alone, so SWA gets a darker line
    var vsSwa = sel === 2 || sel === 3;
    // context first, then the reference, then the pick, so the two stay on top
    S.forEach(function (s, i) {
      if (i !== 0 && i !== sel && !(vsSwa && i === 1)) CH.line(body, ax, pts(s), { color: CH.GREY2, w: 1.5 });
    });
    if (vsSwa) CH.line(body, ax, pts(S[1]), { color: C.ink3, w: 2 });
    CH.line(body, ax, pts(full), { color: C.amber, w: 2.6 });
    CH.dots(body, ax, pts(full), { color: C.amber, r: 3.6 });
    var pick = S[sel], col = sel === 0 ? C.amber : C.blue;
    if (sel !== 0) {
      CH.line(body, ax, pts(pick), { color: C.blue, w: 2.6 });
      CH.dots(body, ax, pts(pick), { color: C.blue, r: 3.6 });
    }
    // end values, labelled right of the last checkpoint
    var e = endOf(full), pe = endOf(pick);
    var yF = ax.sy(e.y), yP = ax.sy(pe.y);
    if (sel !== 0 && Math.abs(yF - yP) < 16) { yF -= 8; yP += 8; }
    K.mono(body, ax.sx(e.x) + 12, yF + 4.5, CH.fmt(e.y, 1), { size: 13.5, weight: 600, color: C.ink });
    if (sel !== 0) K.mono(body, ax.sx(pe.x) + 12, yP + 4.5, CH.fmt(pe.y, 1), { size: 13.5, weight: 600, color: C.ink });

    // key: the reference, the pick, then sliding windows alone when it is the yardstick, the context
    CH.key(body, 0, 428, 'MLA, full attention', { color: C.amber });
    if (sel !== 0) CH.key(body, 192, 428, NAME[sel], { color: C.blue });
    if (vsSwa) {
      CH.key(body, 316, 428, 'SWA alone', { color: C.ink3, dot: false });
      CH.key(body, 432, 428, 'the other two designs', { color: CH.GREY2, dot: false });
    } else {
      CH.key(body, sel === 0 ? 192 : 400, 428, sel === 0 ? 'the four sliding-window designs' : 'the other designs',
        { color: CH.GREY2, dot: false });
    }
    K.callout(body, 0, 446, 640, message(sel), { cols: 74, color: sel === 0 ? C.amber : C.blue,
      tint: sel === 0 ? K.TINT.amber : K.TINT.blue });
    // reserve the tallest callout's height in every state, so the foot (placed once, under
    // whatever is drawn first) clears all of them
    body.appendChild(K.n('rect', { x: 0, y: 446, width: 1, height: SLOT, fill: 'none' }));
  }

  // the callout for each design; gaps are differences of two values at the same x
  function message(sel) {
    var pick = S[sel], p100 = at(pick, B), pEnd = endOf(pick).y;
    var s100 = at(S[1], B), sEnd = endOf(S[1]).y, f = function (v) { return CH.fmt(v, 1); };
    return [
      'Full attention: ' + f(f100) + ' at the end of midtraining (100%) and ' + f(fEnd) + ' at the last SFT ' +
        'checkpoint. Pick a design to see how far below it each one lands.',
      'Sliding windows alone: ' + f(p100) + ' at 100% and ' + f(pEnd) + ' at the last checkpoint, ' +
        f(f100 - p100) + ' and ' + f(fEnd - pEnd) + ' points below full attention.',
      'Sliding windows plus YOCO: ' + f(p100) + ' at 100% and ' + f(pEnd) + ' at the last checkpoint, ' +
        f(s100 - p100) + ' and ' + f(sEnd - pEnd) + ' points below sliding windows alone (dark grey). ' +
        'By late SFT, YOCO costs almost nothing.',
      'Sliding windows plus CLA: ' + f(p100) + ' at 100% and ' + f(pEnd) + ' at the last checkpoint, ' +
        f(s100 - p100) + ' and ' + f(sEnd - pEnd) + ' points below sliding windows alone (dark grey). ' +
        'CLA costs more, even at the end.',
      'All three tricks, the design ML4 began with: ' + f(p100) + ' at 100% and ' + f(pEnd) + ' at the last ' +
        'checkpoint, ' + f(f100 - p100) + ' and ' + f(fEnd - pEnd) + ' points below full attention.'
    ][sel];
  }
  // the kit's callout wraps at min(cols, what a 640 box holds) and is 19px a line plus 22
  var COLS = Math.min(74, Math.floor((640 - 32) / (13.5 * 0.625)));
  var SLOT = Math.max.apply(null, [0, 1, 2, 3, 4].map(function (i) {
    return K.wrap(message(i), COLS).length * 19 + 22;
  }));

  var sw = K.switcher(svg, 0, 0, ['MLA, full attention', 'SWA', 'SWA × YOCO', 'SWA × CLA',
    'SWA × CLA × YOCO'], function (i) {
    draw(i);
    // the reference wears amber, the comparisons blue
    if (sw) {
      var b = sw.btns[i], c0 = i === 0 ? C.amber : C.blue;
      b.firstChild.setAttribute('stroke', c0);
      b.firstChild.setAttribute('fill', i === 0 ? K.TINT.amber : K.TINT.blue);
      b.lastChild.setAttribute('fill', c0);
    }
  }, { tint: 'blue' });
  sw.pick(4);

  K.foot(svg, 'MLA is full attention. Figure 4 is raster: values are pixel reads, good to about ±0.1 ' +
    'points (one hidden SWA point ±0.3).' + (swapped ? ' Up to 100%, MLA and all three tricks match ' +
    'Figure 6\'s two baselines at every checkpoint, so they use its exact values.' : ''));
};
