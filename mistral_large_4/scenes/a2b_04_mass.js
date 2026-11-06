/* S_MASS: section 3.2.2's attention-mass check. Converted layers put around
   55% of their attention more than 512 tokens back, native long layers 67%,
   measured with 32k-token probes from six sources across intermediate
   checkpoints. The window layer's 0% is not a measurement: a 512-token window
   cannot reach further, so it follows from the definition. The beams link the
   probe's two regions to their shares of the mass; they show shares, not
   positions inside the probe. */
window.SCENES.S_MASS = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var FAR = F['mass.far'], PK = F['mass.probe_k'];
  var svg = K.board(host, { alt: 'Share of attention mass beyond 512 tokens for converted, native long and window layers' });
  K.head(svg, 'How far back the layers look', PK + 'k-token probes · ' + F['mass.sources'] +
    ' data sources · intermediate checkpoints');

  var TYPES = [
    { name: 'converted layers', tag: 'converted', far: F['mass.converted'], col: C.amber },
    { name: 'native long layers', tag: 'native long', far: F['mass.native'], col: C.blue },
    { name: 'window layers, before', tag: 'window, before', far: 0, col: CH.GREY, byDef: true }
  ];
  var X0 = 12, W = 616, XR = X0 + W;
  // the last 512 tokens of a 32k probe, to scale (k taken as 1,024; at this width either reading is 10 px)
  var SLV = Math.max(4, W * FAR / (+PK * 1024));
  var RT = 70, RB = 94, BT = 188, BB = 228;

  // ---- the probe, with the query at its right end ----
  K.label(svg, 0, 58, 'one ' + PK + 'k-token probe');
  K.mono(svg, 640, 58, 'last ' + FAR + ' tokens, by the query', { size: 12.5, anchor: 'end', color: C.ink2 });
  svg.appendChild(K.n('rect', { x: X0, y: RT, width: W, height: RB - RT, rx: 3, fill: '#fff', stroke: CH.GREY2 }));
  for (var t = 1; t < 8; t++) svg.appendChild(K.n('path', { d: 'M' + (X0 + W * t / 8) + ' ' + (RT + 6) + ' V' + (RB - 6),
    stroke: CH.GREY2 }));
  svg.appendChild(K.n('rect', { x: XR - SLV, y: RT, width: SLV, height: RB - RT, fill: C.ink2 }));

  // the stacked bar names itself ("55% of attention mass beyond 512"), so it needs no
  // separate header that could be read as the strip's
  var beams = CH.body(svg), bar = CH.body(svg), strip = CH.body(svg), note = CH.body(svg);

  // ---- every layer type on one line: the share beyond 512 ----
  var SY = 308, sx = CH.scale(0, 100, X0, XR);
  K.label(svg, 0, 266, 'beyond ' + FAR + ' tokens, by layer type');
  svg.appendChild(K.n('path', { d: 'M' + X0 + ' ' + SY + ' H' + XR, stroke: C.ink3 }));
  [0, 50, 100].forEach(function (v) {
    svg.appendChild(K.n('path', { d: 'M' + sx(v) + ' ' + SY + ' v6', stroke: C.ink3 }));
    K.mono(svg, sx(v), SY + 24, v + '%', { size: 12.5, anchor: v === 0 ? 'start' : (v === 100 ? 'end' : 'middle'),
      color: C.ink3 });
  });

  var MSG = [
    'Converted layers, the former window layers, put around ' + TYPES[0].far + '% of their attention ' +
      'more than ' + FAR + ' tokens back. Before the surgery, none of that text was within their reach.',
    'Native long layers, full attention from the start, put about ' + TYPES[1].far + '% beyond ' + FAR +
      '. Converted layers fall about ' + (TYPES[1].far - TYPES[0].far) + ' points short: the paper reads that ' +
      'as the old initialization surviving in part.',
    'Before the surgery a window layer could not look past the last ' + FAR + ' tokens, so its share ' +
      'beyond ' + FAR + ' is 0 by definition. Unlike the other two, it is not a measurement.'
  ];

  function show(i) {
    var T = TYPES[i], f = T.far / 100, xs = X0 + W * f;
    CH.clear(beams); CH.clear(bar); CH.clear(strip); CH.clear(note);
    // beams: the far region to the far share, the last 512 to the near share
    if (f > 0) beams.appendChild(K.n('path', { d: 'M' + X0 + ' ' + RB + ' L' + (XR - SLV) + ' ' + RB + ' L' + xs + ' ' + BT +
      ' L' + X0 + ' ' + BT + ' Z', fill: T.col, opacity: 0.2 }));
    beams.appendChild(K.n('path', { d: 'M' + (XR - SLV) + ' ' + RB + ' L' + XR + ' ' + RB + ' L' + XR + ' ' + BT +
      ' L' + xs + ' ' + BT + ' Z', fill: C.ink2, opacity: 0.16 }));
    // the stacked bar
    if (f > 0) bar.appendChild(K.n('rect', { x: X0, y: BT, width: xs - X0, height: BB - BT, fill: T.col }));
    bar.appendChild(K.n('rect', { x: xs, y: BT, width: XR - xs, height: BB - BT, fill: CH.GREY2 }));
    if (f > 0) K.mono(bar, X0 + 10, BT + 25, T.far + '% of attention mass beyond ' + FAR,
      { size: 14, weight: 600, color: '#fff' });
    K.mono(bar, XR - 10, BT + 25, (100 - T.far) + '% within ' + FAR + (T.byDef ? ', by definition' : ''),
      { size: 14, weight: 600, color: C.ink, anchor: 'end' });
    // the strip: all three, the picked one drawn large; the window value follows from
    // the definition, not a measurement, so its dot is hollow
    TYPES.forEach(function (U, j) {
      var on = j === i, x = sx(U.far);
      strip.appendChild(K.n('circle', { cx: x, cy: SY, r: on ? 8 : 5.5, fill: U.byDef ? '#fff' : U.col,
        stroke: U.byDef ? U.col : '#fff', 'stroke-width': U.byDef ? 2.2 : 1.6, opacity: on ? 1 : 0.6 }));
      var anc = j === 0 ? 'end' : 'start', dx = j === 0 ? 6 : (j === 1 ? -6 : -4);
      K.mono(strip, x + dx, SY - 16, U.tag + ' · ' + U.far + '%', { size: 13, anchor: anc,
        weight: on ? 600 : 400, color: on ? C.ink : C.ink3 });
    });
    K.callout(note, 0, 362, 640, MSG[i], { cols: 74, color: i === 1 ? C.blue : C.amber,
      tint: i === 1 ? K.TINT.blue : K.TINT.amber });
  }

  K.switcher(svg, 0, 0, TYPES.map(function (U) { return U.name; }), show, { tint: 'amber' });
  K.foot(svg, 'Measured across intermediate checkpoints; the paper gives the converted share as around ' +
    TYPES[0].far + '%. The window value follows from the window itself (hollow dot). The probe is schematic, ' +
    'drawn with one query at its end, and the beams show shares of the mass, not where in the probe it lands.');
};
