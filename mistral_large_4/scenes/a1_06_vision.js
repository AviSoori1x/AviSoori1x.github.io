/* S_VISION: the patch merger, step by step (schematic, following §2.3.2), then
   Figure 2: a frozen 400M version of the encoder against Perception Encoder
   ViT-L/14 (400M, frozen) on ten benchmarks. Figure 2 is raster, so its values
   are read from pixels and approximate (window.ML4.figread.fig2). */
window.SCENES.S_VISION = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts, FR = (api.D.figread || {}).fig2;
  var svg = K.board(host, { alt: 'The vision encoder: patch merger steps, and Figure 2 benchmark curves' });
  K.head(svg, 'The vision encoder', F['vis.params_b'] + 'B parameters · trained from scratch · ' +
    F['vis.tokens_t'] + 'T tokens of captioning');
  var body = CH.body(svg);
  var SHADE = [1, 0.72, 0.5, 0.3];

  function cells(p, x, y, nn, col, op, w) {
    for (var i = 0; i < nn; i++) {
      p.appendChild(K.n('rect', { x: x + i * (w || 20), y: y, width: (w || 20) - 3, height: 18, rx: 3,
        fill: col, opacity: op == null ? 1 : (op.length ? op[Math.floor(i / 4)] : op) }));
    }
  }

  function merger() {
    var stepG = CH.body(body), steps;
    var STEP = [
      'The encoder cuts the image into patches and turns each into an embedding, d numbers wide. ' +
        'Images keep their own aspect ratio, up to ' + F['vis.res_w'] + ' by ' + F['vis.res_h'] + ' pixels.',
      'Each 2 by 2 block of neighboring patches is merged: its four embeddings are laid end to ' +
        'end into one vector, 4d wide.',
      'A linear projection brings the 4d vector back down. Four patches now cost the language ' +
        'model one position instead of four.',
      'An MLP maps the merged embedding into the language model\'s input embedding space. Here ' +
        'the 16 patches become 4 tokens.'
    ];
    function draw(s) {
      CH.clear(stepG);
      // steps not reached yet stay on the board as faint ghosts, so the layout never jumps
      var lay = function (k) {
        var g = K.n('g', { opacity: k <= s ? 1 : 0.16 });
        stepG.appendChild(g);
        return g;
      };
      var g0 = lay(0), g1 = lay(1), g2 = lay(2), g3 = lay(3);
      K.label(g0, 0, 98, 'one image, 4 × 4 patches');
      for (var r = 0; r < 4; r++) for (var c = 0; c < 4; c++) {
        var inB = r < 2 && c < 2, k = r * 2 + c;
        g0.appendChild(K.n('rect', { x: 4 + c * 42, y: 110 + r * 42, width: 38, height: 38, rx: 5,
          fill: inB && s >= 1 ? C.amber : CH.GREY2, opacity: inB && s >= 1 ? SHADE[k] : 0.7 }));
      }
      // the grid sits 4 in, so the block outline starts at 0 and the figure keeps its left edge
      g1.appendChild(K.n('rect', { x: 0, y: 106, width: 88, height: 88, rx: 8,
        fill: 'none', stroke: C.amber, 'stroke-width': 2.4 }));
      var X = 250;
      K.mono(g0, X, 98, s === 0 ? 'one embedding per patch, d wide' : 'the four embeddings of the block',
        { size: 13, color: C.ink2 });
      for (var e = 0; e < 4; e++) cells(g0, X + e * 100, 110, 4, s >= 1 ? C.amber : CH.GREY, s >= 1 ? SHADE[e] : 0.8);
      K.arrow(g1, X + 150, 136, X + 150, 170, { color: C.ink3 });
      K.mono(g1, X + 162, 158, 'concatenate', { size: 13, color: C.ink2 });
      cells(g1, X, 180, 16, C.amber, SHADE);
      K.mono(g1, X + 330, 194, '4d', { size: 13, color: C.ink, weight: 600 });
      K.arrow(g2, X + 150, 206, X + 150, 240, { color: C.ink3 });
      K.mono(g2, X + 162, 228, 'linear projection', { size: 13, color: C.ink2 });
      cells(g2, X + 120, 250, 4, C.amber, 1);
      K.mono(g2, X + 210, 264, 'one merged embedding', { size: 13, color: C.ink, weight: 600 });
      K.arrow(g3, X + 150, 276, X + 150, 310, { color: C.ink3 });
      K.mono(g3, X + 162, 298, 'MLP', { size: 13, color: C.ink2 });
      cells(g3, X + 100, 320, 6, C.ink2, 1);
      K.mono(g3, X + 230, 334, 'LLM input space', { size: 13, color: C.ink, weight: 600 });
      for (var t = 0; t < 4; t++) g3.appendChild(K.n('rect', { x: (t % 2) * 46 + 44, y: 300 + Math.floor(t / 2) * 46,
        width: 40, height: 40, rx: 6, fill: C.ink2, opacity: t ? 0.55 : 1 }));
      K.mono(g3, 0, 412, '16 patches, 4 tokens', { size: 13, color: C.ink, weight: 600 });
      K.callout(stepG, 0, 436, 640, STEP[s], { cols: 74 });
    }
    K.switcher(body, 0, 44, ['1 patches', '2 concatenate', '3 project', '4 MLP'], draw, { tint: 'amber' });
  }

  function figure2() {
    if (!FR || !FR.panels) {
      K.mono(body, 0, 100, 'Figure 2 values are not available.', { size: 14, color: C.ink2 });
      return;
    }
    var W = 112, GAP = 20, PH = 84, out, out2;
    FR.panels.forEach(function (p, i) {
      var col = i % 5, row = Math.floor(i / 5), x = col * (W + GAP), y = 106 + row * 152;
      var all = p.ours.concat(p.pe), lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
      var pad = (hi - lo) * 0.12 + 0.005;
      var ax = { sx: CH.scale(10, 100, x, x + W), sy: CH.scale(lo - pad, hi + pad, y + PH, y) };
      var end = p.ours.length - 1, ahead = p.pe[end] > p.ours[end];
      body.appendChild(K.n('rect', { x: x, y: y, width: W, height: PH, rx: 5, fill: '#fff',
        stroke: ahead ? C.blue : C.line, 'stroke-width': ahead ? 1.6 : 1 }));
      K.mono(body, x, y - 26, p.name, { size: 13, color: C.ink, weight: 600 });
      K.mono(body, x, y - 9, CH.fmt(p.ours[end], 2) + ' vs ' + CH.fmt(p.pe[end], 2), { size: 12.5, color: C.ink2 });
      var pts = function (s) { return p.x.map(function (xv, j) { return [xv, s[j]]; }); };
      CH.line(body, ax, pts(p.pe), { color: C.blue, w: 2 });
      CH.line(body, ax, pts(p.ours), { color: C.amber, w: 2.4 });
      CH.hot(body, 0, 0, 0, function () {
        // PE counts as ahead only beyond the read error of about 0.003
        var from = p.x.filter(function (xv, j) { return p.pe[j] - p.ours[j] > 0.003; })[0];
        out.set(p.full + ' · ' + p.metric.toLowerCase());
        out2.set('at 100% of SFT: ours ' + CH.fmt(p.ours[end], 3) + ', PE ' + CH.fmt(p.pe[end], 3) +
          (from ? ' · PE ahead from ' + from + '%' : ' · ours ahead throughout'));
      }, { rect: [x - 4, y - 40, W + 8, PH + 44], label: p.name });
    });
    CH.key(body, 0, 396, 'ours, 400M downscaled, frozen', { color: C.amber });
    CH.key(body, 300, 396, 'Perception Encoder ViT-L/14, 400M, frozen', { color: C.blue });
    K.mono(body, 0, 420, 'x: share of the SFT budget done, 10% to 100%. Each panel has its own y scale.',
      { size: 12.5, color: C.ink3 });
    out = CH.readout(body, 0, 456, { size: 13.5, text: 'Hover or tap a panel to read its end values.' });
    out2 = CH.readout(body, 0, 478, { size: 13, text: 'Under each name: ours, then PE, at 100% of SFT.' });
  }

  var views = [merger, figure2];
  K.switcher(svg, 0, 0, ['patch merger', 'Figure 2: ten benchmarks'], function (i) {
    CH.clear(body); views[i]();
  }, { tint: 'amber' });
  K.foot(svg, 'The patch merger is a schematic of §2.3.2; d is not given. Figure 2 is a raster ' +
    'image, so its values are read from pixels and approximate, to about ±0.003.');
};
