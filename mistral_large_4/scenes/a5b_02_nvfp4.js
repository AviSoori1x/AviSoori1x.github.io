/* S_NVFP4: which parts of the checkpoint are quantized. §6: the routed experts go
   to NVFP4, every other component keeps its training precision, and the mixed
   BF16, FP8 and NVFP4 checkpoint is 606 GB. The block diagram is a schematic (no
   areas to scale, and the paper does not say which parts are BF16 and which FP8).
   The bit widths are the ones in the format names. */
window.SCENES.S_NVFP4 = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts, n = K.n;
  var GB = F['q.gb'];
  var svg = K.board(host, { alt: 'Checkpoint parts and their precision: routed experts in NVFP4, the rest at training precision, ' + GB + ' GB' });
  K.head(svg, 'A trillion parameters in ' + GB + ' GB', 'post-training quantization · mixed BF16, FP8 and NVFP4 · ' +
    F['cp.gpus'] + ' GB300 GPUs');

  // name, bits (as in the format's own name)
  var FMT = [['NVFP4', 4], ['FP8', 8], ['BF16', 16]];
  var PARTS = [
    { name: 'routed experts', fmt: [0], say: 'The routed experts are quantized to NVFP4 after training. Naive abs-max ' +
      'scaling, calibrated on ' + F['q.calib_m'] + 'M tokens of the SFT mixture, already matches the FP8 ' +
      'reference in all evaluations.' },
    { name: 'attention, indexers', fmt: [1, 2], say: 'Left at training precision. Attention and the ' +
      'sparse-attention indexers stay in BF16 or FP8, as trained; only the routed experts go to 4 bits.' },
    { name: 'router, norms', fmt: [1, 2], say: 'Left at training precision. The router, which picks the ' +
      'experts for each token, and the norms stay in BF16 or FP8, as trained.' },
    { name: 'embeddings, output head, vision encoder', fmt: [1, 2], say: 'Left at training precision. The token ' +
      'embeddings, the output head and the ' + F['vis.params_b'] + 'B-parameter vision encoder stay in BF16 or FP8.' }
  ];

  K.label(svg, 0, 10, 'the checkpoint · click a part');
  K.panel(svg, 0, 24, 404, 150, { r: 10 });
  K.label(svg, 14, 47, 'one layer, repeated');

  // each part is one button: a frame that shows the selection, then its drawing
  var rings = [], cells = [];
  function part(i, x, y, w, h, fill, drawIt) {
    var g = n('g', { tabindex: 0, role: 'button', 'aria-label': PARTS[i].name, style: 'cursor:pointer;outline:none' });
    var ring = n('rect', { x: x, y: y, width: w, height: h, rx: 8, fill: fill, stroke: CH.GREY2 });
    g.appendChild(ring);
    drawIt(g);
    g.addEventListener('click', function () { pick(i); });
    g.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(i); }
    });
    svg.appendChild(g);
    rings.push(ring);
  }
  function named(g, x, y, s) { K.mono(g, x, y, s, { size: 13, anchor: 'middle', color: C.ink }); }
  part(0, 196, 34, 198, 132, '#fff', function (g) {
    named(g, 295, 54, 'routed experts');
    // amber only while the experts are the selected part (pick sets the fill)
    for (var r = 0; r < 4; r++) for (var c = 0; c < 8; c++) {
      var cell = n('rect', { x: 206 + c * 23, y: 64 + r * 24, width: 19, height: 19, rx: 4,
        fill: CH.GREY2, opacity: 0.9 - 0.14 * ((r + c) % 3) });
      g.appendChild(cell);
      cells.push(cell);
    }
  });
  part(1, 12, 58, 172, 48, K.TINT.ink, function (g) { named(g, 98, 86.5, PARTS[1].name); });
  part(2, 12, 114, 172, 48, K.TINT.ink, function (g) { named(g, 98, 142.5, PARTS[2].name); });
  part(3, 0, 186, 404, 40, K.TINT.ink, function (g) { named(g, 202, 210.5, PARTS[3].name); });

  // the precision ladder on the right
  K.label(svg, 430, 10, 'stored as');
  var rows = FMT.map(function (f, j) {
    var y = 30 + j * 58;
    var t = K.mono(svg, 430, y + 16, f[0] + ' · ' + f[1] + '-bit', { size: 13.5, color: C.ink, weight: 600 });
    svg.appendChild(n('rect', { x: 430, y: y + 26, width: 200, height: 12, rx: 6, fill: K.TINT.ink }));
    // the bar is the bit width: 16 bits fill the 200px track
    var b = n('rect', { x: 430, y: y + 26, width: f[1] * 12.5, height: 12, rx: 6, fill: CH.GREY2 });
    svg.appendChild(b);
    return { t: t, b: b };
  });

  var say = CH.body(svg);
  function pick(i) {
    rings.forEach(function (r, j) {
      r.setAttribute('stroke', j === i ? C.amber : CH.GREY2);
      r.setAttribute('stroke-width', j === i ? 2.4 : 1);
    });
    cells.forEach(function (c) { c.setAttribute('fill', i === 0 ? C.amber : CH.GREY2); });
    rows.forEach(function (rw, j) {
      var on = PARTS[i].fmt.indexOf(j) >= 0;
      rw.t.setAttribute('fill', on ? C.ink : C.ink3);
      rw.t.setAttribute('font-weight', on ? 600 : 400);
      rw.b.setAttribute('fill', on ? C.amber : CH.GREY2);
    });
    CH.clear(say);
    K.callout(say, 0, 240, 640, PARTS[i].say, { cols: 74 });
  }

  // what the mix buys: sizes for about 1T parameters, 2 bytes or 1 byte a weight
  var P = F['size.total_t'] * 1000;              // billions of parameters
  var avg = GB * 8 / P;                          // GB to gigabits, over billions of weights
  var SIZE = [['every weight BF16', P * 2, CH.GREY2, 'about ', ''], ['every weight FP8', P, CH.GREY2, 'about ', ''],
    ['ML4, mixed', GB, C.amber, '', ', about ' + CH.fmt(avg, 1) + ' bits a weight']];
  var bx = CH.scale(0, 2000, 160, 500);
  K.label(svg, 0, 348, 'checkpoint size for about ' + F['size.total_t'] + 'T parameters');
  SIZE.forEach(function (s, j) {
    var y = 364 + j * 28, me = j === 2;
    K.mono(svg, 0, y + 11, s[0], { size: 13, color: me ? C.ink : C.ink2, weight: me ? 600 : 400 });
    svg.appendChild(n('rect', { x: 160, y: y, width: bx(s[1]) - 160, height: 14, rx: 4, fill: s[2] }));
    K.mono(svg, bx(s[1]) + 9, y + 11, s[3] + CH.comma(s[1]) + ' GB' + s[4], { size: 13, color: me ? C.ink : C.ink3,
      weight: me ? 600 : 400 });
  });

  // at fleet scale
  K.label(svg, 0, 478, 'at fleet scale');
  var CH3 = ['llm-d routing', 'Model Express P2P transfer', 'new replica in under ' + F['ops.start_s'] + ' s'];
  var cx = 0;
  CH3.forEach(function (s) {
    var c = K.chip(svg, cx, 490, s, { size: 12.5, color: C.ink2, stroke: CH.GREY });
    cx += c._w + 22;
  });

  K.foot(svg, 'Schematic, not to scale; the paper does not say which parts are BF16 and which FP8. ' +
    'The all-BF16 and all-FP8 sizes and the bits a weight are arithmetic on about 1T parameters.');
  pick(0);
};
