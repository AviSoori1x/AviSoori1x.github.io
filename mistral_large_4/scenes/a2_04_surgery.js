/* S_SURGERY: §3.2 as a schematic on the Figure 3 layer pattern. After surgery
   every layer keeps its own KV over the whole context. A layer that used to borrow
   KV gets KV-producer weights copied from the last long layer that had its own:
   for CLA the long layer just before it, for YOCO the halfway layer, 35. Pick a
   former borrower to see its copy. The window bars are illustrative. */
window.SCENES.S_SURGERY = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var EVERY = { four: 4 }[F['kv.long_every']] || 4;
  var WIN = F['kv.swa_window'];
  var SUP = { 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  var sup = function (v) { return String(v).split('').map(function (d) { return SUP[d]; }).join(''); };
  var svg = K.board(host, { alt: 'Schematic of model surgery: each layer that borrowed KV gets KV weights copied from the last long layer with its own' });
  K.head(svg, 'Model surgery, layer by layer', 'schematic · §3.2 · windows removed, every layer makes its own KV');
  var body = CH.body(svg);

  // Figure 3's own labels: 35 is the YOCO anchor, 7 reuses the KV of 3, 39 and 43 reuse 35's
  var ANCHOR = F['a2.yoco_anchor'], L7 = F['a2.cla_layer'], L39 = F['a2.yoco_l1'], L43 = F['a2.yoco_l2'];
  // the same layer pattern as the previous beat: 0 to 7, 32 to 35, then 36 to 43
  var P = 48, CW = 40, CHT = 64, R1 = 84, R2 = 228, XB = 456;
  var CELLS = [];
  for (var i = 0; i < 2 * EVERY; i++) CELLS.push({ l: i, x: i * P, y: R1 });
  for (var j = 0; j < EVERY; j++) CELLS.push({ l: ANCHOR - EVERY + 1 + j, x: XB + j * P, y: R1 });
  for (var m = 0; m < 2 * EVERY; m++) CELLS.push({ l: ANCHOR + 1 + m, x: m * P, y: R2 });
  var AT = {};
  CELLS.forEach(function (c) { AT[c.l] = c; });
  var isLong = function (l) { return l % EVERY === EVERY - 1; };
  // the layers that borrowed KV in the original design, and where their copy comes from:
  // CLA copies the long layer just before (one group of four back, so 3 for 7), YOCO the anchor
  var BORROW = [
    { l: L7, from: L7 - EVERY, how: 'CLA' },
    { l: L39, from: ANCHOR, how: 'YOCO' },
    { l: L43, from: ANCHOR, how: 'YOCO' }
  ];
  var MSG = {};
  MSG[L7] = 'Layer ' + L7 + ' used CLA: it stored no KV of its own and read the KV of layer ' +
    (L7 - EVERY) + '. Surgery gives it KV weights copied from layer ' + (L7 - EVERY) + ', the long ' +
    'layer just before it, ' + EVERY + ' layers back, which takes a very similar residual stream as input.';
  MSG[L39] = 'Layer ' + L39 + ' used YOCO: it read the KV made at layer ' + ANCHOR + ', the halfway ' +
    'point. It now gets KV weights copied from ' + ANCHOR + '. Here that is ' + (L39 - ANCHOR) +
    ' layers back; for the long layers after it, the source is farther away.';
  MSG[L43] = 'Layer ' + L43 + ' used YOCO too, so its KV weights are also copied from layer ' + ANCHOR +
    ', now ' + (L43 - ANCHOR) + ' layers back. Every later long layer in the second half copies from ' +
    ANCHOR + ' as well, from farther and farther away.';
  function arrowHead(p, x, y, col) {
    p.appendChild(K.n('path', { d: 'M' + x + ' ' + y + ' l-5 -8 h10 Z', fill: col }));
  }

  function draw(sel) {
    CH.clear(body);
    var B = BORROW.filter(function (b) { return b.l === sel; })[0];
    K.label(body, 0, 70, 'first half');
    K.label(body, 0, 214, 'second half');
    K.label(body, 640, 70, 'halfway point', { anchor: 'end' });
    [[400, R1], [400, R2]].forEach(function (d) {
      for (var k = 0; k < 3; k++) body.appendChild(K.n('circle', { cx: d[0] + k * 16, cy: d[1] + CHT / 2,
        r: 2.4, fill: C.ink3 }));
    });
    CELLS.forEach(function (c) {
      var lg = isLong(c.l), hot = c.l === B.l || c.l === B.from;
      // a former borrower keeps its dashed ink outline even when picked, so all three
      // stay recognisable as pickable
      var was = BORROW.some(function (b) { return b.l === c.l; });
      body.appendChild(K.n('rect', { x: c.x, y: c.y, width: CW, height: CHT, rx: 5,
        fill: hot ? C.amber : K.TINT.amber, stroke: was ? C.ink : C.amber,
        'stroke-width': was ? 1.6 : 1.3, 'stroke-dasharray': was ? '4 3' : null }));
      if (lg) {
        body.appendChild(K.n('rect', { x: c.x, y: c.y - 6, width: CW, height: 3, rx: 1.5, fill: C.ink }));
        K.mono(body, c.x + CW / 2, c.y + CHT / 2 + 5, String(c.l), { size: 14, anchor: 'middle',
          weight: 600, color: hot ? '#fff' : C.ink2 });
      }
    });
    // the copy: an arc inside the first half for CLA, a lane down from 35 for YOCO
    var s = AT[B.from], t = AT[B.l], sx = s.x + CW / 2, tx = t.x + CW / 2;
    if (B.how === 'CLA') {
      var top = R1 - 8;
      body.appendChild(K.n('path', { d: 'M' + sx + ' ' + top + ' C' + sx + ' ' + (top - 28) + ' ' + tx + ' ' +
        (top - 28) + ' ' + tx + ' ' + (top - 9), fill: 'none', stroke: C.amber, 'stroke-width': 2.2 }));
      arrowHead(body, tx, top - 1, C.amber);
      K.mono(body, (sx + tx) / 2, top - 28, 'copy KV weights from ' + B.from, { size: 13, anchor: 'middle',
        color: C.ink, weight: 600 });
    } else {
      var lane = R1 + CHT + 24;
      body.appendChild(K.n('path', { d: 'M' + sx + ' ' + (R1 + CHT + 2) + ' V' + lane + ' H' + tx + ' V' + (R2 - 9),
        fill: 'none', stroke: C.amber, 'stroke-width': 2.2 }));
      arrowHead(body, tx, R2 - 7, C.amber);
      K.mono(body, 500, lane - 8, 'copy KV weights from ' + B.from, { size: 13, anchor: 'middle',
        color: C.ink, weight: 600 });
    }
    K.label(body, 0, 18, 'pick layer ' + L7 + ', ' + L39 + ' or ' + L43 + ', the dashed outlines');
    // which cells are long: the free space right of the second row holds the reminder
    body.appendChild(K.n('rect', { x: 476, y: R2 + 18, width: 26, height: 3, rx: 1.5, fill: C.ink }));
    K.mono(body, 512, R2 + 23, 'long layer', { size: 13, color: C.ink2 });
    K.mono(body, 476, R2 + 50, 'unnumbered: short', { size: 13, color: C.ink2 });

    // legend
    var LX = [0, 200, 420], LY = 326;
    body.appendChild(K.n('rect', { x: LX[0], y: LY - 13, width: 16, height: 16, rx: 3, fill: K.TINT.amber,
      stroke: C.amber }));
    K.mono(body, LX[0] + 24, LY, 'own KV, all tokens', { size: 13, color: C.ink2 });
    body.appendChild(K.n('rect', { x: LX[1], y: LY - 13, width: 16, height: 16, rx: 3, fill: K.TINT.amber,
      stroke: C.ink, 'stroke-dasharray': '3 2', 'stroke-width': 1.4 }));
    K.mono(body, LX[1] + 24, LY, 'used to borrow its KV', { size: 13, color: C.ink2 });
    body.appendChild(K.n('rect', { x: LX[2], y: LY - 13, width: 16, height: 16, rx: 3, fill: C.amber }));
    K.mono(body, LX[2] + 24, LY, 'picked, and its source', { size: 13, color: C.ink2 });

    // the short layers: the window goes, nothing is copied (bar lengths illustrative)
    K.label(body, 0, 368, 'short layers: the window is removed, nothing copied');
    var TX = 72, TW = 400, WW = 40;
    [['before', 392, 'last ' + WIN + ' tokens'], ['after', 418, 'whole context']].forEach(function (r, k) {
      K.mono(body, 0, r[1] + 4.5, r[0], { size: 13, color: C.ink2 });
      body.appendChild(K.n('rect', { x: TX, y: r[1] - 6, width: TW, height: 12, rx: 3, fill: K.TINT.ink }));
      body.appendChild(K.n('rect', { x: k ? TX : TX + TW - WW, y: r[1] - 6, width: k ? TW : WW, height: 12,
        rx: 3, fill: k ? K.TINT.amber : CH.GREY, stroke: k ? C.amber : 'none' }));
      K.mono(body, TX + TW + 12, r[1] + 4.5, r[2], { size: 13, color: C.ink2 });
    });

    K.callout(body, 0, 442, 640, MSG[B.l], { cols: 74 });
  }

  // a hit target over each former borrower, outside the redrawn body so it persists
  var hits = K.n('g', {});
  BORROW.forEach(function (b) {
    var c = AT[b.l];
    var r = K.n('rect', { x: c.x - 4, y: c.y - 8, width: CW + 8, height: CHT + 12, rx: 6, fill: 'transparent',
      tabindex: 0, role: 'button', 'aria-label': 'layer ' + b.l + ', used ' + b.how,
      style: 'cursor:pointer;outline:none' });
    r.addEventListener('click', function () { draw(b.l); });
    r.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); draw(b.l); }
    });
    hits.appendChild(r);
  });
  svg.appendChild(hits);
  draw(L7);

  K.foot(svg, 'Schematic on Figure 3\'s layer pattern; window bars not to scale. Full surgery, as ' +
    'tested at 8×10' + sup(F['surg.small_exp']) + ' FLOPs; the full-scale model kept YOCO.');
};
