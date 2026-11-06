/* S_FINALARCH: the paper's Figure 3 layer pattern, before and after the
   conversion described in section 3.2.2. Before: three short sliding-window
   layers (RoPE) per long full-attention layer (NoPE), CLA pairing 7 with 3,
   YOCO pointing 39 and 43 at the anchor 35. After the surgery (DSA came later,
   section 3.2.3): windows and CLA gone, the former SWA layers full-context with
   their own KV and LeRoPE + SSMax, the long layers still NoPE, YOCO kept. A
   schematic; no counts beyond the paper's. */
window.SCENES.S_FINALARCH = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  // "one in every four layers uses full attention": the fact is a word, so map it
  var EVERY = { four: 4 }[F['kv.long_every']] || 4;
  var WIN = F['kv.swa_window'], ANCHOR = F['a2b.anchor'], SHORT = F['a2b.swa_per_group'];
  var svg = K.board(host, { alt: 'Layer schematic after Figure 3, before and after the conversion: which layers see the whole context and which share KV' });
  K.head(svg, 'Almost a full-attention model', 'schematic after Figure 3 · before and after the conversion');
  var body = CH.body(svg);

  // the layers Figure 3 draws: 0 to 7 and 32 to 35 in the first half, 36 to 43 in the second
  var P = 48, CW = 40, CHT = 64, R1 = 112, R2 = 236, XB = 456, LANE = 204;
  var CELLS = [];
  for (var i = 0; i < 8; i++) CELLS.push({ l: i, x: i * P, y: R1 });
  for (var j = 0; j < 4; j++) CELLS.push({ l: ANCHOR - 3 + j, x: XB + j * P, y: R1 });
  for (var m = 0; m < 8; m++) CELLS.push({ l: ANCHOR + 1 + m, x: m * P, y: R2 });
  var AT = {};
  CELLS.forEach(function (c) { AT[c.l] = c; });
  var isLong = function (l) { return l % EVERY === EVERY - 1; };

  // where a long layer gets its KV from, or null if it makes its own
  function source(l, after) {
    if (!isLong(l)) return null;
    if (l > ANCHOR) return ANCHOR;                                    // YOCO, kept
    if (!after && ((l - (EVERY - 1)) / EVERY) % 2 === 1) return l - EVERY;   // CLA, removed
    return null;
  }

  function wave(p, x, y, col, w) {
    var d = '';
    for (var k = 0; k <= 24; k++) d += (k ? 'L' : 'M') + (x + k) + ' ' + (y - 5 * Math.sin(Math.PI * k / 6)).toFixed(1);
    p.appendChild(K.n('path', { d: d, fill: 'none', stroke: col, 'stroke-width': w, 'stroke-linecap': 'round' }));
  }
  function arrowHead(p, x, y, col) {
    p.appendChild(K.n('path', { d: 'M' + x + ' ' + y + ' l-5 -8 h10 Z', fill: col }));
  }
  function cell(p, x, y, w, h, kind) {
    // kind: own (amber), win (white with the window bar), borrow (dashed)
    p.appendChild(K.n('rect', { x: x, y: y, width: w, height: h, rx: Math.min(5, w / 4),
      fill: kind === 'own' ? C.amber : '#fff', stroke: kind === 'own' ? C.amber : CH.GREY,
      'stroke-width': 1.3, 'stroke-dasharray': kind === 'borrow' ? '4 3' : null }));
    if (kind === 'win') p.appendChild(K.n('rect', { x: x + w * 0.15, y: y + h - Math.max(4, h * 0.17),
      width: w * 0.7, height: Math.max(3, h * 0.11), rx: 1.5, fill: CH.GREY }));
  }

  var MSG = [
    'As ML4 began. ' + SHORT + ' short layers in every ' + EVERY + ' see only the last ' + WIN +
      ' tokens, with RoPE. Long layers see everything, with no positional encoding (NoPE). Layer 7 ' +
      'reuses the KV of 3 (CLA); ' + (ANCHOR + 4) + ' and ' + (ANCHOR + 8) + ' reuse ' + ANCHOR + ' (YOCO).',
    'After the surgery. Every former window layer sees the whole context with its own KV, LeRoPE and ' +
      'SSMax. CLA is gone: 7 makes its own KV. Long layers stay NoPE and YOCO stays: ' + (ANCHOR + 4) +
      ' and ' + (ANCHOR + 8) + ' read ' + ANCHOR + '. DSA comes next.'
  ];

  function draw(after) {
    CH.clear(body);
    K.label(body, 0, 58, 'first half');
    K.label(body, 0, R2 - 14, 'second half');
    K.label(body, 640, 58, 'YOCO anchor', { anchor: 'end' });
    // the layers Figure 3 leaves out
    [[400, R1], [400, R2]].forEach(function (d) {
      for (var k = 0; k < 3; k++) body.appendChild(K.n('circle', { cx: d[0] + k * 16, cy: d[1] + CHT / 2,
        r: 2.4, fill: C.ink3 }));
    });
    CELLS.forEach(function (c) {
      var lg = isLong(c.l), src = source(c.l, after);
      cell(body, c.x, c.y, CW, CHT, src != null ? 'borrow' : (lg || after ? 'own' : 'win'));
      if (lg) {
        // a long layer wears a dark cap, as in the paper's figure
        body.appendChild(K.n('rect', { x: c.x, y: c.y - 6, width: CW, height: 3, rx: 1.5, fill: C.ink }));
        K.mono(body, c.x + CW / 2, c.y + CHT / 2 + 5, String(c.l), { size: 14, anchor: 'middle',
          weight: 600, color: src == null ? '#fff' : C.ink3 });
      } else {
        wave(body, c.x + 8, c.y + CHT / 2 - 4, after ? '#fff' : CH.GREY, after ? 2.2 : 1.6);
      }
    });
    // CLA: before, 7 reads 3 along an arc over the first row; after, a faint dashed
    // ghost of that arc marks the link the surgery removed
    var a = AT[EVERY - 1], b = AT[2 * EVERY - 1], ax = a.x + CW / 2, bx = b.x + CW / 2, top = R1 - 8;
    body.appendChild(K.n('path', { d: 'M' + ax + ' ' + top + ' C' + ax + ' ' + (top - 26) + ' ' + bx + ' ' +
      (top - 26) + ' ' + bx + ' ' + (top - 9), fill: 'none', stroke: after ? CH.GREY : C.ink2,
      'stroke-width': after ? 1.3 : 1.6, 'stroke-dasharray': after ? '3 4' : null }));
    if (!after) arrowHead(body, bx, top - 1, C.ink2);
    K.mono(body, (ax + bx) / 2, top - 26, after ? 'CLA removed: 7 makes its own KV' : 'CLA: 7 reads the KV of 3',
      { size: 13, anchor: 'middle', color: after ? C.ink3 : C.ink2 });
    // YOCO, both states: down from the anchor along a lane, into each borrower
    var bor = CELLS.filter(function (c) { return c.y === R2 && source(c.l, after) === ANCHOR; });
    var s = AT[ANCHOR], sx = s.x + CW / 2, far = bor[0].x + CW / 2;
    body.appendChild(K.n('path', { d: 'M' + sx + ' ' + (R1 + CHT + 2) + ' V' + LANE + ' H' + far,
      fill: 'none', stroke: after ? C.amber : C.ink2, 'stroke-width': after ? 2.2 : 1.6 }));
    bor.forEach(function (c) {
      var tx = c.x + CW / 2;
      body.appendChild(K.n('path', { d: 'M' + tx + ' ' + LANE + ' V' + (R2 - 11), stroke: after ? C.amber : C.ink2,
        'stroke-width': after ? 2.2 : 1.6 }));
      arrowHead(body, tx, R2 - 9, after ? C.amber : C.ink2);
    });
    K.mono(body, 400, LANE - 8, (after ? 'YOCO kept: ' : 'YOCO: ') + bor.map(function (c) { return c.l; }).join(' and ') +
      ' read ' + ANCHOR + '\'s KV', { size: 13, anchor: 'middle', weight: after ? 600 : 400, color: after ? C.ink : C.ink2 });

    // legend: what a cell's fill says about KV, and what its glyph says about position
    var LA = 336, LB = 366, LX = [0, 190, 420];
    cell(body, LX[0], LA - 13, 16, 16, 'own');
    K.mono(body, LX[0] + 24, LA, 'own KV, all tokens', { size: 13, color: C.ink2 });
    cell(body, LX[1], LA - 13, 16, 16, 'win');
    K.mono(body, LX[1] + 24, LA, 'own KV, last ' + WIN, { size: 13, color: C.ink2 });
    cell(body, LX[2], LA - 13, 16, 16, 'borrow');
    K.mono(body, LX[2] + 24, LA, 'reads another layer\'s KV', { size: 13, color: C.ink2 });
    cell(body, LX[0], LB - 13, 16, 16, 'win');
    body.appendChild(K.n('path', { d: 'M' + (LX[0] + 3) + ' ' + (LB - 6) + ' q3 -6 5 0 t5 0', fill: 'none',
      stroke: CH.GREY, 'stroke-width': 1.6 }));
    K.mono(body, LX[0] + 24, LB, 'RoPE, fixed', { size: 13, color: C.ink2 });
    cell(body, LX[1], LB - 13, 16, 16, 'own');
    body.appendChild(K.n('path', { d: 'M' + (LX[1] + 3) + ' ' + (LB - 5) + ' q3 -6 5 0 t5 0', fill: 'none',
      stroke: '#fff', 'stroke-width': 1.8 }));
    K.mono(body, LX[1] + 24, LB, 'LeRoPE + SSMax, learned', { size: 13, color: C.ink2 });
    // the cap is the only thing that marks a long layer, so the key shows the cap over
    // just the top edge of a cell, open below, not a whole cell of a type that never appears
    body.appendChild(K.n('path', { d: 'M' + LX[2] + ' ' + (LB + 2) + ' V' + (LB - 8) + ' H' + (LX[2] + 16) + ' V' + (LB + 2),
      fill: 'none', stroke: CH.GREY, 'stroke-width': 1.3 }));
    body.appendChild(K.n('rect', { x: LX[2], y: LB - 14, width: 16, height: 3, rx: 1.5, fill: C.ink }));
    K.mono(body, LX[2] + 24, LB, 'cap: long layer, NoPE', { size: 13, color: C.ink2 });

    // the meter: per group of four, how many layers attend over the whole context
    var full = after ? EVERY : 1;
    K.label(body, 0, 404, 'layers per group of ' + EVERY + ' that see the whole context');
    for (var q = 0; q < EVERY; q++) body.appendChild(K.n('rect', { x: 484 + q * 22, y: 391, width: 16, height: 16,
      rx: 3, fill: q < full ? C.amber : '#fff', stroke: q < full ? C.amber : CH.GREY }));
    K.mono(body, 640, 404, full + ' of ' + EVERY, { size: 14, anchor: 'end', weight: 600, color: C.ink });
    K.callout(body, 0, 424, 640, MSG[after ? 1 : 0], { cols: 74 });
  }

  K.switcher(svg, 0, 0, ['before: Figure 3', 'after: converted'], function (i) { draw(i === 1); },
    { tint: 'amber' });
  K.foot(svg, 'Schematic after the paper\'s Figure 3, which also leaves out layers 8 to 31; the dots stand ' +
    'for layers not drawn. The paper does not give the total layer count.');
};
