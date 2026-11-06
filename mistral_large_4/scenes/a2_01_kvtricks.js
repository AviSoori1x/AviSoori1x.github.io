/* S_KVTRICKS: Figure 3 redrawn as a schematic, with three switches. The layer
   pattern is the paper's: one long (full-attention) layer in every four, the
   first half ending at layer 35, CLA pairing 7 with 3, YOCO pointing the second
   half at 35. The meter is my arithmetic on that pattern (window caches of 512
   tokens ignored); with all three on it lands on the paper's own 16x. */
window.SCENES.S_KVTRICKS = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  // "one in every four layers uses full attention": the fact is the word, so map it
  var EVERY = { four: 4 }[F['kv.long_every']] || 4;
  var WIN = F['kv.swa_window'], RED = F['kv.reduction'];
  var svg = K.board(host, { alt: 'Layer schematic after Figure 3: sliding windows, CLA and YOCO and which layers keep their own KV cache' });
  K.head(svg, 'Three ways to shrink the KV cache', 'schematic after Figure 3 · the layer pattern ML4 began with');
  var body = CH.body(svg);

  // Figure 3's own labels: 35 is the YOCO anchor, 7 reuses the KV of 3, 39 and 43 reuse 35's
  var ANCHOR = F['a2.yoco_anchor'], CLA_L = F['a2.cla_layer'], CLA_S = F['a2.cla_src'];
  var YOCO_L = [F['a2.yoco_l1'], F['a2.yoco_l2']];
  // the layers Figure 3 draws: the first two groups of four, the group that ends at the
  // anchor (0 to 7 and 32 to 35), then the two groups after it (36 to 43)
  var P = 48, CW = 40, CHT = 64, R1 = 82, R2 = 226, XB = 456;
  var CELLS = [];
  for (var i = 0; i < 2 * EVERY; i++) CELLS.push({ l: i, x: i * P, y: R1 });
  for (var j = 0; j < EVERY; j++) CELLS.push({ l: ANCHOR - EVERY + 1 + j, x: XB + j * P, y: R1 });
  for (var m = 0; m < 2 * EVERY; m++) CELLS.push({ l: ANCHOR + 1 + m, x: m * P, y: R2 });
  var AT = {};
  CELLS.forEach(function (c) { AT[c.l] = c; });
  var isLong = function (l) { return l % EVERY === EVERY - 1; };

  var on = { swa: false, cla: false, yoco: false }, last = null;
  // where a long layer gets its KV from, or null if it makes its own
  function source(l) {
    if (!isLong(l)) return null;
    if (on.yoco && l > ANCHOR) return { from: ANCHOR, how: 'YOCO' };
    if (on.cla && ((l - (EVERY - 1)) / EVERY) % 2 === 1) return { from: l - EVERY, how: 'CLA' };
    return null;
  }
  // share of full attention's KV that is still stored, per layer of the network:
  // short layers (3 in 4) keep the whole context unless windowed; long layers
  // (1 in 4) are halved by CLA, and YOCO drops the second half of them
  function kvShare() {
    var long1 = on.cla ? 0.5 : 1, long2 = on.yoco ? 0 : long1;
    return (on.swa ? 0 : (EVERY - 1) / EVERY) + (1 / EVERY) * (0.5 * long1 + 0.5 * long2);
  }

  function arrowHead(p, x, y, col) {
    p.appendChild(K.n('path', { d: 'M' + x + ' ' + y + ' l-5 -8 h10 Z', fill: col }));
  }

  var MSG = {
    none: 'Full attention: every layer keeps keys and values for the whole context. Turn the ' +
      'tricks on one at a time.',
    swa: 'Sliding windows: three layers in four keep keys and values for the last ' + WIN +
      ' tokens only. The long layers still hold the whole context.',
    swaOff: 'Without windows, the short layers keep the whole context again.',
    cla: 'CLA: every other long layer reuses the KV of the long layer before it. Layer ' + CLA_L +
      ' stores no KV of its own and reads layer ' + CLA_S + '\'s.',
    // the two "off" messages depend on whether the other sharing trick is still on
    claOff: function () {
      return on.yoco ? 'Without CLA, every long layer in the first half makes its own KV again. The second half still ' +
        'reads layer ' + ANCHOR + '\'s through YOCO.' : 'Without CLA, every long layer makes its own KV again.';
    },
    yoco: 'YOCO: long layers in the second half read the KV made at layer ' + ANCHOR + ', the ' +
      'halfway point. It also allows a two-stage prefill, ' + F['kv.yoco_prefill'] + '× faster at ' +
      F['kv.yoco_ctx_m'] + 'M context.',
    yocoOff: function () {
      return on.cla ? 'Without YOCO, the second half falls back to CLA pairs: ' + YOCO_L[0] + ' reads ' +
        ANCHOR + '\'s KV and ' + YOCO_L[1] + ' makes its own.' : 'Without YOCO, the second half makes its own KV again.';
    },
    all: 'All three on, as ML4 began: ' + RED + '× less KV cache than full attention. Early ' +
      'ablations said the cost in quality was negligible or small.'
  };

  function draw() {
    CH.clear(body);
    // short row labels, so the CLA arc and the YOCO drops never cross them
    K.label(body, 0, 68, 'first half');
    K.label(body, 0, 212, 'second half');
    // the layers Figure 3 skips
    [[400, R1], [400, R2]].forEach(function (d) {
      for (var k = 0; k < 3; k++) body.appendChild(K.n('circle', { cx: d[0] + k * 16, cy: d[1] + CHT / 2,
        r: 2.4, fill: C.ink3 }));
    });
    CELLS.forEach(function (c) {
      var lg = isLong(c.l), src = source(c.l), own = lg ? !src : !on.swa;
      body.appendChild(K.n('rect', { x: c.x, y: c.y, width: CW, height: CHT, rx: 5,
        fill: own ? C.amber : '#fff', stroke: own ? C.amber : CH.GREY,
        'stroke-width': 1.3, 'stroke-dasharray': src ? '4 3' : null }));
      if (!lg && on.swa) body.appendChild(K.n('rect', { x: c.x + 4, y: c.y + CHT - 11, width: CW - 8,
        height: 7, rx: 2, fill: CH.GREY }));
      if (lg) {
        // a long layer wears a dark cap, as in the paper's figure
        body.appendChild(K.n('rect', { x: c.x, y: c.y - 6, width: CW, height: 3, rx: 1.5, fill: C.ink }));
        K.mono(body, c.x + CW / 2, c.y + CHT / 2 + 5, String(c.l), { size: 14, anchor: 'middle',
          weight: 600, color: own ? '#fff' : C.ink3 });
      }
    });
    // CLA inside the first half: 7 reads 3, drawn as an arc over the row
    if (on.cla) {
      var a = AT[CLA_S], b = AT[CLA_L], ax = a.x + CW / 2, bx = b.x + CW / 2, top = R1 - 8;
      body.appendChild(K.n('path', { d: 'M' + ax + ' ' + top + ' C' + ax + ' ' + (top - 26) + ' ' + bx + ' ' +
        (top - 26) + ' ' + bx + ' ' + (top - 9), fill: 'none', stroke: C.ink2, 'stroke-width': 1.6 }));
      arrowHead(body, bx, top - 1, C.ink2);
      K.mono(body, (ax + bx) / 2, top - 26, 'CLA: ' + CLA_L + ' reads the KV of ' + CLA_S,
        { size: 13, anchor: 'middle', color: C.ink2 });
    }
    // the second half: down from 35 along a lane, then into each borrower
    var borrowers = YOCO_L.filter(function (l) { return source(l); });
    if (borrowers.length) {
      var s = AT[ANCHOR], sx = s.x + CW / 2, lane = 176, far = AT[borrowers[0]].x + CW / 2;
      body.appendChild(K.n('path', { d: 'M' + sx + ' ' + (R1 + CHT + 2) + ' V' + lane + ' H' + far,
        fill: 'none', stroke: C.ink2, 'stroke-width': 1.6 }));
      borrowers.forEach(function (l) {
        var tx = AT[l].x + CW / 2;
        body.appendChild(K.n('path', { d: 'M' + tx + ' ' + lane + ' V' + (R2 - 9), stroke: C.ink2, 'stroke-width': 1.6 }));
        arrowHead(body, tx, R2 - 7, C.ink2);
      });
      var how = source(borrowers[0]).how;
      K.mono(body, 500, lane - 8, how + ': ' + borrowers.join(' and ') + (borrowers.length > 1 ? ' read ' : ' reads ') +
        ANCHOR + '\'s KV', { size: 13, anchor: 'middle', color: C.ink2 });
      if (how === 'YOCO') K.label(body, 640, 68, 'halfway point', { anchor: 'end' });
    }

    // legend: what a cell's fill means
    var LX = [0, 196, 384], LY = 326;
    body.appendChild(K.n('rect', { x: LX[0], y: LY - 13, width: 16, height: 16, rx: 3, fill: C.amber }));
    K.mono(body, LX[0] + 24, LY, 'own KV, all tokens', { size: 13, color: C.ink2 });
    body.appendChild(K.n('rect', { x: LX[1], y: LY - 13, width: 16, height: 16, rx: 3, fill: '#fff', stroke: CH.GREY }));
    body.appendChild(K.n('rect', { x: LX[1] + 3, y: LY - 3, width: 10, height: 4, rx: 1, fill: CH.GREY }));
    K.mono(body, LX[1] + 24, LY, 'own KV, last ' + WIN, { size: 13, color: C.ink2 });
    body.appendChild(K.n('rect', { x: LX[2], y: LY - 13, width: 16, height: 16, rx: 3, fill: '#fff',
      stroke: CH.GREY, 'stroke-dasharray': '3 2' }));
    K.mono(body, LX[2] + 24, LY, 'reads another layer\'s KV', { size: 13, color: C.ink2 });

    // the meter: KV still stored, against full attention
    // big cuts read as a factor (4x, 8x, the paper's 16x); the small ones (CLA or YOCO
    // without windows) read better as the share that is left: 87.5% or 81.3%
    var share = kvShare(), ratio = 1 / share, all = on.swa && on.cla && on.yoco;
    var rtxt = share === 1 ? 'full attention, the baseline' :
      ratio < 2 ? CH.fmt(share * 100, 1) + '% of full attention' :
      (all ? RED : (Math.abs(ratio - Math.round(ratio)) < 1e-9 ? Math.round(ratio) : CH.fmt(ratio, 1))) +
      '× less than full attention';
    K.label(body, 0, 374, 'KV cache at long context');
    K.mono(body, 640, 374, rtxt, { size: 14, anchor: 'end', weight: 600, color: C.ink });
    K.bar(body, 0, 386, 640, 16, share, { color: C.amber });
    var m = MSG[all ? 'all' : (last || 'none')];
    K.callout(body, 0, 424, 640, typeof m === 'function' ? m() : m, { cols: 74 });
  }

  // three independent on/off switches (K.switcher picks one of many, so a local helper)
  var TOG = [['swa', 'sliding windows'], ['cla', 'CLA'], ['yoco', 'YOCO']], tx0 = 0, togs = [];
  TOG.forEach(function (t) {
    var w = t[1].length * 8.1 + 46;
    var g = K.n('g', { tabindex: 0, role: 'button', 'aria-label': t[1], 'aria-pressed': 'false',
      style: 'cursor:pointer;outline:none' });
    var r = K.n('rect', { x: tx0, y: 0, width: w, height: 27, rx: 13.5, fill: 'none', stroke: C.line });
    var dot = K.n('circle', { cx: tx0 + 16, cy: 13.5, r: 5, fill: 'none', stroke: C.ink3, 'stroke-width': 1.4 });
    g.appendChild(r);
    g.appendChild(dot);
    var lab = K.mono(g, tx0 + 28, 18, t[1], { size: 13, color: C.ink3 });
    function paint() {
      var v = on[t[0]];
      r.setAttribute('fill', v ? K.TINT.amber : 'none');
      r.setAttribute('stroke', v ? C.amber : C.line);
      dot.setAttribute('fill', v ? C.amber : 'none');
      dot.setAttribute('stroke', v ? C.amber : C.ink3);
      lab.setAttribute('fill', v ? C.amber : C.ink3);
      g.setAttribute('aria-pressed', v ? 'true' : 'false');
    }
    function flip() {
      on[t[0]] = !on[t[0]];
      last = on[t[0]] ? t[0] : t[0] + 'Off';
      paint();
      draw();
    }
    g.addEventListener('click', flip);
    g.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); flip(); }
    });
    svg.appendChild(g);
    togs.push(g);
    tx0 += w + 8;
  });
  draw();

  K.foot(svg, 'Schematic after the paper\'s Figure 3, which also skips layers 8 to 31. The meter is ' +
    'my arithmetic on this layer pattern and ignores the small ' + WIN + '-token window caches; the ' +
    'paper states only the combined ' + RED + '×.');
};
