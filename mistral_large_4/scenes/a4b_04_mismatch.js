/* S_MISMATCH: the three fixes for train and inference disagreeing (§4.4.2):
   router replay, FP8 with blockwise scales on both sides, and retokenization.
   The retokenization example ("the" + "re" becoming "there") and the k2 bound
   are the paper's. The expert count, routing picks and FP8 block size are
   schematic: the paper gives none of them for this section. */
window.SCENES.S_MISMATCH = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var GEN = [F['a4b.tok_a'], F['a4b.tok_b']], CTX = [F['a4b.tok_ab']];
  var SUP = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  var sup = function (s) { return String(s).split('').map(function (c) { return SUP[c] || c; }).join(''); };
  var svg = K.board(host, { alt: 'Three fixes that reduce the mismatch between the trainer and the generator' });
  K.head(svg, 'Making trainer and generator agree', '§4.4.2 · ' + F['a4b.levels'] +
    ' fixes · expert picks and FP8 blocks are schematic');

  var view = 2, replay = false, retok = false;
  var body = K.n('g', {}), note = K.n('g', {});
  var LX = 236;                 // where the drawing column starts; labels sit left of it
  var BTN = { x: 0, y: 252, w: 214 };

  var TX = 172;                 // text column in the retokenization view
  function rowText(y, a, b) {
    K.mono(body, TX, y, a, { size: 13.5, weight: 600, color: C.ink });
    K.mono(body, TX, y + 19, b, { size: 12.5, color: C.ink2 });
  }
  function toggle(label, fn) {
    CH.button(body, BTN.x, BTN.y, label, function () { fn(); draw(); }, { w: BTN.w });
  }

  /* view 1: one token's experts, schematic. The generator picks two of eight. */
  var NE = 8, PICK_GEN = [1, 5], PICK_DRIFT = [1, 6], EW = 40, EG = 8;
  function experts(y, lit, ghost) {
    for (var i = 0; i < NE; i++) {
      var on = lit.indexOf(i) >= 0, gh = ghost && ghost.indexOf(i) >= 0;
      var drift = on && PICK_GEN.indexOf(i) < 0;
      body.appendChild(K.n('rect', { x: LX + i * (EW + EG), y: y, width: EW, height: EW, rx: 7,
        fill: on ? (drift ? C.blue : C.amber) : gh ? '#fff' : CH.GREY2,
        stroke: gh ? C.blue : 'none', 'stroke-width': gh ? 2 : 0, 'stroke-dasharray': gh ? '4 3' : null }));
    }
  }
  function drawReplay() {
    K.label(body, LX, 62, 'experts for one token, schematic');
    K.mono(body, 0, 100, 'generator, at inference', { size: 13, weight: 600, color: C.ink });
    experts(74, PICK_GEN);
    K.mono(body, 0, 168, replay ? 'trainer, routing replayed' : 'trainer, routing recomputed',
      { size: 13, weight: 600, color: C.ink });
    var lost = PICK_GEN.filter(function (i) { return PICK_DRIFT.indexOf(i) < 0; });
    experts(142, replay ? PICK_GEN : PICK_DRIFT, replay ? null : lost);
    var ms = replay ? 'same experts as at inference' : 'router drift: one expert differs';
    K.mono(body, LX, 212, ms, { size: 13, weight: 600, color: C.ink });
    K.mono(body, LX, 231, replay ? 'routing adds no divergence' :
      'dashed: the expert the generator used', { size: 12.5, color: C.ink2 });
    toggle(replay ? 'turn off router replay' : 'turn on router replay', function () { replay = !replay; });
  }

  /* view 2: values in blocks, each block with its own scale, schematic. The paper
     says both sides use blockwise scales; it does not give the block size or the
     values, so "=" means the same scheme, not measured equal numbers. */
  var FB = 36, FG = 4, FY = 80, FW = 4 * FB + 3 * FG;
  function grid(x, y, title) {
    K.label(body, x, y - 10, title);
    var rnd = CH.rng(19);
    for (var r = 0; r < 4; r++) for (var c = 0; c < 4; c++) {
      body.appendChild(K.n('rect', { x: x + c * (FB + FG), y: y + r * (FB + FG), width: FB, height: FB, rx: 3,
        fill: C.amber, 'fill-opacity': (0.18 + 0.72 * rnd()).toFixed(2) }));
    }
  }
  function drawFP8() {
    var x2 = LX + FW + 48;
    grid(LX, FY, 'generator, fp8');
    K.big(body, LX + FW + 24, FY + FW / 2 + 12, '=', { size: 34, color: C.ink3, anchor: 'middle' });
    grid(x2, FY, 'trainer, fp8');
    K.mono(body, 0, FY + FW / 2 - 19, 'one block of values,', { size: 13, color: C.ink2 });
    K.mono(body, 0, FY + FW / 2, 'one scale per block;', { size: 13, color: C.ink2 });
    K.mono(body, 0, FY + FW / 2 + 19, 'shade is that scale', { size: 13, color: C.ink2 });
    K.mono(body, LX, FY + FW + 26, 'blockwise FP8 scaling on both sides', { size: 13, weight: 600, color: C.ink });
    K.mono(body, LX, FY + FW + 45, 'faster generation, and it matches the trainer', { size: 12.5, color: C.ink2 });
  }

  /* view 3: the paper's retokenization example */
  function chip(x, y, s, col, dashed) {
    var w = s.length * 13.7 + 26;
    body.appendChild(K.n('rect', { x: x, y: y, width: w, height: 40, rx: 8,
      fill: dashed ? '#fff' : (col === C.amber ? K.TINT.amber : K.TINT.blue),
      stroke: dashed ? CH.GREY : col, 'stroke-width': dashed ? 1.4 : 2, 'stroke-dasharray': dashed ? '5 4' : null }));
    K.mono(body, x + w / 2, y + 28, s, { size: 22, anchor: 'middle', weight: 600,
      color: dashed ? C.ink3 : C.ink });
    return w;
  }
  function drawRetok() {
    var x = 0;
    GEN.forEach(function (t) { x += chip(x, 60, t, C.amber) + 8; });
    rowText(76, 'as generated: ' + GEN.length + ' tokens', 'each has its own probability ratio');
    K.arrow(body, 52, 104, 52, 126, { color: C.ink3, w: 1.4 });
    K.mono(body, 14, 152, GEN.join(''), { size: 22, weight: 600, color: C.ink });
    body.appendChild(K.n('path', { d: 'M8 160 H112', stroke: C.ink3, 'stroke-width': 1.2,
      'stroke-dasharray': '2 3' }));
    rowText(142, 'sent into the RL env as a string', 'as plain text, the token boundary is gone');
    K.arrow(body, 52, 168, 52, 190, { color: C.ink3, w: 1.4 });
    if (retok) {
      chip(0, 194, CTX[0], C.blue);
      rowText(210, 'as context: ' + CTX.length + ' token', 'the same text, a different sequence');
    } else {
      chip(0, 194, '?', null, true);
      rowText(210, 'as context: not retokenized yet', 'press the button below');
    }
    toggle(retok ? 'show it as generated' : 'retokenize the string', function () { retok = !retok; });
  }

  var NOTE = [
    [ // router replay: recomputed, replayed
      'Recomputed in the trainer, the router can send a token to a different expert than the ' +
        'generator did, so the same token runs through different weights. Turn on router replay.',
      'Router replay re-executes the routing decisions made at inference in the training forward ' +
        'pass, so router drift cannot cause divergence.'
    ],
    [
      'FP8 inference speeds up generation and matches the trainer, and both the generator and the ' +
        'trainer use blockwise scales. The paper does not give the block size; this one is schematic.'
    ],
    [ // retokenization: as generated, retokenized
      'The generator produced "' + GEN.join('" and "') + '" as ' + GEN.length + ' tokens. The ' +
        'environment receives the string, and the token boundary is gone. Retokenize it to see what ' +
        'it becomes as context.',
      'Tokenized again, the string is the single token "' + CTX[0] + '". The same text is one ' +
        'sequence as generated and another as context. ML4 handles both efficiently with ' +
        'blockwise attention from FA4.'
    ]
  ];

  function draw() {
    CH.clear(body);
    if (view === 0) drawReplay(); else if (view === 1) drawFP8(); else drawRetok();
    CH.clear(note);
    var st = view === 0 ? (replay ? 1 : 0) : view === 2 ? (retok ? 1 : 0) : 0;
    K.callout(note, 0, 396, 640, NOTE[view][st], { cols: 74 });
  }

  var pk = CH.pick(svg, 0, 0, ['router replay', 'FP8, blockwise scales', 'retokenization'], function (i) {
    view = i;
    if (body.parentNode) draw();
  }, { horizontal: true, w: 202, gap: 17, h: 30 });
  svg.appendChild(body);   // after the view picker, so its toggle comes after it in tab order

  // the run-level result, the same under every view
  K.panel(svg, 0, 302, 640, 78, { r: 10 });
  K.big(svg, 18, 354, '≤ ' + F['mis.k2_m'] + '×10⁻' + sup(F['mis.k2_e']),
    { size: 36, color: C.amber });
  K.mono(svg, 214, 330, 'average k2 (a KL estimate), train vs inference', { size: 13, weight: 600, color: C.ink });
  K.mono(svg, 214, 349, 'stayed at or below this for the whole run', { size: 12.5, color: C.ink2 });
  K.mono(svg, 214, 368, 'average staleness: ' + F['mis.stale'] + ' steps', { size: 12.5, color: C.ink2 });
  // reserve room for a three-line callout, so the foot (placed once) never lands on it
  svg.appendChild(K.n('rect', { x: 0, y: 396, width: 640, height: 3 * 19 + 22, fill: 'none' }));
  svg.appendChild(note);
  pk.pick(2);

  K.foot(svg, 'The example tokens, the divergence bound and the staleness are the paper\'s. The expert count, ' +
    'routing picks and FP8 block size are schematic; the paper gives none of them.');
};
