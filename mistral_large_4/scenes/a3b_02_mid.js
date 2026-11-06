/* S_MID: midtraining hyperparameters (§4.2.1). The paper gives no trade-off curve,
   no scores and no grid of runs. It states the objective, how the learning rate was
   parameterized, the two chosen ratios at ML4 scale and the batch-size call. So the
   figure lays the decision out as stated and draws only the two ratios to scale
   (ratios need no absolute values). The batch view borrows the pretraining ramp
   numbers (§2.3.1), labelled as such, because midtraining reports no batch value. */
window.SCENES.S_MID = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var LRD = F['mid.lr_div'], RHD = F['mid.rehearsal_div'];
  var svg = K.board(host, { alt: 'Midtraining hyperparameters: the objective, and the learning rate (1/' +
    LRD + ' of the pretraining peak), rehearsal (1/' + RHD + ' of pretraining rehearsal) and batch size choices' });
  K.head(svg, 'Midtraining: three knobs', '§4.2.1 · one objective · values at ML4 scale · schematic');

  // a white box with a bold first line and a quieter second line
  function box(g, x, y, w, h, t1, t2, o) {
    o = o || {};
    g.appendChild(K.n('rect', { x: x, y: y, width: w, height: h, rx: 9,
      fill: o.fill || '#fff', stroke: o.stroke || C.line, 'stroke-width': o.sw || 1 }));
    K.mono(g, x + 14, y + h / 2 - 4, t1, { size: 13.5, weight: 600, color: C.ink });
    K.mono(g, x + 14, y + h / 2 + 15, t2, { size: 13, color: C.ink3 });
  }
  // an operator sign between two boxes, drawn as a path so it costs no text
  function op(g, cx, cy, kind) {
    var s = 6, d = kind === 'x' ?
      'M' + (cx - s) + ' ' + (cy - s) + ' L' + (cx + s) + ' ' + (cy + s) + ' M' + (cx + s) + ' ' + (cy - s) +
        ' L' + (cx - s) + ' ' + (cy + s) :
      kind === '+' ? 'M' + (cx - s - 1) + ' ' + cy + ' H' + (cx + s + 1) + ' M' + cx + ' ' + (cy - s - 1) +
        ' V' + (cy + s + 1) :
      'M' + (cx - s - 1) + ' ' + (cy - 3.5) + ' H' + (cx + s + 1) + ' M' + (cx - s - 1) + ' ' + (cy + 3.5) +
        ' H' + (cx + s + 1);
    g.appendChild(K.n('path', { d: d, stroke: C.ink2, 'stroke-width': 2.2, 'stroke-linecap': 'round',
      fill: 'none' }));
  }

  // the objective, the same in every view
  var top = CH.body(svg);
  K.label(top, 0, 60, 'what every setting was scored on');
  box(top, 0, 72, 296, 54, 'new capabilities', 'such as agentic skills');
  box(top, 344, 72, 296, 54, 'retained knowledge', 'the long tail from pretraining');
  op(top, 320, 99, '+');
  top.appendChild(K.n('path', { d: 'M148 128 V140 H492 V128 M320 140 V150', stroke: C.ink3,
    'stroke-width': 1.3, fill: 'none' }));
  top.appendChild(K.n('rect', { x: 0, y: 152, width: 640, height: 34, rx: 8, fill: K.TINT.ink,
    stroke: C.line }));
  K.mono(top, 320, 174, 'a weighted average of both evaluation suites', { size: 13.5, anchor: 'middle',
    color: C.ink, weight: 500 });

  var body = CH.body(svg);

  // geometry shared by the views
  var BX = 128, BW = 512;               // ratio bars: x and width of the 1x reference
  var FW = 196, FG = 26;                // flow boxes: width and gap
  var FX = [0, FW + FG, 2 * (FW + FG)];

  // a reference bar cut into div equal segments, and the chosen value as one of them
  function ratio(g, y, div, valText, refName) {
    K.mono(g, 0, y + 18, refName || 'pretraining', { size: 13.5, color: C.ink2 });
    K.mono(g, 0, y + 56, 'midtraining', { size: 13.5, color: C.ink, weight: 600 });
    g.appendChild(K.n('rect', { x: BX, y: y, width: BW, height: 26, rx: 5, fill: CH.GREY2 }));
    for (var i = 1; i < div; i++) {
      var sx = BX + BW * i / div;
      g.appendChild(K.n('path', { d: 'M' + sx + ' ' + y + ' V' + (y + 26), stroke: '#fff',
        'stroke-width': 1.6 }));
    }
    var w = BW / div;   // the chosen ratio, 1/div of the reference, drawn to scale
    g.appendChild(K.n('path', { d: 'M' + (BX + w) + ' ' + (y + 26) + ' V' + (y + 38), stroke: C.ink3,
      'stroke-width': 1.2, 'stroke-dasharray': '3 3' }));
    g.appendChild(K.n('rect', { x: BX, y: y + 38, width: w, height: 26, rx: 5, fill: C.amber }));
    K.mono(g, BX + w + 12, y + 56, valText, { size: 13.5, color: C.ink, weight: 600 });
  }

  // three boxes in a row. With ops, they are joined by arithmetic signs (a op b op c).
  // Without ops, the first two are bracketed together and feed the third: no sum implied.
  function flow(g, y, items, ops, hot) {
    items.forEach(function (it, i) {
      var on = i === hot;
      box(g, FX[i], y, FW, 58, it[0], it[1], on ? { fill: K.TINT.amber, stroke: C.amber, sw: 1.6 } : {});
      if (ops && i < 2) op(g, FX[i] + FW + FG / 2, y + 29, ops[i]);
    });
    if (ops) return;
    var b = y + 58, j = b + 12, c0 = FX[0] + FW / 2, c1 = FX[1] + FW / 2, c2 = FX[2] + FW / 2;
    g.appendChild(K.n('path', { d: 'M' + c0 + ' ' + b + ' V' + j + ' H' + c2 + ' V' + (b + 3) +
      ' M' + c1 + ' ' + b + ' V' + j, stroke: C.ink3, 'stroke-width': 1.3, fill: 'none' }));
    g.appendChild(K.n('path', { d: 'M' + (c2 - 5) + ' ' + (b + 7) + ' L' + c2 + ' ' + (b + 1) + ' L' +
      (c2 + 5) + ' ' + (b + 7) + ' Z', fill: C.ink3 }));
  }

  // reserve the callout, so the honesty line (placed once) never lands on it
  svg.appendChild(K.n('rect', { x: 0, y: 420, width: 640, height: 3 * 19 + 22, fill: 'none' }));

  var NOTE = [
    'At ML4 scale, midtraining uses 1/' + LRD + ' of the peak pretraining learning rate. The ' +
      'multiplier is fitted on top of the scaling-law prediction, the form that transferred best ' +
      'across scale.',
    'The paper says "1/' + RHD + ' of pretraining rehearsal" without naming the reference, so the ' +
      'bar shows only the ratio. Tuned with the learning rate, this pair was the best compromise ' +
      'at ML4 scale.',
    'Keeping the batch large cost a very small regret on the objective, and the MFU gains ' +
      'more than offset it. The paper gives no midtraining batch size, so none is drawn.'
  ];

  function draw(s) {
    CH.clear(body);
    if (s === 0) {
      K.label(body, 0, 222, 'peak learning rate, drawn to scale');
      ratio(body, 234, LRD, '1/' + LRD + ' of the peak');
      K.label(body, 0, 322, 'how the learning rate is set');
      flow(body, 334, [['scaling-law LR', 'from pretraining'], ['a fitted multiplier', 'transferred best'],
        ['midtraining LR', '1/' + LRD + ' of peak at ML4']], ['x', '='], 2);
    } else if (s === 1) {
      // the paper gives the ratio but never says what it is a fraction of, so the top
      // bar is a generic reference, not a pretraining rehearsal level
      K.label(body, 0, 222, 'rehearsal rate, the ratio drawn to scale');
      ratio(body, 234, RHD, '1/' + RHD + ' of the reference', 'reference');
      K.label(body, 0, 322, 'tuned jointly with the learning rate');
      // a bracket, not + and =: the two rates are chosen together, not added
      flow(body, 334, [['learning rate', '1/' + LRD + ' of the peak'],
        ['rehearsal rate', '1/' + RHD + ' of the reference'], ['best compromise', 'at ML4 scale']],
        null, 2);
    } else {
      K.label(body, 0, 222, 'batch size: kept large');
      // two stated facts side by side, no sign between them: they are in different units
      box(body, 0, 234, 296, 58, 'a very small regret', 'on the objective');
      box(body, 344, 234, 296, 58, 'higher MFU', 'which more than offsets it',
        { fill: K.TINT.amber, stroke: C.amber, sw: 1.6 });
      batchRef(body);
    }
    K.callout(body, 0, 420, 640, NOTE[s], { cols: 74 });
  }

  // the pretraining batch ramp (§2.3.1) made a similar trade, and its numbers are reported.
  // These are ramp-up schedules, not ranges: the batch grows in stages from the first value
  // to the second. Ramping 0.25x up to 2x the predicted batch was best for optimization;
  // 1x up to 4x was chosen for wall-clock. Each bar ends in an arrowhead to show the ramp.
  function batchRef(g) {
    var lo = F['batch.opt_from'], hi = F['batch.opt_to'], c0 = F['batch.chosen_from'],
      c1 = F['batch.chosen_to'];
    var sx = CH.scale(0, c1, 214, 626);
    K.label(g, 0, 322, 'a similar trade: the pretraining batch ramp (§2.3.1)');
    [[lo, hi, 332, 'best ramp: ' + lo + '× up to ' + hi + '×', CH.GREY, C.ink2],
      [c0, c1, 354, 'chosen ramp: ' + c0 + '× up to ' + c1 + '×', C.amber, C.ink]].forEach(function (r) {
      var x0 = sx(r[0]), x1 = sx(r[1]), y = r[2];
      g.appendChild(K.n('path', { d: 'M' + (x0 + 5) + ' ' + (y + 2) + ' H' + (x1 - 10) + ' V' + (y - 1) +
        ' L' + x1 + ' ' + (y + 7) + ' L' + (x1 - 10) + ' ' + (y + 15) + ' V' + (y + 12) + ' H' + (x0 + 5) +
        ' A5 5 0 0 1 ' + (x0 + 5) + ' ' + (y + 2) + ' Z', fill: r[4] }));
      K.mono(g, 0, y + 11.5, r[3], { size: 13, color: r[5], weight: r[5] === C.ink ? 600 : 400 });
    });
    g.appendChild(K.n('path', { d: 'M' + sx(0) + ' 378 H' + sx(c1), stroke: C.ink3, 'stroke-width': 1 }));
    K.mono(g, 0, 396, '× predicted batch', { size: 12.5, color: C.ink3 });
    for (var t = 0; t <= c1; t++) {
      g.appendChild(K.n('path', { d: 'M' + sx(t) + ' 378 v4', stroke: C.ink3, 'stroke-width': 1 }));
      K.mono(g, sx(t), 396, t ? t + '×' : '0', { size: 12.5, anchor: 'middle', color: C.ink3 });
    }
  }

  K.switcher(svg, 0, 0, ['learning rate', 'rehearsal', 'batch size'], draw, { tint: 'amber' });
  K.foot(svg, 'Schematic. The two ratios and the §2.3.1 batch ramps are drawn to scale. The paper gives no ' +
    'absolute midtraining values, no averaging weights and no trade-off curve.');
};
