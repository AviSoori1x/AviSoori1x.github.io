/* S_LADDER: a log-linear learning-rate predictor, a batch-size predictor, and the scaling ladder that tests them.
   Which predictor leans on which input is from §2.1; the ladder's rungs are a
   schematic, since the paper does not list their sizes. */
window.SCENES.S_LADDER = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH;
  var svg = K.board(host, { alt: 'Five inputs feeding a batch-size predictor and a learning-rate predictor' });
  K.head(svg, 'Predicted, not searched', 'log-linear learning rate · batch size predicted first · tested on a ladder');

  var INS = ['width', 'depth', 'training horizon', 'expansion rate', 'batch size'];
  var IX = 170, BY = 58, LY = 186, PX = 236, PW = 206;
  var wires = K.n('g', {});
  svg.appendChild(wires);

  // the two predictors
  K.panel(svg, PX, BY - 30, PW, 60, { r: 9 });
  K.mono(svg, PX + PW / 2, BY - 3, 'batch-size predictor', { size: 13, anchor: 'middle', weight: 600, color: C.ink });
  K.label(svg, PX + PW / 2, BY + 17, 'four inputs', { anchor: 'middle' });
  K.panel(svg, PX, LY - 30, PW, 60, { r: 9 });
  K.mono(svg, PX + PW / 2, LY - 3, 'learning-rate predictor', { size: 13, anchor: 'middle', weight: 600, color: C.ink });
  K.label(svg, PX + PW / 2, LY + 17, 'log-linear', { anchor: 'middle' });
  K.arrow(svg, PX + PW, BY, PX + PW + 28, BY, { color: C.ink3 });
  K.mono(svg, PX + PW + 36, BY + 5, 'batch size', { size: 13.5, color: C.ink });
  K.arrow(svg, PX + PW, LY, PX + PW + 28, LY, { color: C.ink3 });
  K.mono(svg, PX + PW + 36, LY + 5, 'peak learning rate', { size: 13.5, color: C.ink });

  var note = CH.body(svg);
  var EXPLAIN = [
    'Width feeds both predictors. The batch size depends mainly on other inputs, so its ' +
      'wire to the batch predictor is faint.',
    'Depth is the same story as width: an input to both, a secondary one for the batch size.',
    'The optimal batch size depends primarily on the horizon: total training tokens, counting ' +
      'repeats. It feeds the learning-rate predictor too.',
    'Expansion rate, total experts over active experts, is the other input the batch size ' +
      'depends on primarily.',
    'Batch size is not set by hand. It is predicted first, and the learning-rate prediction is ' +
      'conditioned on the predicted value.'
  ];
  function wire(x1, y1, x2, y2, how) {
    var on = how > 0;
    K.arrow(wires, x1, y1, x2, y2, { curve: true, color: on ? C.amber : CH.GREY2,
      w: how === 2 ? 3.2 : how === 1 ? 1.8 : 1.1, dash: how === 1.5 ? '4 4' : null, head: on });
  }
  function draw(sel) {
    CH.clear(wires);
    INS.forEach(function (name, i) {
      var y = 25 + i * 46 + 16, me = i === sel;
      if (i < 4) {
        var strong = i >= 2;
        wire(IX, y, PX, BY, me ? (strong ? 2 : 1.5) : 0);
        wire(IX, y, PX, LY, me ? 1 : 0);
      } else wire(IX, y, PX, LY, me ? 1 : 0);
    });
    // the predicted batch is what the learning-rate predictor conditions on
    var me = sel === 4;
    K.arrow(wires, PX + PW / 2, BY + 30, PX + PW / 2, LY - 30,
      { color: me ? C.amber : C.ink3, w: me ? 3.2 : 1.2 });
    CH.clear(note);
    K.callout(note, 0, 268, 640, EXPLAIN[sel], { cols: 74 });
  }
  var pk = CH.pick(svg, 0, 25, INS, draw, { w: IX, h: 32, gap: 14 });

  // the scaling ladder, schematic
  var LYB = 470, sizes = [22, 34, 50, 74], x = 0;
  K.label(svg, 0, 366, 'scaling ladder (schematic)');
  sizes.forEach(function (s, i) {
    var last = i === sizes.length - 1;
    svg.appendChild(K.n('rect', { x: x, y: LYB - s, width: s, height: s, rx: 4,
      fill: last ? K.TINT.amber : '#fff', stroke: last ? C.amber : CH.GREY, 'stroke-width': last ? 2 : 1.2 }));
    if (last) K.mono(svg, x + s / 2, LYB + 22, 'ML4', { size: 13, anchor: 'middle', weight: 600, color: C.amber });
    x += s + 20;
  });
  K.mono(svg, 0, LYB + 22, 'smaller proxies', { size: 13, color: C.ink3 });
  K.para(svg, 290, 410, 'Every rung uses the same predictors and multipliers: tokens per ' +
    'parameter, learning rate, batch size. The ladder differs from the real model "in scale only".',
    40, { size: 13.5, lh: 20 });

  K.foot(svg, 'Which input matters to which predictor follows §2.1 of the paper. The ladder\'s ' +
    'rung sizes are not given, so the squares only show the idea.');
  pk.pick(2);
};
