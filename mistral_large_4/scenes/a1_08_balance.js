/* S_BALANCE: global-batch balancing against per-EP-group balancing. Illustrative
   numbers: the paper reports no per-device loads, only that the per-group loss
   targets expert inbound tokens and was favored in pretraining comparisons. */
window.SCENES.S_BALANCE = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH;
  var svg = K.board(host, { alt: 'Illustration: tokens arriving at each device in two expert-parallel groups' });
  K.head(svg, 'Balance each group', 'load-balancing loss per expert-parallel group · illustrative numbers');
  var body = CH.body(svg);
  // tokens arriving at devices 1 to 4 of each group; the column totals match in both modes
  var LOADS = [
    [[190, 70, 80, 60], [10, 130, 120, 140]],
    [[106, 96, 100, 98], [94, 104, 100, 102]]
  ];
  var MEAN = 100;

  function draw(m) {
    CH.clear(body);
    var L = LOADS[m], all = L[0].concat(L[1]), worst = Math.max.apply(null, all);
    var BASE = 330, SC = 1.1, BW = 52, GAP = 18;
    var sy = function (v) { return BASE - v * SC; };
    [0, 1].forEach(function (gi) {
      var x0 = gi * 340 + 10;
      body.appendChild(K.n('rect', { x: x0 - 10, y: 70, width: 4 * BW + 3 * GAP + 20, height: 290, rx: 10,
        fill: 'none', stroke: C.line, 'stroke-dasharray': '4 4' }));
      K.label(body, x0, 92, 'EP group ' + (gi + 1));
      L[gi].forEach(function (v, d) {
        var x = x0 + d * (BW + GAP), hot = v === worst;
        body.appendChild(K.n('rect', { x: x, y: sy(v), width: BW, height: v * SC, rx: 4,
          fill: hot ? (m === 0 ? C.red : C.amber) : CH.GREY }));
        K.mono(body, x + BW / 2, BASE + 20, 'dev ' + (d + 1), { size: 12.5, anchor: 'middle', color: C.ink3 });
      });
      body.appendChild(K.n('path', { d: 'M' + (x0 - 4) + ' ' + sy(MEAN) + ' H' + (x0 + 4 * BW + 3 * GAP + 4),
        stroke: C.ink2, 'stroke-width': 1.3, 'stroke-dasharray': '5 4' }));
    });
    K.mono(body, 311, sy(MEAN) + 4, 'mean', { size: 12.5, anchor: 'middle', color: C.ink2 });
    // totals per device position, summed over both groups
    var tot = L[0].map(function (v, d) { return v + L[1][d]; });
    K.label(body, 0, 392, 'each expert set (same dev number), both groups summed');
    K.mono(body, 0, 416, tot.join(' · ') + ', the same in both modes', { size: 13.5, color: C.ink });
    var ratio = worst / MEAN;
    K.big(body, 0, 478, CH.fmt(ratio, 2) + '×', { size: 44, color: m === 0 ? C.red : C.amber });
    K.mono(body, 150, 458, 'busiest device against the mean', { size: 13.5, color: C.ink, weight: 600 });
    K.mono(body, 150, 478, m === 0 ? 'the whole step waits for it' : 'every device finishes at about the same time',
      { size: 13.5, color: C.ink2 });
  }
  K.switcher(svg, 0, 0, ['balance the global batch', 'balance each EP group (ML4)'], draw, { tint: 'amber' });
  K.foot(svg, 'Illustrative numbers. The paper reports no per-device loads; it says the per-group ' +
    'loss targets the tokens arriving at each expert, the quantity behind the slowdowns, and that ' +
    'pretraining comparisons favored it.');
};
