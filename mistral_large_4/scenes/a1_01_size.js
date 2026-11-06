/* S_SIZE: a thousand squares, one per billion parameters. Each token lights a
   different 52. Schematic: the paper gives the 52B total, not which weights
   every token shares, so which squares light up is illustrative. */
window.SCENES.S_SIZE = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var act = F['size.active_b'], tot = Math.round(F['size.total_t'] * 1000);
  var svg = K.board(host, { alt: 'A grid of ' + tot + ' squares, one per billion parameters, ' +
    act + ' of them lit for the current token' });
  K.head(svg, act + 'B of ' + F['size.total_t'] + 'T, on every token',
    'mixture of experts · each square is one billion parameters');

  K.big(svg, 0, 46, act + 'B', { size: 54, color: C.amber });
  K.label(svg, 132, 20, 'active on each token');
  K.mono(svg, 132, 42, 'out of ' + F['size.total_t'] + 'T parameters, ' +
    CH.fmt(act / tot * 100, 1) + '% of the model', { size: 13.5 });

  // legend
  svg.appendChild(K.n('rect', { x: 0, y: 64, width: 12, height: 12, rx: 2.5, fill: C.amber }));
  K.mono(svg, 20, 75, 'working on this token', { size: 12.5 });
  svg.appendChild(K.n('rect', { x: 210, y: 64, width: 12, height: 12, rx: 2.5, fill: CH.GREY2 }));
  K.mono(svg, 230, 75, 'idle for this token', { size: 12.5 });

  var COLS = 40, PITCH = 15, CELL = 12, Y0 = 92, cells = [];
  var grid = K.n('g', {});
  for (var i = 0; i < tot; i++) {
    var r = K.n('rect', { x: (i % COLS) * PITCH, y: Y0 + Math.floor(i / COLS) * PITCH,
      width: CELL, height: CELL, rx: 2.5, fill: CH.GREY2, opacity: 0.6 });
    grid.appendChild(r);
    cells.push(r);
  }
  svg.appendChild(grid);
  var gridBottom = Y0 + Math.ceil(tot / COLS) * PITCH;

  var rand = CH.rng(52), token = 0, lit = [], last = 0;
  var out = CH.readout(svg, 196, gridBottom + 36, { size: 13.5 });
  function route() {
    var prev = {}, next = {}, keep = 0;
    lit.forEach(function (j) { prev[j] = 1; });
    var picked = [];
    while (picked.length < act) {
      var j = Math.floor(rand() * tot);
      if (!next[j]) { next[j] = 1; picked.push(j); if (prev[j]) keep++; }
    }
    lit.forEach(function (j) { cells[j].setAttribute('fill', CH.GREY2); cells[j].setAttribute('opacity', 0.6); });
    picked.forEach(function (j) { cells[j].setAttribute('fill', C.amber); cells[j].setAttribute('opacity', 1); });
    lit = picked;
    token++;
    out.set(token === 1 ? 'token 1: ' + act + ' squares lit' :
      'token ' + token + ': ' + act + ' lit, ' + keep + ' of them also lit for token ' + (token - 1));
  }
  CH.button(svg, 0, gridBottom + 15, 'send a token', function () { route(); last = -1; }, { w: 176 });
  route();

  K.foot(svg, 'Schematic. In a real mixture of experts some weights, such as attention, serve ' +
    'every token, and routing happens separately in every layer. The paper gives the 52B ' +
    'total, not that split, so which squares light up here is illustrative.');

  return {
    tick: function (t) {
      if (last < 0) last = t;
      if (t - last > 2.2) { last = t; route(); }
    },
    start: function () { last = -1; },
    stop: function () {}
  };
};
