/* S_ROLLSTATS: where the RL run's rollouts and output tokens went (§4.4.1).
   The paper names two categories, code agents and reasoning; "everything else"
   is 100% minus those two, which the paper does not break down. Length per
   rollout is derived: share of tokens over share of rollouts, times the run
   average (output tokens over rollouts). */
window.SCENES.S_ROLLSTATS = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var R = F['rl.rollouts_m'], TOK = F['rl.out_tokens_b'];
  var r1 = function (v) { return Math.round(v * 10) / 10; };
  var CAT = [
    { name: 'Code agents', roll: F['rl.code_roll'], tok: F['rl.code_tok'], col: C.amber },
    { name: 'Reasoning', roll: F['rl.reason_roll'], tok: F['rl.reason_tok'], col: C.blue },
    // the remainder: 100% minus the two categories the paper names
    { name: 'Everything else', roll: r1(100 - F['rl.code_roll'] - F['rl.reason_roll']),
      tok: r1(100 - F['rl.code_tok'] - F['rl.reason_tok']), col: CH.GREY2 }
  ];
  // average output tokens per rollout over the whole run, in thousands
  var AVG_K = TOK * 1e9 / (R * 1e6) / 1000;
  CAT.forEach(function (c) { c.len = c.tok / c.roll; });   // length relative to the average rollout

  var svg = K.board(host, { alt: 'Share of RL rollouts and of output tokens by task category' });
  K.head(svg, 'Where the tokens went', '§4.4.1 · one RL run · ' + R + 'M rollouts · ' + TOK +
    'B output tokens');

  // whole squares that still add to 100 (largest remainder)
  function squares(key) {
    var raw = CAT.map(function (c) { return c[key]; });
    var fl = raw.map(Math.floor), left = 100 - fl.reduce(function (a, b) { return a + b; }, 0);
    raw.map(function (v, i) { return [v - fl[i], i]; }).sort(function (a, b) { return b[0] - a[0]; })
      .slice(0, left).forEach(function (q) { fl[q[1]]++; });
    return fl;
  }

  // the four run totals; the one being counted is lit
  var TILE = [[R + 'M', 'rollouts'], [F['rl.groups_k'] + 'k', 'groups'],
    [F['rl.instances_k'] + 'k', 'instances'], [TOK + 'B', 'output tokens']];
  var tiles = TILE.map(function (t, i) {
    var x = i * 164, g = K.panel(svg, x, 46, 148, 76, { r: 9 });
    K.big(svg, x + 14, 46 + 40, t[0], { size: 28 });
    K.label(svg, x + 14, 46 + 62, t[1]);
    return g.firstChild;
  });

  // the waffle: one square is one percent of whatever is being counted
  var SQ = 20, GAP = 3, WY = 150, cellG = K.n('g', {}), cells = [];
  svg.appendChild(cellG);
  for (var i = 0; i < 100; i++) {
    var rc = K.n('rect', { x: (i % 10) * (SQ + GAP), y: WY + Math.floor(i / 10) * (SQ + GAP),
      width: SQ, height: SQ, rx: 3.5, fill: CH.GREY2 });
    cellG.appendChild(rc);
    cells.push(rc);
  }
  var cap = CH.readout(svg, 0, WY + 10 * (SQ + GAP) + 18, { size: 12.5, color: C.ink3 });
  cap.el.setAttribute('font-weight', 400);

  // one row per category, to the right of the waffle
  var RX = 262, rows = CAT.map(function (c, j) {
    var y = WY + j * 80;
    svg.appendChild(K.n('rect', { x: RX, y: y + 2, width: 14, height: 14, rx: 3, fill: c.col }));
    K.mono(svg, RX + 24, y + 14, c.name, { size: 14, weight: 600, color: C.ink });
    return {
      big: K.big(svg, RX, y + 56, '', { size: 32 }),
      d1: K.mono(svg, RX + 118, y + 38, '', { size: 13, color: C.ink2 }),
      d2: K.mono(svg, RX + 118, y + 57, '', { size: 13, color: C.ink3 })
    };
  });

  var note = CH.body(svg);
  var first = true;
  function draw(mode) {
    var key = mode ? 'tok' : 'roll', other = mode ? 'roll' : 'tok';
    tiles.forEach(function (bg, i) {
      var on = mode ? i === 3 : i === 0;
      bg.setAttribute('fill', on ? K.TINT.amber : C.panel);
      bg.setAttribute('stroke', on ? C.amber : C.line);
      bg.setAttribute('stroke-width', on ? 2 : 1);
    });
    var sq = squares(key), k = 0;
    CAT.forEach(function (c, j) {
      for (var s = 0; s < sq[j]; s++) cells[k++].setAttribute('fill', c.col);
      var r = rows[j];
      r.big.textContent = CH.fmt(c[key], c[key] % 1 ? 1 : 0) + '%';
      r.d1.textContent = CH.fmt(c[other], c[other] % 1 ? 1 : 0) + '% of ' + (mode ? 'rollouts' : 'output tokens');
      r.d2.textContent = CH.fmt(c.len, c.len < 1 ? 2 : 1) + '× as long as the average';
    });
    if (first) {
      // stagger only after the first paint, so the figure enters already filled
      first = false;
      requestAnimationFrame(function () {
        cells.forEach(function (rc, i) { rc.style.transition = 'fill .26s ease ' + (i * 1.6).toFixed(0) + 'ms'; });
      });
    }
    cap.set(mode ? 'One square is 1% of output tokens, about ' + CH.fmt(TOK / 100, 1) + 'B.' :
      'One square is 1% of rollouts, about ' + CH.fmt(R * 1000 / 100, 1) + 'k.');
    CH.clear(note);
    K.callout(note, 0, 420, 640, mode ?
      'Counted by tokens, code agents are ' + CAT[0].tok + '% of the run: about ' +
        Math.round(AVG_K * CAT[0].len) + 'k output tokens per rollout, against a run average of about ' +
        Math.round(AVG_K) + 'k.' :
      'Counted by rollouts, code agents are ' + CAT[0].roll + '% of the run and reasoning ' +
        CAT[1].roll + '%. Everything else, ' + CAT[2].roll + '%, the paper does not break down.', { cols: 74 });
  }
  K.switcher(svg, 0, 0, ['count rollouts', 'count tokens'], draw, { tint: 'amber' });

  // the per-rollout numbers in the tokens callout, derived from the run totals and shares
  K.foot(svg, 'Shares are the paper\'s; everything else is 100% minus the two named categories. ' +
    'Squares round to whole percents. Derived: length is share of tokens over share of rollouts; ' +
    TOK + 'B / ' + R + 'M is about ' + Math.round(AVG_K) + 'k tokens a rollout, and code agents (' +
    CH.fmt(CAT[0].len, 1) + '×) write about ' + Math.round(AVG_K * CAT[0].len) + 'k.');
};
