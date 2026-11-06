/* S_MIX: Figure 1, the two pretraining phases and how the mixture moves. */
window.SCENES.S_MIX = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var CATS = [['Web', 'web'], ['Docs', 'docs'], ['Code', 'code'], ['Misc', 'misc']];
  var P = [F['mix.p1_share'], F['mix.p2_share']];
  var svg = K.board(host, { alt: 'Pretraining data mixture in phase 1 and phase 2 (Figure 1)' });
  K.head(svg, 'Two phases of data', 'Figure 1 · phase 1 is ' + P[0] + '% of training, phase 2 is ' + P[1] + '%');
  var body = CH.body(svg);

  function draw(ph) {
    CH.clear(body);
    var other = 1 - ph;
    // the training timeline, split by phase
    K.label(body, 0, 60, 'training tokens');
    var W = 640, w1 = W * P[0] / 100;
    [[0, w1 - 1.5, 0], [w1 + 1.5, W - w1 - 1.5, 1]].forEach(function (s) {
      var on = s[2] === ph;
      body.appendChild(K.n('rect', { x: s[0], y: 70, width: s[1], height: 34, rx: 6,
        fill: on ? C.amber : K.TINT.ink, stroke: on ? 'none' : C.line }));
      K.mono(body, s[0] + 12, 92, 'phase ' + (s[2] + 1) + ' · ' + P[s[2]] + '% · ' +
        F[s[2] ? 'mix.seq_to_k' : 'mix.seq_from_k'] + 'K context',
        { size: 13, color: on ? '#fff' : C.ink3, weight: on ? 600 : 400 });
    });

    // one row per category: this phase as a bar, the other phase as a tick
    var X0 = 72, BW = 420, sx = function (v) { return X0 + v / 60 * BW; };
    K.label(body, X0, 150, 'share of the mixture, phase ' + (ph + 1));
    K.label(body, 640, 150, 'against phase ' + (other + 1), { anchor: 'end' });
    CATS.forEach(function (c, i) {
      var y = 168 + i * 66;
      var v = F['mix.p' + (ph + 1) + '_' + c[1]], o = F['mix.p' + (other + 1) + '_' + c[1]];
      var d = v - o, big = c[1] === 'code' || c[1] === 'web';
      K.mono(body, 0, y + 15, c[0], { size: 14, weight: 600, color: C.ink });
      body.appendChild(K.n('rect', { x: X0, y: y, width: BW, height: 20, rx: 4, fill: K.TINT.ink }));
      body.appendChild(K.n('rect', { x: X0, y: y, width: sx(v) - X0, height: 20, rx: 4,
        fill: big ? C.amber : CH.GREY }));
      body.appendChild(K.n('path', { d: 'M' + sx(o) + ' ' + (y - 5) + ' V' + (y + 25),
        stroke: C.ink2, 'stroke-width': 2, 'stroke-dasharray': '3 2' }));
      K.mono(body, sx(Math.max(v, o)) + 10, y + 15, CH.fmt(v, 1) + '%',
        { size: 13.5, weight: 600, color: C.ink });
      K.mono(body, sx(o), y + 42, 'phase ' + (other + 1) + ': ' + CH.fmt(o, 1) + '%',
        { size: 12.5, color: C.ink3, anchor: o < 12 ? 'start' : 'middle' });
      var dd = Math.abs(d) < 0.05 ? 'the same' : CH.fmt(Math.abs(d), 1) + ' pts ' + (d > 0 ? 'higher' : 'lower');
      K.mono(body, 640, y + 15, dd, { size: 13, color: big ? C.ink : C.ink3, anchor: 'end',
        weight: big ? 600 : 400 });
    });
    K.callout(body, 0, 444, 640, ph === 1 ?
      'Phase 2 shifts toward higher-quality and STEM-heavy data. Code more than doubles and ' +
      'web pages fall by ' + CH.fmt(F['mix.p1_web'] - F['mix.p2_web'], 1) + ' points.' :
      'Phase 1 is about broad coverage of domains and languages, so web pages are more ' +
      'than half of the mixture.', { cols: 74 });
  }

  K.switcher(svg, 0, 0, ['phase 1', 'phase 2'], draw, { tint: 'amber' });
  K.foot(svg, 'Dashed ticks mark the other phase. Amber marks the two biggest movers. The ' +
    'categories are the paper\'s own; it does not break them down further.');
};
