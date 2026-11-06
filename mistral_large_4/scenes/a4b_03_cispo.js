/* S_CISPO: the weight a token's gradient gets as the probability ratio (trainer
   over generator) moves, for the paper's CISPO-style loss and for PPO (§4.4.1).
   The cap of 3, the missing floor and the stop-gradient are the paper's. PPO is
   drawn with the common clip range of 0.2, which this paper does not give. */
window.SCENES.S_CISPO = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var CAP = F['rl.clip'], EPS = 0.2;   // EPS: illustrative PPO clip range, not from the paper
  var XM = 5, YM = 5.2;
  var svg = K.board(host, { alt: 'Gradient weight against probability ratio for CISPO, capped at ' + CAP + ', and PPO' });
  K.head(svg, 'Clip the weight, not the gradient', '§4.4.1 · CISPO-style REINFORCE · weight capped at ' +
    CAP + ' · no floor');

  var pos = true, slider = null;
  var cispo = function (x) { return Math.min(x, CAP); };
  // gradient of PPO's clipped surrogate, as a multiple of A times grad log p
  var ppo = function (x) { return pos ? (x <= 1 + EPS ? x : 0) : (x >= 1 - EPS ? x : 0); };
  var f2 = function (v) { return CH.fmt(v, 2); };

  var X0 = 50, PW = 570, Y0 = 74, PH = 186;
  // the capped region, behind everything
  var sxa = CH.scale(0, XM, X0, X0 + PW);
  svg.appendChild(K.n('rect', { x: sxa(CAP), y: Y0, width: X0 + PW - sxa(CAP), height: PH,
    fill: K.TINT.amber }));
  var ax = CH.axes(svg, X0, Y0, PW, PH, { x0: 0, x1: XM, y0: 0, y1: YM, xt: [0, 1, 2, 3, 4, 5], yt: [0, 1, 3, 5],
    xl: 'probability ratio, trainer over generator', yl: 'weight on the gradient' });
  CH.key(svg, 330, 60, 'CISPO, as in ML4', { color: C.amber });
  CH.key(svg, 500, 60, 'PPO, clip ' + EPS, { color: C.blue });

  // lines, then the live cursor and dots, then labels on top of both
  var curves = CH.body(svg), live = CH.body(svg), labels = CH.body(svg), note = CH.body(svg);
  // a label on a backing in the colour behind it, so the cursor passes under it cleanly.
  // Width is estimated from the character count (mono, about 0.62 em a character).
  function tag(xv, yv, s, o, tinted) {
    var x = ax.sx(xv), y = ax.sy(yv), w = s.length * 12.5 * 0.62 + 8;
    var x0 = o.anchor === 'end' ? x - w + 4 : x - 4;
    labels.appendChild(K.n('rect', { x: x0, y: y - 13, width: w, height: 18, rx: 4, fill: C.bg }));
    if (tinted) labels.appendChild(K.n('rect', { x: x0, y: y - 13, width: w, height: 18, rx: 4,
      fill: K.TINT.amber }));
    K.mono(labels, x, y, s, o);
  }

  function drawCurves() {
    CH.clear(curves); CH.clear(labels);
    var e = pos ? 1 + EPS : 1 - EPS;
    // where the two curves coincide, PPO is drawn wider underneath, so it shows as a rim
    if (pos) {
      CH.line(curves, ax, [[0, 0], [e, e]], { color: C.blue, w: 7.5 });
      CH.line(curves, ax, [[e, e], [e, 0]], { color: C.blue, w: 1.6, dash: '3 3' });
      CH.line(curves, ax, [[e, 0], [XM, 0]], { color: C.blue, w: 3.4 });
    } else {
      CH.line(curves, ax, [[0, 0], [e, 0]], { color: C.blue, w: 3.4 });
      CH.line(curves, ax, [[e, 0], [e, e]], { color: C.blue, w: 1.6, dash: '3 3' });
      CH.line(curves, ax, [[e, e], [CAP, CAP]], { color: C.blue, w: 7.5 });
      CH.line(curves, ax, [[CAP, CAP], [XM, XM]], { color: C.blue, w: 3 });
    }
    CH.line(curves, ax, [[0, 0], [CAP, CAP], [XM, CAP]], { color: C.amber, w: 3 });
    // under the cap line, inside the capped region
    tag(CAP + 0.1, 2.3, 'CISPO: capped at ' + CAP, { size: 12.5, color: C.ink, weight: 600 }, true);
    // PPO: by its zero segment (left of the capped region), or in the wedge under its uncapped line
    if (pos) tag(1.27, 0.42, 'PPO: no gradient past ' + e, { size: 12.5, color: C.ink2 }, false);
    else tag(XM - 0.06, 3.45, 'PPO: no cap', { size: 12.5, anchor: 'end', color: C.ink2 }, true);
  }

  function update(r) {
    CH.clear(live);
    live.appendChild(K.n('path', { d: 'M' + ax.sx(r) + ' ' + (Y0 + PH) + ' V' + Y0, stroke: C.ink,
      'stroke-width': 1.1, 'stroke-dasharray': '3 3' }));
    CH.dots(live, ax, [[r, ppo(r)]], { color: C.blue, r: 9.5 });
    CH.dots(live, ax, [[r, cispo(r)]], { color: C.amber, r: 6.5 });
    var c = cispo(r), p = ppo(r), s;
    if (pos) {
      s = r <= 1 + EPS ? 'both give the token a weight of ' + f2(r) + '. Close to 1, CISPO and PPO agree.' :
        r <= CAP ? 'CISPO gives the token a weight of ' + f2(c) + '. PPO clips it inside the objective, so its ' +
          'gradient is zero and the token stops learning.' :
        'CISPO caps the weight at ' + CAP + ', and the token still learns. PPO clips it inside the ' +
          'objective, so its gradient is zero.';
    } else {
      s = r < 1 - EPS ? 'CISPO has no floor, so the token keeps a weight of ' + f2(c) + '. PPO clips it, and ' +
          'its gradient is zero.' :
        r <= CAP ? 'both give the token a weight of ' + f2(r) + '. On this side PPO only clips below ' +
          (1 - EPS) + '.' :
        'CISPO caps the weight at ' + CAP + ', which bounds the variance from stale or mismatched ' +
          'tokens. PPO lets it grow to ' + f2(p) + '.';
    }
    CH.clear(note);
    K.callout(note, 0, 354, 640, 'Ratio ' + f2(r) + ': ' + s, { cols: 74 });
  }

  K.switcher(svg, 0, 0, ['advantage above zero', 'advantage below zero'], function (i) {
    pos = i === 0;
    drawCurves();
    if (slider) update(slider.get());
  }, { tint: 'amber' });
  slider = CH.slider(svg, X0, 330, PW, { min: 0, max: XM, step: 0.05, value: 4.5, label: 'probability ratio',
    fmt: function (v) { return 'ratio ' + f2(v); }, onChange: update });

  K.code(svg, 0, 436, 640, [
    { t: 'loss on a token: -w × A × log p(token)', c: '#e8c9a8' },
    'where w = min(p_trainer / p_generator, ' + CAP + '), held constant',
    'group scale: rollouts in group / trainable tokens in group',
    'optimizer: Muon, constant lr ' + F['muon.lr_m'] + 'e-' + F['muon.lr_e'] + ' after ' +
      F['muon.warmup'] + ' warmup steps'
  ]);

  K.foot(svg, 'PPO is drawn with the common clip range of ' + EPS + ', which this paper does not give. ' +
    '"Held constant" is the stop-gradient: w scales the gradient, but no gradient flows through w.');
};
