/* S_SPEC: speculative decoding, stepped through on one illustrative draft. The
   sentence, the six-token draft and which tokens survive are a schematic; the
   numbers are §6's: DSpark trained on 70B tokens, +1.2% and +0.5% acceptance
   length from its two additions, and +90% (no speculation) and +15% (EAGLE-3)
   in latency and throughput at eight concurrent requests. */
window.SCENES.S_SPEC = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts, n = K.n;
  var svg = K.board(host, { alt: 'Speculative decoding: a speculator drafts six tokens and the full model verifies them in one pass' });
  K.head(svg, 'Draft ahead, verify in blocks', 'speculative decoding · schematic · DSpark trained on ' +
    F['spec.train_b'] + 'B tokens');
  var body = CH.body(svg);

  var CTX = 'Speculative decoding lets a small model';
  var DRAFT = ['guess', 'the', 'next', 'words', 'for', 'it'];
  var OWN = 'tokens', NACC = 3;          // three accepted, the fourth replaced by the full model's token
  var SW = 96, SG = (640 - 6 * SW) / 5;
  function sx(i) { return i * (SW + SG); }
  function mid(i) { return sx(i) + SW / 2; }

  var STEP = [
    'A small speculator drafts tokens ahead of the full model, six in this made-up example; ' +
      'ML4 uses DSpark. Nothing drafted is final until the full model has checked it.',
    'In the usual scheme the full model checks the draft in one pass and keeps a prefix, three ' +
      'tokens here. ML4 adds a recurrent conditioning head (+' + F['spec.head'] + '% acceptance ' +
      'length) and unique mask tokens (+' + F['spec.mask'] + '%) to DSpark.',
    'Accepted tokens are kept, the same pass gives one more token from the full model, and the ' +
      'rest is thrown away: four tokens for one pass here, against one for plain decoding.'
  ];

  function check(p, cx, cy) {
    p.appendChild(n('circle', { cx: cx, cy: cy, r: 13, fill: K.TINT.amber, stroke: C.amber, 'stroke-width': 1.6 }));
    p.appendChild(n('path', { d: 'M' + (cx - 6) + ' ' + cy + ' L' + (cx - 1.5) + ' ' + (cy + 4.5) +
      ' L' + (cx + 6.5) + ' ' + (cy - 5), fill: 'none', stroke: C.amber, 'stroke-width': 2.6,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
  }
  function cross(p, cx, cy) {
    p.appendChild(n('path', { d: 'M' + (cx - 4.5) + ' ' + (cy - 4.5) + ' L' + (cx + 4.5) + ' ' + (cy + 4.5) +
      ' M' + (cx + 4.5) + ' ' + (cy - 4.5) + ' L' + (cx - 4.5) + ' ' + (cy + 4.5),
      stroke: C.red, 'stroke-width': 2.4, 'stroke-linecap': 'round' }));
  }

  function draw(s) {
    CH.clear(body);
    // the text so far; on the keep step the kept words join it
    K.label(body, 0, 52, 'the text so far');
    K.panel(body, 0, 62, 640, 38, { r: 8, flat: true });
    var t = n('text', { x: 14, y: 86, 'font-family': "'JetBrains Mono',monospace", 'font-size': 14 });
    var t1 = n('tspan', { fill: C.ink2 });
    t1.textContent = CTX;
    t.appendChild(t1);
    var ul = n('rect', { x: 0, y: 92, width: 0, height: 3, rx: 1.5, fill: C.amber });
    if (s === 2) {
      var t2 = n('tspan', { fill: C.ink, 'font-weight': 700 });
      t2.textContent = ' ' + DRAFT.slice(0, NACC).join(' ') + ' ' + OWN;
      t.appendChild(t2);
      body.appendChild(ul);
      // underline the kept words once the text is laid out
      requestAnimationFrame(function () {
        try {
          var a = t.getStartPositionOfChar(CTX.length + 1).x;
          var e = t.getEndPositionOfChar(t.getNumberOfChars() - 1).x;
          ul.setAttribute('x', a);
          ul.setAttribute('width', Math.max(0, e - a));
        } catch (er) {}
      });
    }
    body.appendChild(t);

    // the draft
    K.label(body, 0, 134, 'DSpark drafts six tokens ahead');
    DRAFT.forEach(function (w, i) {
      var acc = i < NACC, rej = i === NACC, gone = i > NACC;
      var fill = '#fff', stroke = CH.GREY, dash = '4 3', col = C.ink2, op = 1, wt = 400;
      if (s >= 1) { dash = null; stroke = acc ? C.amber : rej ? C.red : CH.GREY2; }
      if (s >= 1 && gone) { dash = '4 3'; col = C.ink3; }
      if (s === 2) {
        if (acc) { fill = C.amber; col = '#fff'; wt = 700; }
        if (rej || gone) { op = 0.45; col = C.ink3; }
      }
      var g = n('g', { opacity: op });
      g.appendChild(n('rect', { x: sx(i), y: 144, width: SW, height: 40, rx: 8, fill: fill,
        stroke: stroke, 'stroke-width': s >= 1 && !gone ? 1.8 : 1.3, 'stroke-dasharray': dash }));
      K.mono(g, mid(i), 169, w, { size: 14, anchor: 'middle', color: col, weight: wt });
      if (s === 2 && !acc) {
        g.appendChild(n('path', { d: 'M' + (mid(i) - w.length * 4.6 - 4) + ' 164.5 H' +
          (mid(i) + w.length * 4.6 + 4), stroke: C.ink3, 'stroke-width': 1.4 }));
      }
      body.appendChild(g);
    });

    // the verify bar shows on every step (neutral until it runs); the verdict marks
    // are hidden until the verify step, but stay in place so the layout never jumps
    var on = s >= 1;
    body.appendChild(n('rect', { x: 0, y: 222, width: 640, height: 36, rx: 8, fill: on ? K.TINT.amber : '#fff',
      stroke: on ? C.amber : CH.GREY2, 'stroke-opacity': on ? 0.5 : 1, 'stroke-dasharray': on ? null : '4 3' }));
    K.mono(body, 320, 245, (on ? 'The full model checks' : 'Next, the full model checks') + ' all six positions in one pass',
      { size: 13.5, anchor: 'middle', color: on ? C.ink : C.ink3, weight: on ? 600 : 400 });
    // before verification: one empty slot per drafted token, nothing decided yet
    if (!on) DRAFT.forEach(function (w, i) {
      body.appendChild(n('circle', { cx: mid(i), cy: 290, r: 13, fill: 'none', stroke: CH.GREY2,
        'stroke-width': 1.4, 'stroke-dasharray': '3 3' }));
    });
    var v = n('g', { opacity: on ? 1 : 0 });
    body.appendChild(v);
    cross(v, mid(NACC) - 38, 204);
    K.mono(v, mid(NACC) - 26, 208.5, 'rejected', { size: 12.5, color: C.ink2 });
    for (var i = 0; i < NACC; i++) check(v, mid(i), 290);
    v.appendChild(n('rect', { x: sx(NACC), y: 272, width: SW, height: 38, rx: 8, fill: C.amber }));
    K.mono(v, mid(NACC), 296, OWN, { size: 14, anchor: 'middle', color: '#fff', weight: 700 });
    v.appendChild(n('rect', { x: sx(NACC + 1), y: 272, width: 640 - sx(NACC + 1), height: 38, rx: 8,
      fill: 'none', stroke: CH.GREY2, 'stroke-dasharray': '4 3' }));
    K.mono(v, (sx(NACC + 1) + 640) / 2, 296, 'thrown away', { size: 13, anchor: 'middle', color: C.ink3 });
    K.mono(v, mid(1), 334, 'accepted, ' + NACC + ' of ' + DRAFT.length, { size: 13, anchor: 'middle', color: C.ink2 });
    K.mono(v, mid(NACC), 334, 'the full model\'s token', { size: 13, anchor: 'middle', color: C.ink2 });

    K.callout(body, 0, 352, 640, STEP[s], { cols: 74 });
  }

  // the measured gains, under every step
  K.label(svg, 0, 462, 'latency and throughput gain · ' + F['spec.conc'] + ' concurrent requests');
  K.big(svg, 0, 506, '+' + F['spec.gain'] + '%', { size: 40, color: C.ink });
  K.mono(svg, 0, 530, 'over no speculation, at 50K input', { size: 13, color: C.ink2 });
  K.big(svg, 330, 506, '+' + F['spec.vs_eagle'] + '%', { size: 40, color: C.ink });
  K.mono(svg, 330, 530, 'over an EAGLE-3 speculator', { size: 13, color: C.ink2 });

  K.switcher(svg, 0, 0, ['1 draft', '2 verify', '3 keep'], draw, { tint: 'amber' });
  K.foot(svg, 'Schematic of speculative decoding in general, not ML4\'s exact procedure: the paper ' +
    'gives no draft length, acceptance pattern or verification details, so the sentence and which ' +
    'tokens survive are made up. Block verification, in the work it cites, checks the draft as a ' +
    'whole rather than token by token.');
};
