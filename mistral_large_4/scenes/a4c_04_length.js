/* S_LENGTH: the group-relative length bonus of §4.5.1 (Equations 2 and 3), live.
   A schematic: the group of six and its lengths are illustrative. The rule is the
   paper's: bins of width len.bin x Lmin anchored at the shortest correct rollout,
   a linear bonus from lambda (shortest bin) to zero (longest bin), correct only. */
window.SCENES.S_LENGTH = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var BIN = F['len.bin'];   // 0.1: bin width as a fraction of Lmin
  var svg = K.board(host, { alt: 'A group of six rollouts with draggable lengths and their length bonuses' });
  K.head(svg, 'A bonus for being right and brief', 'group-relative length bonus · Equations 2 and 3 · schematic');
  var swg = CH.body(svg);   // the presets come first in reading and tab order

  var PRE = [
    { L: [1400, 1030, 1200, 1260, 2050, 1640], ok: [1, 0, 1, 1, 1, 1] },
    { L: [1250, 1800, 1210, 1300, 960, 1280], ok: [1, 0, 1, 1, 0, 1] },
    { L: [1400, 1030, 1200, 1260, 2050, 1640], ok: [0, 0, 0, 0, 0, 0] }
  ];
  var L = PRE[0].L.slice(), ok = PRE[0].ok.slice(), N = L.length;
  var LO = 800, HI = 2400, X0 = 110, X1 = 500, MINL = 900, MAXL = 2300;
  var sx = CH.scale(LO, HI, X0, X1), inv = CH.scale(X0, X1, LO, HI);
  var rowY = function (i) { return 92 + i * 40; };

  // Equations 2 and 3
  function bonus() {
    var C2 = [];
    for (var i = 0; i < N; i++) if (ok[i]) C2.push(i);
    var r = { b: L.map(function () { return 0; }), C: C2 };
    if (!C2.length) { r.state = 'none'; return r; }
    var Lmin = Math.min.apply(null, C2.map(function (i) { return L[i]; })), D = BIN * Lmin;
    var Lt = L.map(function (l) { return Lmin + D * Math.floor((l - Lmin) / D + 1e-9); });
    var Ltmax = Math.max.apply(null, C2.map(function (i) { return Lt[i]; }));
    r.Lmin = Lmin; r.D = D; r.Lt = Lt; r.Ltmax = Ltmax;
    if (Ltmax - Lmin < 1e-6) { r.state = 'onebin'; return r; }
    C2.forEach(function (i) { r.b[i] = (Ltmax - Lt[i]) / (Ltmax - Lmin); });
    r.state = 'ok';
    return r;
  }

  var back = CH.body(svg);     // bins and row tracks, redrawn
  var mid = CH.body(svg);      // quantization marks, redrawn
  var handles = CH.body(svg);  // the draggable dots, kept alive across redraws
  var front = CH.body(svg);    // labels, bonuses and the callout, redrawn
  K.label(svg, 516, 62, 'bonus');
  // the length axis
  svg.appendChild(K.n('path', { d: 'M' + X0 + ' 318 H' + X1, stroke: C.ink3 }));
  [1000, 1500, 2000].forEach(function (v) {
    svg.appendChild(K.n('path', { d: 'M' + sx(v) + ' 318 v5', stroke: C.ink3 }));
    K.mono(svg, sx(v), 338, CH.comma(v), { size: 12.5, anchor: 'middle', color: C.ink3 });
  });
  K.label(svg, X1, 360, 'rollout length, tokens', { anchor: 'end' });
  K.mono(svg, 0, 392, 'bonus = λ × (longest bin − its bin) / (longest bin − Lmin), correct only',
    { size: 13, color: C.ink2 });
  // reserve the tallest callout, so the foot (placed once) never lands on it
  svg.appendChild(K.n('rect', { x: 0, y: 410, width: 640, height: 3 * 19 + 22, fill: 'none' }));

  // right or wrong toggles, one per rollout
  var chips = [];
  for (var i = 0; i < N; i++) (function (i) {
    var g = K.n('g', { tabindex: 0, role: 'button', 'aria-label': 'rollout ' + (i + 1) + ': toggle right or wrong',
      style: 'cursor:pointer;outline:none' });
    var r = K.n('rect', { x: 0, y: rowY(i) - 13, width: 86, height: 26, rx: 13, fill: '#fff' });
    g.appendChild(r);
    var t = K.mono(g, 43, rowY(i) + 4.5, '', { size: 13, anchor: 'middle', color: C.ink2 });
    svg.appendChild(g);
    var flip = function () { ok[i] = ok[i] ? 0 : 1; update(); };
    g.addEventListener('click', flip);
    g.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); flip(); }
    });
    chips.push({ r: r, t: t });
  })(i);

  // draggable length handles
  var dots = [];
  for (var j = 0; j < N; j++) (function (i) {
    var g = K.n('g', { tabindex: 0, role: 'slider', 'aria-label': 'rollout ' + (i + 1) + ' length',
      'aria-valuemin': MINL, 'aria-valuemax': MAXL, style: 'cursor:ew-resize;outline:none;touch-action:none' });
    var hit = K.n('circle', { r: 14, cy: rowY(i), fill: 'transparent' });
    var d = K.n('circle', { r: 8, cy: rowY(i), 'stroke-width': 2.2 });
    g.appendChild(hit); g.appendChild(d);
    handles.appendChild(g);
    var set = function (v) { L[i] = Math.round(Math.max(MINL, Math.min(MAXL, v))); update(); };
    // pointer x in board units, read against the untransformed parent group: g itself
    // carries translate(sx(L[i])), which would offset its own local frame
    var px = function (e) { return CH.toLocal(handles, e).x; };
    var drag = false, grab = 0;
    g.addEventListener('pointerdown', function (e) {
      drag = true; try { g.setPointerCapture(e.pointerId); } catch (er) {}
      grab = px(e) - sx(L[i]);   // where on the dot it was grabbed, so it does not jump
      e.preventDefault();
    });
    g.addEventListener('pointermove', function (e) { if (drag) set(inv(px(e) - grab)); });
    g.addEventListener('pointerup', function () { drag = false; });
    g.addEventListener('pointercancel', function () { drag = false; });
    g.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { set(L[i] + 20); e.preventDefault(); }
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') { set(L[i] - 20); e.preventDefault(); }
    });
    dots.push({ g: g, hit: hit, d: d });
  })(j);

  // headers and a key for the two shaded bins
  K.label(svg, 0, 62, 'answer');
  // each key entry is shown only while its band is drawn (see update)
  var keyShort = K.n('g', {}), keyLong = K.n('g', {});
  svg.appendChild(keyShort); svg.appendChild(keyLong);
  keyShort.appendChild(K.n('rect', { x: X0, y: 51, width: 14, height: 14, rx: 3, fill: K.TINT.amber,
    stroke: C.amber }));
  K.mono(keyShort, X0 + 20, 63, 'shortest correct bin', { size: 12.5, color: C.ink2 });
  keyLong.appendChild(K.n('rect', { x: X0 + 202, y: 51, width: 14, height: 14, rx: 3, fill: 'rgba(31,37,48,.08)',
    stroke: CH.GREY }));
  K.mono(keyLong, X0 + 222, 63, 'longest correct bin', { size: 12.5, color: C.ink2 });

  // a label per row on the right; the bonus bar is drawn underneath
  var outs = [];
  for (var k = 0; k < N; k++) outs.push(K.mono(svg, 640, rowY(k) + 4.5, '', { size: 13, anchor: 'end',
    weight: 600, color: C.ink }));
  var note = CH.body(svg);

  function update() {
    var r = bonus();
    CH.clear(back); CH.clear(mid); CH.clear(note);
    keyShort.setAttribute('display', r.state !== 'none' ? 'inline' : 'none');
    keyLong.setAttribute('display', r.state === 'ok' ? 'inline' : 'none');
    var top = rowY(0) - 20, bot = rowY(N - 1) + 20;
    if (r.state !== 'none') {
      // bin boundaries from Lmin upward: each length is pushed down to the edge of its bin
      for (var e = r.Lmin; e <= HI; e += r.D) {
        back.appendChild(K.n('path', { d: 'M' + sx(e) + ' ' + top + ' V' + bot, stroke: C.ink3,
          'stroke-width': 0.8, 'stroke-dasharray': '2 3', opacity: 0.55 }));
      }
      if (r.state === 'ok') {
        back.appendChild(K.n('rect', { x: sx(r.Ltmax), y: top, width: sx(r.Ltmax + r.D) - sx(r.Ltmax),
          height: bot - top, fill: 'rgba(31,37,48,.08)' }));
      }
      back.appendChild(K.n('rect', { x: sx(r.Lmin), y: top, width: sx(r.Lmin + r.D) - sx(r.Lmin),
        height: bot - top, fill: K.TINT.amber, stroke: C.amber, 'stroke-width': 1 }));
    }
    for (var i = 0; i < N; i++) {
      var y = rowY(i), right = !!ok[i];
      back.appendChild(K.n('path', { d: 'M' + X0 + ' ' + y + ' H' + X1, stroke: C.line2, 'stroke-width': 6,
        'stroke-linecap': 'round' }));
      // quantization: a bracket from the true length back to its bin's lower edge
      if (right && r.Lt && r.Lt[i] < L[i] - 0.5) {
        mid.appendChild(K.n('path', { d: 'M' + sx(L[i]) + ' ' + y + ' H' + sx(r.Lt[i]) + ' m0 -7 v14',
          stroke: C.ink2, 'stroke-width': 1.6 }));
      }
      chips[i].r.setAttribute('stroke', right ? C.ink3 : C.red);
      chips[i].r.setAttribute('fill', right ? '#fff' : K.TINT.red);
      chips[i].t.textContent = right ? 'right' : 'wrong';
      dots[i].g.setAttribute('transform', 'translate(' + sx(L[i]) + ' 0)');
      dots[i].g.setAttribute('aria-valuenow', L[i]);
      dots[i].d.setAttribute('fill', right ? C.ink2 : '#fff');
      dots[i].d.setAttribute('stroke', right ? C.ink2 : CH.GREY);
      var b = r.b[i];
      if (b > 0) note.appendChild(K.n('rect', { x: 516, y: y - 9, width: 72 * b, height: 18, rx: 3,
        fill: C.amber, opacity: 0.85 }));
      outs[i].textContent = b > 0 ? CH.fmt(b, 2) + 'λ' : '0';
      outs[i].setAttribute('fill', b > 0 ? C.ink : C.ink3);
    }
    var msg;
    if (r.state === 'none') {
      msg = 'No rollout is correct, so there is no Lmin and every bonus is zero. Brevity only ' +
        'earns anything on top of a right answer.';
    } else if (r.state === 'onebin' && r.C.length === 1) {
      msg = 'Only one rollout is correct, so there is no other correct length to compare it ' +
        'with, and its bonus is zero.';
    } else if (r.state === 'onebin') {
      msg = 'Every correct rollout lands in the same ' + CH.comma(r.D) + '-token bin above Lmin = ' +
        CH.comma(r.Lmin) + ', so every bonus is zero. Small length gaps earn nothing.';
    } else {
      msg = 'The shortest correct rollout sets Lmin = ' + CH.comma(r.Lmin) + ' tokens, so bins are ' +
        CH.comma(r.D) + ' tokens wide. That bin earns the full λ, the longest correct bin earns 0, linear ' +
        'in between. Wrong answers earn nothing, however short.';
    }
    K.callout(note, 0, 410, 640, msg, { cols: 74 });
  }

  K.switcher(swg, 0, 0, ['a mixed group', 'all in one bin', 'none correct'], function (p) {
    L = PRE[p].L.slice(); ok = PRE[p].ok.slice(); update();
  }, { tint: 'amber' });
  K.foot(svg, 'Schematic: the group of six and its lengths are illustrative; the rule is Equations 2 ' +
    'and 3 of §4.5.1. λ is a coefficient varied across runs. Drag a dot or flip an answer.');
};
