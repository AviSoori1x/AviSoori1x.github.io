/* S_TERM: §4.1.2, terminal-agent tasks. Three seed sources (pick one to see
   what it contributes and why), all fed through the same authoring pipeline into
   a task with four parts, which is then checked three ways. A schematic: the
   paper shows no example seeds, and the only count it gives is "millions" of
   real-world seeds. Every label is the paper's own wording or close to it. */
window.SCENES.S_TERM = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH;
  var svg = K.board(host, { alt: 'Terminal tasks: real-world seeds, synthetic seeds and internal traces ' +
    'feed one authoring pipeline that writes a four-part task, which is then checked' });
  K.head(svg, 'Terminal tasks, seeded three ways', '§4.1.2 · schematic · three sources, one pipeline');
  var body = CH.body(svg);

  var BW = 200, GAP = 20, PY = 56, PH = 140;
  var SRC = [
    { name: 'real-world seeds', goal: 'goal: broad coverage of terminal use', from: 'a real-world seed' },
    { name: 'synthetic seeds', goal: 'goal: supplement the real-world seeds', from: 'a synthetic seed' },
    { name: 'internal traces', goal: 'goal: work with real codebases', from: 'an internal trace' }
  ];
  var DOMAINS = [['software engineering', 'security'], ['system administration', 'databases'],
    ['optimization']];
  var SLOTS = ['skill', 'domain', 'difficulty', 'verbosity', 'other attributes'];

  function realWorld(p) {
    K.big(p, 20, 140, 'millions', { size: 44, color: C.amber });
    K.mono(p, 20, 164, 'of technical seeds, mined', { size: 13, color: C.ink2 });
    K.mono(p, 20, 182, 'from the pretraining corpus', { size: 13, color: C.ink2 });
    DOMAINS.forEach(function (row, r) {
      var x = 330;
      row.forEach(function (d) {
        var c = K.chip(p, x, PY + 34 + r * 32, d, { color: C.ink2 });
        x += c._w + 8;
      });
      if (r === 2) K.mono(p, x + 4, PY + 34 + r * 32 + 17, 'and so on', { size: 12.5, color: C.ink3 });
    });
  }

  function synthetic(p) {
    var x = 40, y = 108, h = 36;
    // the tuple's brackets, drawn rather than typed
    p.appendChild(K.n('path', { d: 'M30 ' + (y - 4) + ' Q20 ' + (y + h / 2) + ' 30 ' + (y + h + 4),
      fill: 'none', stroke: C.ink2, 'stroke-width': 2, 'stroke-linecap': 'round' }));
    SLOTS.forEach(function (s, i) {
      var w = s.length * 8.06 + 24;
      p.appendChild(K.n('rect', { x: x, y: y, width: w, height: h, rx: 7, fill: K.TINT.amber,
        stroke: C.amber, 'stroke-width': 1.4, 'stroke-dasharray': '4 3' }));
      K.mono(p, x + w / 2, y + 23, s, { size: 13, color: C.ink, anchor: 'middle', weight: 600 });
      x += w;
      if (i < SLOTS.length - 1) {
        // a comma at the slot text's baseline, so the row reads as one tuple
        K.mono(p, x + 3, y + 25, ',', { size: 16, color: C.ink2, weight: 600 });
        x += 14;
      }
    });
    p.appendChild(K.n('path', { d: 'M' + (x + 10) + ' ' + (y - 4) + ' Q' + (x + 20) + ' ' + (y + h / 2) +
      ' ' + (x + 10) + ' ' + (y + h + 4), fill: 'none', stroke: C.ink2, 'stroke-width': 2,
      'stroke-linecap': 'round' }));
    K.mono(p, 20, 178, 'each seed is one tuple, drawn from a taxonomy such as Figure 10',
      { size: 13, color: C.ink2 });
  }

  function internal(p) {
    var y = 96, h = 84;
    p.appendChild(K.n('rect', { x: 20, y: y, width: 260, height: h, rx: 8, fill: C.ink }));
    K.mono(p, 36, y + 22, 'internal terminal-agent trace', { size: 12.5, color: '#fff', weight: 600 });
    // command lines, as prompt chevrons and bars: the trace content is not shown in the paper
    [150, 96, 182].forEach(function (w, i) {
      var ly = y + 42 + i * 14;
      p.appendChild(K.n('path', { d: 'M36 ' + (ly - 4) + ' l4 4 l-4 4', fill: 'none', stroke: C.amber,
        'stroke-width': 1.6, 'stroke-linecap': 'round' }));
      p.appendChild(K.n('rect', { x: 48, y: ly - 3, width: w, height: 6, rx: 3, fill: '#fff', opacity: 0.38 }));
    });
    // the trace is only seed content here; the shared pipeline below makes it verifiable
    K.arrow(p, 292, y + h / 2, 332, y + h / 2, { color: C.amber, w: 2.2 });
    K.panel(p, 344, y, 276, h, { r: 8 });
    p.appendChild(K.n('path', { d: 'M344 ' + (y + 8) + ' Q344 ' + y + ' 352 ' + y + ' H612 Q620 ' + y +
      ' 620 ' + (y + 8) + ' V' + (y + 4) + ' H344 Z', fill: C.amber }));
    K.mono(p, 360, y + 32, 'used as seed content', { size: 13.5, color: C.ink, weight: 600 });
    K.mono(p, 360, y + 54, 'the pipeline below turns it', { size: 12.5, color: C.ink3 });
    K.mono(p, 360, y + 72, 'into a verifiable task', { size: 12.5, color: C.ink3 });
  }

  var head = null;
  function draw(i) {
    CH.clear(body);
    // every source feeds the pipeline; the picked one is drawn as the live link
    SRC.forEach(function (s, k) {
      var cx = k * (BW + GAP) + BW / 2, on = k === i;
      body.appendChild(K.n('path', { d: 'M' + cx + ' 36 V' + PY, stroke: on ? C.amber : CH.GREY2,
        'stroke-width': on ? 2.4 : 1.3 }));
    });
    K.panel(body, 0, PY, 640, PH, { r: 9, stroke: C.amber });
    K.label(body, 20, PY + 26, SRC[i].goal, { color: C.ink2 });
    [realWorld, synthetic, internal][i](body);
    if (head) head.textContent = 'terminal task, from ' + SRC[i].from;
  }

  // one pipeline for all three sources
  K.arrow(svg, 320, PY + PH + 2, 320, 212, { color: C.ink3 });
  svg.appendChild(K.n('rect', { x: 0, y: 214, width: 640, height: 38, rx: 8, fill: C.ink }));
  K.mono(svg, 320, 238, 'agentic instance-authoring pipeline, as in Figure 9', { size: 13, color: '#fff',
    anchor: 'middle', weight: 600 });
  K.arrow(svg, 320, 254, 320, 274, { color: C.ink3 });

  // what comes out: a task in four parts, run inside a sandbox
  var WY = 276;
  K.panel(svg, 0, WY, 640, 120, { r: 9 });
  svg.appendChild(K.n('path', { d: 'M0 ' + (WY + 9) + ' Q0 ' + WY + ' 9 ' + WY + ' H631 Q640 ' + WY + ' 640 ' +
    (WY + 9) + ' V' + (WY + 30) + ' H0 Z', fill: K.TINT.ink }));
  [16, 32, 48].forEach(function (x) {
    svg.appendChild(K.n('circle', { cx: x, cy: WY + 15, r: 4.5, fill: CH.GREY2 }));
  });
  head = K.mono(svg, 66, WY + 20, '', { size: 13, color: C.ink2, weight: 600 });
  ['problem statement', 'test cases', 'environment and sandbox config', 'reference solution']
    .forEach(function (s, k) {
      var x = 16 + (k % 2) * 312, y = WY + 42 + Math.floor(k / 2) * 38;
      svg.appendChild(K.n('rect', { x: x, y: y, width: 296, height: 30, rx: 6, fill: '#fff', stroke: C.line }));
      svg.appendChild(K.n('rect', { x: x, y: y, width: 4, height: 30, rx: 2, fill: C.ink2 }));
      K.mono(svg, x + 16, y + 20, s, { size: 13, color: C.ink, weight: 600 });
    });

  // and the checks every task then faces
  K.label(svg, 0, 426, 'then every task is checked');
  ['instruction clarity audited; tests neither under- nor over-specified',
    'an adversarial agent probes for reward-hacking vulnerabilities',
    'solver pass rates calibrate the difficulty of each task'].forEach(function (s, k) {
    var y = 450 + k * 24;
    svg.appendChild(K.n('circle', { cx: 7, cy: y - 4.5, r: 7, fill: C.amber }));
    svg.appendChild(K.n('path', { d: 'M3.5 ' + (y - 4.5) + ' l2.5 2.8 l4.5 -5.2', fill: 'none', stroke: '#fff',
      'stroke-width': 1.7, 'stroke-linecap': 'round' }));
    K.mono(svg, 24, y, s, { size: 13, color: C.ink2 });
  });

  var pk = CH.pick(svg, 0, 0, SRC.map(function (s) { return s.name; }), draw,
    { w: BW, h: 36, gap: GAP, horizontal: true });
  K.foot(svg, 'A schematic of §4.1.2. The paper shows no example seeds, and the only count it gives is ' +
    'millions of real-world seeds.');
  pk.pick(0);
};
