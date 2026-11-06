/* S_ASYNC: §4.3.1 and Figure 12, the asynchronous RL pipeline, redrawn as a
   schematic. Box and arrow labels follow the paper's figure; the peak counts
   come from window.ML4.facts (rlinf.*). Click any box to read what it does and
   what it hands to the next stage. */
window.SCENES.S_ASYNC = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var ROLL = F['rlinf.rollouts_k'], BOX = F['rlinf.sandboxes_k'], GPU = F['rlinf.gpus'],
    RATIO = F['rlinf.gen_ratio'], ALL = F['rlinf.all_gpus_k'];
  var svg = K.board(host, { alt: 'Figure 12 redrawn: the asynchronous RL pipeline, from dataloader ' +
    'to trainer, with generators, sandboxes and tool and verifier services serving the runners' });
  K.head(svg, 'A fully asynchronous RL pipeline', 'Figure 12, schematic · at peak: ' + ROLL +
    'k rollouts, about ' + ALL + 'k GPUs with services');
  var body = CH.body(svg);

  // geometry: the pipeline runs down the left column, the services sit on the right
  var SX = 44, SW = 262, BH = 44, RX = 360, RW = 252, BUS = 333;
  var SY = [24, 100, 176, 252, 328];
  // [name, figure sublabel, x, y, w, pool?, tag]
  var BOXES = [
    ['Dataloader', 'prompt sampling', SX, SY[0], SW, false],
    ['Runner pool', 'group of episodes', SX, SY[1], SW, true, ROLL + 'k at peak'],
    ['Groupwise scoring pool', 'group-level rewards', SX, SY[2], SW, true],
    ['Tokenization & packing pool', 'advantages, batching', SX, SY[3], SW, true],
    ['Trainer', 'policy update', SX, SY[4], SW, false],
    ['Generators', 'inference engines', RX, SY[1], RW, false],
    ['Sandboxes', 'isolated execution', RX, SY[2], RW, false, 'about ' + BOX + 'k'],
    ['Tool & verifier services', 'tool calls, rewards', RX, SY[3], RW, false]
  ];

  // what each box does and hands on, in the paper's words (§4.3.1); [handoff, detail]
  var SAY = [
    ['Dataloader: hands prompts to the runner pool',
      'Data loading is curriculum-driven, with feedback from the packing pool ' +
      '(Figure 12). For each prompt the pipeline samples a group of rollouts, so a ' +
      'prompt enters the runner pool as a group of episodes.'],
    ['Runner pool: hands each complete group to scoring',
      'Groups are admitted up to a maximum concurrency and as backpressure allows. An ' +
      'episode starts as soon as a slot is free, with priority for siblings of ' +
      'started groups. At peak, ' + ROLL + 'k rollouts ran at once.'],
    ['Scoring pool: hands scored episodes to packing',
      'Only complete groups arrive here. Groupwise verification applies pairwise ' +
      'rewards and groupwise penalties. Verifiers, LLM judges and pairwise reward models ' +
      'run as supporting services.'],
    ['Packing pool: batches to the trainer, feedback to the dataloader',
      'Trie-based tokenization handles possible re-tokenization events. A centralized ' +
      'bin-packer then packs individual trajectories into fixed-length microbatches, ' +
      'sent to each data-parallel rank.'],
    ['Trainer: hands new weights to the generators',
      'Each training step sends updated weights to the generators via NCCL. At peak: ' + CH.comma(GPU) + ' GPUs with a ' +
      '1:' + RATIO + ' trainer-to-generator ratio, about ' + ALL + 'k with judges and services.'],
    ['Generators: hand model outputs back to the runners',
      'vLLM engines on Ray. Concurrency adapts to backpressure from generators, ' +
      'services and trainer, limiting staleness. Heavy objects such as MoE routing ' +
      'decisions for router replay travel on the Ray data plane.'],
    ['Sandboxes: hand execution results back to the runners',
      'Isolated execution for the episodes. Like every supporting service, they run on ' +
      'Ray or optional multi-cluster infrastructure and autoscale with the load. At ' +
      'peak, about ' + BOX + 'k sandboxes served ' + ROLL + 'k concurrent rollouts.'],
    ['Tool & verifier services: hand tool results and rewards back',
      'Verifiers, tools, LLM judges and pairwise reward models serve the episodes and ' +
      'autoscale to the load imposed by in-progress rollouts. Including them, the ' +
      'pipeline used about ' + ALL + 'k GPUs at once at peak.']
  ];

  // --- connectors, drawn first so the boxes sit on top ---------------------
  // local helper: a path plus an optional arrowhead at (hx, hy) pointing along (dx, dy)
  var EDGES = [], eg = K.n('g', {});
  body.appendChild(eg);
  function edge(d, owner, hd, o) {
    o = o || {};
    var g = K.n('g', {});
    var p = K.n('path', { d: d, fill: 'none', 'stroke-width': 1.5,
      'stroke-dasharray': o.dash || null, 'stroke-linejoin': 'round' });
    g.appendChild(p);
    var h = null;
    if (hd) {
      var x = hd[0], y = hd[1], dx = hd[2], dy = hd[3], s = 7, w = 4;
      h = K.n('path', { d: 'M' + x + ' ' + y + ' L' + (x - dx * s - dy * w) + ' ' +
        (y - dy * s + dx * w) + ' L' + (x - dx * s + dy * w) + ' ' + (y - dy * s - dx * w) + ' Z' });
      g.appendChild(h);
    }
    eg.appendChild(g);
    var e = { g: g, p: p, h: h, owner: owner, base: o.color || C.ink3 };
    EDGES.push(e);
    return e;
  }
  var MX = SX + SW / 2, LABS = [];
  // the main chain, with the figure's arrow labels beside each hop
  [['prompt', 0], ['episodes', 1], ['scored episodes', 2], ['batches', 3]].forEach(function (a) {
    var y0 = SY[a[1]] + BH + 2, y1 = SY[a[1] + 1] - 2;
    edge('M' + MX + ' ' + y0 + ' V' + y1, a[1], [MX, y1, 0, 1]);
    LABS.push({ owner: a[1], t: K.mono(body, MX + 12, (y0 + y1) / 2 + 4.5, a[0],
      { size: 12.5, color: C.ink3 }) });
  });
  // the runners' bus to the three services (generate, execute, call / verify)
  var RY = SY[1] + BH / 2;
  edge('M' + (SX + SW) + ' ' + RY + ' H' + RX, 5);
  edge('M' + (SX + SW) + ' ' + RY + ' H' + BUS + ' V' + (SY[2] + BH / 2) + ' H' + RX, 6);
  edge('M' + (SX + SW) + ' ' + RY + ' H' + BUS + ' V' + (SY[3] + BH / 2) + ' H' + RX, 7);
  // weight update: trainer to generators, around the right edge
  var TY = SY[4] + BH / 2, GX = RX + RW;
  edge('M' + (SX + SW) + ' ' + TY + ' H628 V' + RY + ' H' + (GX + 2), 4,
    [GX + 2, RY, -1, 0], { dash: '7 4', color: CH.GREY });
  LABS.push({ owner: 4, t: K.mono(body, SX + SW + 34, TY - 9, 'updated policy weights (NCCL)',
    { size: 12.5, color: C.ink3 }) });
  // curriculum feedback: as in Figure 12, from the packing pool back to the
  // dataloader, around the left edge
  var PY = SY[3] + BH / 2, CY = (SY[0] + SY[3] + BH) / 2;
  edge('M' + SX + ' ' + PY + ' H26 V' + (SY[0] + BH / 2) + ' H' + (SX - 2), 3,
    [SX - 2, SY[0] + BH / 2, 1, 0], { dash: '3 4', color: CH.GREY });
  var cf = K.mono(body, 17, CY, 'curriculum feedback', { size: 12.5, color: C.ink3, anchor: 'middle' });
  cf.setAttribute('transform', 'rotate(-90 17 ' + CY + ')');
  LABS.push({ owner: 3, t: cf });

  // --- the boxes, each one a button --------------------------------------
  var TINTED = '#fcf1e8';
  var BX = BOXES.map(function (b, i) {
    var x = b[2], y = b[3], w = b[4];
    // a stacked card behind the three actor pools, as in the figure's "pool of workers"
    if (b[5]) K.panel(body, x + 5, y - 5, w, BH, { r: 8, flat: true });
    var base = K.panel(body, x, y, w, BH, { r: 8 }).firstChild;
    K.mono(body, x + 14, y + 19, b[0], { size: 13.5, weight: 600, color: C.ink });
    K.mono(body, x + 14, y + 36, b[1], { size: 12.5, color: C.ink3 });
    if (b[6]) K.mono(body, x + w - 12, y + 19, b[6], { size: 12.5, color: C.ink2, anchor: 'end' });
    var hit = K.n('rect', { x: x, y: y - (b[5] ? 5 : 0), width: w + (b[5] ? 5 : 0),
      height: BH + (b[5] ? 5 : 0), rx: 8, fill: 'transparent', tabindex: 0, role: 'button',
      'aria-label': b[0], style: 'cursor:pointer;outline:none' });
    hit.addEventListener('click', function () { sel(i); });
    hit.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sel(i); }
    });
    body.appendChild(hit);
    return base;
  });

  // --- the readout: what the chosen box does and passes on -----------------
  var RY0 = SY[4] + BH + 22;
  body.appendChild(K.n('rect', { x: 0, y: RY0, width: 640, height: 98, rx: 9,
    fill: K.TINT.amber, stroke: C.amber, 'stroke-opacity': .45 }));
  var hd = K.mono(body, 16, RY0 + 24, '', { size: 13.5, weight: 600, color: C.ink });
  var ln = [0, 1, 2].map(function (k) {
    return K.mono(body, 16, RY0 + 47 + k * 19, '', { size: 13, color: C.ink2 });
  });

  function sel(i) {
    BX.forEach(function (r, j) {
      var on = j === i;
      r.setAttribute('stroke', on ? C.amber : C.line);
      r.setAttribute('stroke-width', on ? 2 : 1);
      r.setAttribute('fill', on ? TINTED : C.panel);
    });
    EDGES.forEach(function (e) {
      var on = e.owner === i;
      e.p.setAttribute('stroke', on ? C.amber : e.base);
      e.p.setAttribute('stroke-width', on ? 2.2 : 1.5);
      if (e.h) e.h.setAttribute('fill', on ? C.amber : e.base);
    });
    // the highlighted connector goes on top, so a shared bus segment shows it
    EDGES.forEach(function (e) { if (e.owner === i) eg.appendChild(e.g); });
    LABS.forEach(function (l) {
      var on = l.owner === i;
      l.t.setAttribute('fill', on ? C.ink : C.ink3);
      l.t.setAttribute('font-weight', on ? 600 : 400);
    });
    hd.textContent = SAY[i][0];
    var w = K.wrap(SAY[i][1], 74);
    ln.forEach(function (t, k) { t.textContent = w[k] || ''; });
  }
  sel(1);

  K.foot(svg, 'Schematic redrawn from Figure 12. Stacked boxes are Ray actor pools; peak counts ' +
    'are from §4.3.1, which gives the 1:' + RATIO +
    ' ratio, not a GPU split.');
};
