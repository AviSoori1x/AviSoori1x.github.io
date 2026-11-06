/* S_CP: §6 production inference. Tensor parallelism replicates the KV cache on
   every rank; context parallelism shards it along the sequence. Schematic: the
   cell count and the four contiguous slices are illustrative, not measured. The
   numbers (20M tokens, 4 x GB300, +51%, 10% to 15%) are the paper's own. */
window.SCENES.S_CP = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var G = F['cp.gpus'], NC = 16, PER = NC / G;        // 16 cells, one quarter per rank
  var svg = K.board(host, { alt: 'Schematic: where the KV cache of one long sequence lives on ' +
    G + ' GPUs under tensor parallelism and under context parallelism' });
  K.head(svg, 'Where the KV cache lives', '§6 · schematic · one long sequence on ' + G + ' GPUs');

  // geometry: GPU rows on the left, the CP exchange bus on the far right
  var X0 = 76, CW = 24, X1 = X0 + NC * CW, RX = X1 + 16;      // strip and row label x
  var RY = [132, 186, 240, 294], RH = 44, PW = 584, BUS = 614;
  var ORD = ['1st quarter', '2nd quarter', '3rd quarter', '4th quarter'];

  // the sequence itself, in token order; the quarter cuts show only under CP.
  // Each block is its own group, so the page's entrance stagger runs row by row.
  var seq = CH.body(svg);
  K.label(seq, 0, 58, 'the KV cache of one long sequence, in token order');
  K.mono(seq, 0, 89, 'sequence', { size: 13, color: C.ink2, weight: 600 });
  for (var c = 0; c < NC; c++) {
    seq.appendChild(K.n('rect', { x: X0 + c * CW + 1, y: 72, width: CW - 2, height: 24, rx: 3,
      fill: CH.GREY2, stroke: 'none' }));
  }
  var cuts = CH.body(seq);
  for (var q = 1; q < G; q++) {
    var qx = X0 + q * PER * CW;
    cuts.appendChild(K.n('path', { d: 'M' + qx + ' 66 V102', stroke: C.ink2, 'stroke-width': 1.6,
      'stroke-dasharray': '3 2' }));
  }
  K.mono(seq, X0, 118, 'token 1', { size: 12.5, color: C.ink3 });
  K.mono(seq, X1, 118, 'newest token', { size: 12.5, color: C.ink3, anchor: 'end' });
  var seqTag = K.mono(seq, RX, 89, '', { size: 13, color: C.ink3 });

  // one row per GPU; cells and the right-hand label are restyled per state
  var rows = RY.map(function (y, i) {
    var g = CH.body(svg);
    var box = K.panel(g, 0, y, PW, RH, { r: 9 }).firstChild;
    K.mono(g, 14, y + 27, 'GPU ' + i, { size: 13.5, color: C.ink, weight: 600 });
    var cells = [];
    for (var k = 0; k < NC; k++) {
      var r = K.n('rect', { x: X0 + k * CW + 1, y: y + 10, width: CW - 2, height: 24, rx: 3 });
      g.appendChild(r);
      cells.push(r);
    }
    var tag = K.mono(g, RX, y + 27, '', { size: 13, color: C.ink2 });
    var tick = K.n('path', { d: 'M' + PW + ' ' + (y + RH / 2) + ' H' + BUS, fill: 'none',
      'stroke-width': 2.2 });
    g.appendChild(tick);
    return { box: box, cells: cells, tag: tag, tick: tick };
  });
  var bus = K.n('path', { d: 'M' + BUS + ' ' + (RY[0] + RH / 2) + ' V' + (RY[G - 1] + RH / 2),
    fill: 'none', 'stroke-width': 2.2, 'stroke-linecap': 'round' });
  svg.appendChild(bus);

  // the line under the rows: what the layout means, then the bus key
  var gist = K.mono(svg, 0, 364, '', { size: 13.5, color: C.ink, weight: 600 });
  var keyG = CH.body(svg), low = CH.body(svg);
  // reserve the lowest block, so the foot (placed once) never lands on a state's text
  svg.appendChild(K.n('rect', { x: 0, y: 404, width: 640, height: 112, fill: 'none' }));

  // set or drop an attribute (setAttribute(x, null) would write the string "null")
  function attr(el, k, v) { if (v == null) el.removeAttribute(k); else el.setAttribute(k, v); }

  function draw(s) {
    var cp = s > 0, col = cp ? C.amber : C.blue;
    rows.forEach(function (row, i) {
      row.cells.forEach(function (cell, k) {
        var mine = !cp || Math.floor(k / PER) === i;
        cell.setAttribute('fill', mine ? col : 'none');
        cell.setAttribute('stroke', mine ? 'none' : CH.GREY2);
        attr(cell, 'stroke-dasharray', mine ? null : '3 2');
      });
      row.tag.textContent = cp ? ORD[i] : 'full copy';
      attr(row.tick, 'display', cp ? null : 'none');
      row.tick.setAttribute('stroke', s === 1 ? C.red : C.amber);
      attr(row.tick, 'stroke-dasharray', s === 1 ? '4 3' : null);
    });
    attr(bus, 'display', cp ? null : 'none');
    bus.setAttribute('stroke', s === 1 ? C.red : C.amber);
    attr(bus, 'stroke-dasharray', s === 1 ? '4 3' : null);
    attr(cuts, 'display', cp ? null : 'none');
    seqTag.textContent = cp ? 'cut in ' + G + ' quarters' : 'copied to all ranks';

    gist.textContent = cp ? 'Each token\'s KV is stored on one rank only: sharded, not replicated.' :
      'Every rank stores the KV of every token: the cache is replicated.';
    CH.clear(keyG);
    if (s === 1) CH.key(keyG, 0, 388, 'standard CP collectives: slow, and run at every decode step',
      { color: C.red, dash: '4 3', dot: false });
    if (s === 2) CH.key(keyG, 0, 388, 'rewritten CP collectives: latency-first, overlapped, fused',
      { color: C.amber, dot: false });
    CH.clear(low);
    if (s === 0) lowTP();
    if (s === 1) lowCP();
    if (s === 2) lowFix();
  }

  // one stat tile: a mono caption, a display word or number, a plain line under it
  function tile(x, cap, word, sub, o) {
    o = o || {};
    K.label(low, x, 416, cap);
    K.big(low, x, 470, word, { size: o.size || 50, color: o.color || C.ink });
    K.mono(low, x, 500, sub, { size: 13, color: C.ink2 });
  }

  // tensor parallelism: the best latency, paid for in replicated memory
  function lowTP() {
    tile(0, 'latency', 'best', 'plain TP gives ML4 its best latency');
    tile(336, 'concurrency, cache hit rate', 'limited', 'KV replication severely limits both',
      { size: 40, color: C.ink2 });
  }

  // context parallelism: the capacity, and the catch
  function lowCP() {
    K.big(low, 0, 456, '~' + F['cp.kv_m'] + 'M', { size: 54 });
    K.mono(low, 160, 430, 'tokens of in-memory KV-cache space on ' + G + ' × GB300',
      { size: 13, color: C.ink, weight: 600 });
    K.mono(low, 160, 450, 'with CP, a native FP8 KV cache and offloading combined',
      { size: 13, color: C.ink2 });
    K.callout(low, 0, 472, 640, 'The catch: kernel inefficiencies made standard CP much worse ' +
      'than TP.', { color: C.red, tint: K.TINT.red, cols: 72 });
  }

  // the fix: rewritten collectives
  function lowFix() {
    // measured on the Figure 16 baseline hardware (8 x GB300), not on the 4-GPU schematic
    tile(0, 'decode interactivity, ' + F['fig16.base_gpus'] + ' × GB300', '+' + F['cp.dcp_gain'] + '%',
      'rewritten vs off-the-shelf CP kernels');
    tile(336, 'communication overhead', F['cp.ovh_lo'] + '% to ' + F['cp.ovh_hi'] + '%',
      'at most, compared with TP', { size: 40, color: C.ink2 });
  }

  K.switcher(svg, 0, 0, ['tensor parallel', 'context parallel', 'CP, rewritten kernels'], draw,
    { tint: 'amber' });
  K.foot(svg, 'Schematic: the 16 cells and the four contiguous slices are illustrative, not ' +
    'measured. The ' + F['cp.kv_m'] + 'M tokens figure needs CP together with a native FP8 KV ' +
    'cache and offloading. Figure 16 measures the ' + F['cp.dcp_gain'] + '% on ' +
    F['fig16.base_gpus'] + ' × GB300, at ' +
    F['KEYa5_03.dcp_conc'] + ' concurrent requests and ' + F['fig16.input_k'] + 'K input tokens.');
};
