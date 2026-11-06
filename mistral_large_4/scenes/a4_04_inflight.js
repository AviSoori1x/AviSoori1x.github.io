/* S_INFLIGHT: in-flight weight updates (§4.4.1). Generation is not drained before an
   update, so one long rollout is written by many policy versions and its later tokens
   attend to KV entries that older policies computed. Schematic: the paper gives no
   update timestamps, so updates are drawn evenly spaced and the figure says so. */
window.SCENES.S_INFLIGHT = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var HRS = F['rlinf.longest_h'], UPD = F['rlinf.updates'], STEPK = F['rl.step_traj_k'];
  var STALE = F['KEYa4_04.stale'] || F['mis.stale'];
  var NV = UPD + 1;          // versions v0 .. v136: the starting policy plus one per update
  var WIN = 7;               // versions shown in the zoom: the current one and six before it
  var CELLS = 4;             // tokens drawn per version in the zoom (schematic)
  var svg = K.board(host, { alt: 'Schematic of one long rollout written by many policy ' +
    'versions, with new weights loaded mid-rollout and the KV cache kept' });
  K.head(svg, 'One rollout, many policies', '§4.4.1, §4.4.2 · in-flight weight updates · schematic');

  // ---- overview: the whole rollout as one strip, one band per policy version ----
  var SX = 12, SW = 616, BW = SW / NV;
  K.label(svg, 0, 14, 'the longest rollout · about ' + HRS + ' hours · ' + UPD + ' updates');
  svg.appendChild(K.n('rect', { x: SX, y: 28, width: SW, height: 26, rx: 4,
    fill: K.TINT.ink, stroke: C.line }));
  var bands = [];
  for (var b = 0; b < NV; b++) {
    var r = K.n('rect', { x: SX + b * BW, y: 28, width: BW + 0.2, height: 26, fill: 'none' });
    svg.appendChild(r);
    bands.push(r);
  }
  var brk = K.n('rect', { x: SX, y: 23, width: 10, height: 36, rx: 3, fill: 'none',
    stroke: C.ink, 'stroke-width': 1.6 });
  svg.appendChild(brk);
  var out = CH.readout(svg, 0, 124, { size: 13.5 });

  // ---- what every update does, while generation keeps running ----
  K.label(svg, 0, 160, 'at every update, with generation not drained');
  var STEPS = [['trainer steps at', STEPK + 'k usable trajectories'],
    ['rollout pauses', 'mid-generation'], ['loads the', 'new weights'], ['resumes; KV is', 'not recomputed']];
  // the first box is wider so "7k usable trajectories" fits: 185 + 3 * 130 + 3 * 20 = 635
  var PW0 = 185, PW1 = 130, PG = 20, PY = 172, PH = 48;
  STEPS.forEach(function (s, i) {
    var first = i === 0, PW = first ? PW0 : PW1, x = first ? 0 : PW0 + PG + (i - 1) * (PW1 + PG);
    svg.appendChild(K.n('rect', { x: x, y: PY, width: PW, height: PH, rx: 8,
      fill: first ? '#fff' : K.TINT.amber, stroke: first ? C.line : C.amber,
      'stroke-opacity': first ? 1 : 0.5 }));
    K.mono(svg, x + 11, PY + 20, s[0], { size: 12.5, color: C.ink, weight: 600 });
    K.mono(svg, x + 11, PY + 38, s[1], { size: 12.5, color: C.ink2 });
    if (i) K.arrow(svg, x - PG + 3, PY + PH / 2, x - 3, PY + PH / 2,
      { color: C.ink3, w: 1.4, headSize: 5 });
  });

  // ---- zoom: the KV behind the cursor, coloured by the policy that computed it ----
  K.label(svg, 0, 254, 'zoom: KV behind the cursor, by the policy that computed it');
  var ZW = 640 / WIN, CY = 286, CH_ = 28, CW = 17, CP = 20.5;
  var zoom = CH.body(svg);
  // key: the two kinds of KV entry
  [[C.amber, 1, 'KV from the policy writing now'], [CH.GREY, 0.8, 'KV from older policies']]
    .forEach(function (k, i) {
      var kx = i ? 270 : 0;
      svg.appendChild(K.n('rect', { x: kx, y: 404, width: 14, height: 14, rx: 3, fill: k[0],
        opacity: k[1] }));
      K.mono(svg, kx + 22, 416, k[2], { size: 12.5, color: C.ink2 });
    });
  // third key entry: the vertical lines in the zoom are the weight updates
  svg.appendChild(K.n('path', { d: 'M476 402 V420', stroke: C.ink3, 'stroke-width': 1.3,
    'stroke-dasharray': '3 3' }));
  K.mono(svg, 488, 416, 'weight update', { size: 12.5, color: C.ink2 });
  // reserve the callout's box, so the foot (placed once, after layout) never lands on it
  svg.appendChild(K.n('rect', { x: 0, y: 432, width: 640, height: 3 * 19 + 22, fill: 'none' }));
  K.callout(svg, 0, 432, 640, 'Long rollouts keep generating while the policy moves on under them. ' +
    'Over training the average staleness was ' + STALE + ' steps; importance weighting in the ' +
    'objective corrects for this lag.', { cols: 74 });

  function ageFill(a) { return a === 0 ? C.amber : CH.GREY; }
  function ageOp(a) { return a === 0 ? 1 : Math.max(0.42, 1 - (a - 1) * 0.11); }

  function draw(u) {
    var k = Math.min(UPD, Math.floor(u + 1e-6));            // version writing now
    var fill = Math.min(CELLS, Math.floor((u - k) * CELLS + 1e-6) + 1);   // its tokens so far
    var lo = Math.max(0, k - (WIN - 1));
    // overview bands: written versions in alternating greys, the current one amber
    bands.forEach(function (r, v) {
      r.setAttribute('fill', v === k ? C.amber : v < k ? (v % 2 ? CH.GREY : CH.GREY2) : 'none');
    });
    brk.setAttribute('x', SX + lo * BW - 2.5);
    brk.setAttribute('width', (k - lo + 1) * BW + 5);

    CH.clear(zoom);
    var newest = null, targets = [];
    for (var v = lo; v <= k; v++) {
      var s = v - lo, a = k - v, x0 = s * ZW + 4;
      K.mono(zoom, x0 + (3 * CP + CW) / 2, CY - 10, 'v' + v, { size: 12.5, anchor: 'middle',
        color: a ? C.ink3 : C.ink, weight: a ? 400 : 600 });
      for (var c = 0; c < CELLS; c++) {
        var cx = x0 + c * CP, done = a > 0 || c < fill;
        zoom.appendChild(K.n('rect', { x: cx, y: CY, width: CW, height: CH_, rx: 3,
          fill: done ? ageFill(a) : 'none', opacity: done ? ageOp(a) : 1,
          stroke: done ? 'none' : C.ink3, 'stroke-dasharray': done ? null : '3 3' }));
        if (a === 0 && c === fill - 1) newest = cx + CW / 2;
        if (a > 0 && c === 1) targets.push(cx + CW / 2);
      }
      // the update boundary in front of this version: the newest one in amber
      if (v > 0) {
        var bx = Math.max(0.8, s * ZW - 2.4);
        zoom.appendChild(K.n('path', { d: 'M' + bx + ' ' + (CY - 24) + ' V' + (CY + CH_ + 6),
          stroke: a ? C.ink3 : C.amber, 'stroke-width': a ? 1.1 : 2.2,
          'stroke-dasharray': a ? '3 3' : null }));
      }
    }

    // attention from the newest token back into KV that older policies computed. ML4 uses
    // sparse attention (DSA, top-2,048 keys per query), so the text says "can attend"
    var base = CY + CH_ + 2;
    targets.forEach(function (tx) {
      var d = Math.abs(newest - tx), dep = 18 + d * 0.15;
      zoom.appendChild(K.n('path', { d: 'M' + newest + ' ' + base + ' Q' + (newest + tx) / 2 +
        ' ' + (base + dep) + ' ' + tx + ' ' + base, fill: 'none', stroke: C.ink2,
        'stroke-width': 1.2, opacity: 0.75 }));
      zoom.appendChild(K.n('circle', { cx: tx, cy: base + 1, r: 2.6, fill: C.ink2 }));
    });
    zoom.appendChild(K.n('circle', { cx: newest, cy: base + 1, r: 3.6, fill: C.amber,
      stroke: '#fff', 'stroke-width': 1.4 }));
    K.mono(zoom, 0, 384, k === 0 ?
      'every token so far was written by v0, the policy the rollout started with' :
      'the newest token may attend to KV entries that earlier policy versions computed',
      { size: 13, color: C.ink2 });

    // where the cursor sits; the hours assume evenly spaced updates (illustrative)
    var hrs = u / NV * HRS;
    out.set(k === 0 ? 'v0, the starting policy, is writing · no updates yet' :
      'v' + k + ' is writing · update ' + k + ' of ' + UPD + ' · about ' +
      Math.round(hrs) + ' h in, if evenly spaced');
  }

  var sl = CH.slider(svg, SX, 84, SW, { min: 0, max: NV, step: 1 / CELLS, value: 88.5,
    label: 'position along the rollout', onChange: draw,
    fmt: function (u) { return 'version ' + Math.min(UPD, Math.floor(u)); } });
  // the rollout strip itself is draggable too (local helper; CH.slider only listens on its track)
  var hit = K.n('rect', { x: SX, y: 23, width: SW, height: 36, fill: 'transparent',
    style: 'cursor:pointer' });
  svg.appendChild(hit);
  var dragging = false;
  function seek(e) {
    var pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    var q = pt.matrixTransform(hit.getScreenCTM().inverse());
    sl.set(NV * (q.x - SX) / SW);
  }
  hit.addEventListener('pointerdown', function (e) {
    dragging = true; try { hit.setPointerCapture(e.pointerId); } catch (er) {}
    seek(e); e.preventDefault();
  });
  hit.addEventListener('pointermove', function (e) { if (dragging) seek(e); });
  hit.addEventListener('pointerup', function () { dragging = false; });
  hit.addEventListener('pointercancel', function () { dragging = false; });
  K.foot(svg, 'Schematic. The paper gives no update timestamps, so updates are drawn evenly ' +
    'spaced; the average gap, ' + HRS + ' hours over ' + UPD + ' updates, is about ' +
    Math.round(HRS * 60 / UPD) + ' minutes (derived). Four tokens per version is illustrative.');
};
