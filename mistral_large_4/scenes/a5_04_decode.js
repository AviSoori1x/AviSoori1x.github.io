/* S_DECODE: §6, Figure 17. One decoding step of an index-producing layer under
   context parallelism, before and after the custom CP kernels. Every bar's start
   and end is exact, extracted from the vector figure (window.ML4.figread.fig17).
   The duration labels (39, 30, 33, 15, 11, 16 us) and the totals are the paper's
   own labels (facts f17.*); the two sums in the note are derived and say so. */
window.SCENES.S_DECODE = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var FR = (api.D.figread || {}).fig17, DV = api.D.derived || {};
  var svg = K.board(host, { alt: 'Figure 17: timeline of one decode step of an index-producing ' +
    'layer under context parallelism, before (' + F['f17.before'] + ' microseconds, one stream) and ' +
    'after (' + F['f17.after'] + ' microseconds, a main and a side stream)' });
  K.head(svg, 'One decode step, ' + F['f17.cut'] + '% shorter',
    'Figure 17 · one decode step · index-producing layer · context parallel');
  if (!FR || !FR.before) { K.mono(svg, 0, 40, 'Figure 17 values are not available.', { size: 14 }); return; }

  // the four kinds of work, in the paper's legend order. Communication is the focus.
  var KIND = {
    proj: { name: 'Projections, norms, RoPE', col: CH.GREY2 },
    indexer: { name: 'Sparse indexer', col: '#7fa7b9' },
    comm: { name: 'Context-parallel communication', col: C.amber },
    attn: { name: 'Sparse attention', col: C.blue }
  };
  var US = 'μs';

  // one shared microsecond axis for both states, 0 to 250, so the shrink is visible
  var sx = CH.scale(0, 250, 8, 612);
  var LANE = { main: 100, side: 246 }, TH = 40, BH = 32;     // track top, track and bar height
  var AXY = 322;                                              // axis baseline

  // The nine bars keep their identity across states, so a switch moves each one to
  // where it runs after the change. Order follows the paper's before stream.
  // nm: the paper's own label, or null where the paper leaves the bar unlabelled.
  var B = FR.before, AM = FR.after_main, AS = FR.after_side;
  var SEG = [
    { b: B[0], a: AM[0], al: 'main', nm: 'Q/KV proj', an: 'Q/KV proj' },
    { b: B[1], a: AS[0], al: 'side', nm: 'indexer', an: 'indexer' },
    { b: B[2], a: AS[1], al: 'side', nm: 'top-k exchange', d: ['f17.b_topk', 'f17.a_topk'] },
    { b: B[3], a: AS[2], al: 'side', nm: 'select', an: 'select' },
    { b: B[4], a: AM[1], al: 'main', nm: 'KV write', an: null, row2: true },
    { b: B[5], a: AM[2], al: 'main', nm: 'Q gather', d: ['f17.b_qg', 'f17.a_qg'] },
    { b: B[6], a: AM[3], al: 'main', nm: 'attention', an: 'attention' },
    { b: B[7], a: AM[4], al: 'main', nm: 'output combine', d: ['f17.b_oc', 'f17.a_oc'] },
    { b: B[8], a: AM[5], al: 'main', nm: 'out proj', an: null }
  ];

  // the shared axis, its gridlines running up through both lanes
  CH.axes(svg, sx(0), 76, sx(250) - sx(0), AXY - 76, { x0: 0, x1: 250,
    xt: [0, 50, 100, 150, 200, 250], noY: true,
    xf: function (v) { return v === 250 ? v + ' ' + US : String(v); } });

  // lane tracks; the side track is restyled per state
  var track = {};
  ['main', 'side'].forEach(function (l) {
    track[l] = K.n('rect', { x: 0, y: LANE[l], width: 620, height: TH, rx: 7,
      fill: K.TINT.ink, stroke: C.line, 'stroke-width': 1 });
    svg.appendChild(track[l]);
  });
  var laneName = {
    main: K.label(svg, 0, LANE.main - 36, 'main stream', { color: C.ink2 }),
    side: K.label(svg, 0, LANE.side - 34, 'side stream', { color: C.ink2 })
  };

  // the time saved, shaded on the main lane between the two end lines (after only)
  var saved = K.n('rect', { x: sx(FR.end_after), y: LANE.main + 3,
    width: sx(FR.end_before) - sx(FR.end_after), height: TH - 6, rx: 5,
    fill: K.TINT.amber, stroke: C.amber, 'stroke-opacity': .5, 'stroke-dasharray': '4 3' });
  svg.appendChild(saved);

  // end-of-step lines, one per state, from the main lane down to the axis. They start
  // below the label row above the main lane, so no line runs through a label.
  var endLine = {};
  [['before', FR.end_before], ['after', FR.end_after]].forEach(function (e) {
    endLine[e[0]] = K.n('path', { d: 'M' + sx(e[1]) + ' ' + LANE.main + ' V' + AXY,
      fill: 'none', 'stroke-width': 1.5 });
    svg.appendChild(endLine[e[0]]);
  });

  // the bars themselves
  SEG.forEach(function (s) {
    s.el = K.n('rect', { rx: 3, height: BH, fill: KIND[s.b.kind].col });
    svg.appendChild(s.el);
  });

  // the key, two by two under the axis, in the paper's legend order
  ['proj', 'indexer', 'comm', 'attn'].forEach(function (k, i) {
    var x = i % 2 ? 330 : 8, y = 380 + Math.floor(i / 2) * 24;
    svg.appendChild(K.n('rect', { x: x, y: y - 11, width: 16, height: 12, rx: 2.5,
      fill: KIND[k].col }));
    K.mono(svg, x + 24, y, KIND[k].name, { size: 13, color: k === 'comm' ? C.ink : C.ink2,
      weight: k === 'comm' ? 600 : 400 });
  });

  var lab = CH.body(svg), note = CH.body(svg);
  // reserve the tallest note, so the foot (placed once) never lands on it
  svg.appendChild(K.n('rect', { x: 0, y: 426, width: 640, height: 3 * 19 + 22, fill: 'none' }));

  // where a bar sits in a state: 0 is before (one stream), 1 is after
  function place(s, st) {
    var g = st ? s.a : s.b, l = st ? s.al : 'main';
    return { x: sx(g.x0), y: LANE[l] + (TH - BH) / 2, w: sx(g.x1) - sx(g.x0), l: l };
  }
  function setBar(s, q) {
    s.el.setAttribute('x', q.x.toFixed(2));
    s.el.setAttribute('y', q.y.toFixed(2));
    s.el.setAttribute('width', Math.max(0.5, q.w).toFixed(2));
    s.cur = q;
  }
  // a short ease between states, so each bar visibly travels to where it runs after
  // the change. A timer lands every bar on its exact final place regardless.
  var anim = 0;
  function moveBars(st, instant) {
    var id = ++anim, from = SEG.map(function (s) { return s.cur; });
    var to = SEG.map(function (s) { return place(s, st); });
    function at(t) {
      var e = t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      SEG.forEach(function (s, i) {
        var a = from[i] || to[i], b = to[i];
        setBar(s, { x: a.x + (b.x - a.x) * e, y: a.y + (b.y - a.y) * e, w: a.w + (b.w - a.w) * e });
      });
    }
    if (instant || !from[0]) { at(1); return; }
    var t0 = null;
    function step(ts) {
      if (id !== anim) return;
      if (t0 == null) t0 = ts;
      var t = Math.min(1, (ts - t0) / 520);
      at(t);
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
    setTimeout(function () { if (id === anim) at(1); }, 700);
  }

  // a label with a short leader to its bar. anchor: middle by default, or pinned to
  // the bar's left or right edge where two labels would otherwise collide.
  function tag(s, q, txt, above, o) {
    o = o || {};
    var cx = q.x + q.w / 2, row2 = o.row2 ? 20 : 0;
    var y = above ? LANE[q.l] - 11 : LANE[q.l] + TH + 17 + row2;
    var ax = o.ax != null ? o.ax : o.anchor === 'end' ? q.x + q.w + 4 :
      o.anchor === 'start' ? q.x - 4 : cx;
    lab.appendChild(K.n('path', { d: above ?
      'M' + cx + ' ' + (y + 4) + ' V' + (q.y - 1) :
      'M' + cx + ' ' + (q.y + BH + 1) + ' V' + (y - 13),
      stroke: C.ink3, 'stroke-width': 1, opacity: .7 }));
    halo(K.mono(lab, ax, y, txt, { size: above ? 13 : 12.5, anchor: o.anchor || 'middle',
      color: above ? C.ink : C.ink2, weight: above ? 600 : 400 }));
  }
  // the end-of-step lines cross some labels; a paper-coloured halo keeps them legible
  function halo(t) {
    t.setAttribute('stroke', '#f1eee7');
    t.setAttribute('stroke-width', 5);
    t.setAttribute('stroke-linejoin', 'round');
    t.setAttribute('paint-order', 'stroke');
    return t;
  }

  // the two totals are the paper's labels; their lines sit at the exact bar ends
  var TOT = [F['f17.before'], F['f17.after']], ENDX = [sx(FR.end_before), sx(FR.end_after)];
  var PIN = { 'Q gather': 'end', 'output combine': 'start' };     // after only

  function draw(st) {
    CH.clear(lab);
    CH.clear(note);
    moveBars(st);

    SEG.forEach(function (s) {
      var q = place(s, st);
      if (s.d) {
        // a communication phase: name and the paper's duration, above its bar
        tag(s, q, s.nm + ' ' + F[s.d[st]] + ' ' + US, true, { anchor: st ? PIN[s.nm] : null });
      } else {
        var nm = st ? s.an : s.nm;
        // out proj ends on the 240 line, so its label sits just left of that line
        var pin = !st && s === SEG[8];
        if (nm) tag(s, q, nm, false, { row2: !st && s.row2, anchor: pin ? 'end' : null,
          ax: pin ? ENDX[0] - 6 : null });
      }
    });

    // the end of the step in this state, and the other state's end as a grey ghost
    ['before', 'after'].forEach(function (k, i) {
      var on = i === st;
      endLine[k].setAttribute('stroke', on ? C.ink : CH.GREY);
      endLine[k].setAttribute('stroke-dasharray', on ? null : '4 4');
      if (on) endLine[k].removeAttribute('stroke-dasharray');
      // the grey ghost starts below the main lane's labels, so it never runs beside a
      // bar or a leader there and reads as a second leader
      endLine[k].setAttribute('d', 'M' + ENDX[i] + ' ' + (on ? LANE.main : LANE.main + TH + 22) +
        ' V' + AXY);
      // the totals sit at the foot of their lines, just above the axis: before's to the
      // left of its line, after's to the right, where both lanes are empty
      K.mono(lab, ENDX[i] + (i ? 6 : -6), AXY - 10, on ? TOT[i] + ' ' + US :
        (i ? 'after: ' : 'was ') + TOT[i] + ' ' + US,
        { size: on ? 14 : 12.5, anchor: i ? 'start' : 'end', color: on ? C.ink : C.ink3, weight: on ? 700 : 400 });
    });

    // the side stream exists only after the change
    track.side.setAttribute('stroke-dasharray', st ? null : '5 4');
    if (st) track.side.removeAttribute('stroke-dasharray');
    track.side.setAttribute('fill', st ? K.TINT.ink : 'none');
    laneName.side.textContent = st ? 'SIDE STREAM' : 'NO SIDE STREAM';
    laneName.side.setAttribute('fill', st ? C.ink2 : C.ink3);
    if (!st) K.mono(lab, 16, LANE.side + TH / 2 + 4.5, 'the indexer waits its turn on the main stream',
      { size: 12.5, color: C.ink3 });

    // the time saved, after only. derived.f17_saved_us is the difference of the two labels
    saved.setAttribute('display', st ? 'inline' : 'none');
    if (st) {
      K.mono(lab, (ENDX[0] + ENDX[1]) / 2, LANE.main + TH / 2 + 4.5,
        DV.f17_saved_us + ' ' + US + ' saved, ' + F['f17.cut'] + '% less',
        { size: 13, anchor: 'middle', color: C.ink, weight: 600 });
      // the main stream's empty stretch, read off the bar positions: it holds no bar
      // from the end of Q/KV proj until the side stream's select has ended
      var g0 = sx(AM[0].x1) + 4, g1 = sx(AM[1].x0) - 4, gy = LANE.main + TH / 2;
      lab.appendChild(K.n('path', { d: 'M' + g0 + ' ' + gy + ' H' + g1, stroke: C.ink3,
        'stroke-width': 1, 'stroke-dasharray': '3 3', fill: 'none' }));
      K.mono(lab, g0, LANE.main + TH + 37, 'main stream empty until select ends',
        { size: 12.5, color: C.ink3 });
    }

    // the note: the sums of the three exchanges are derived from the labels
    var sb = F['f17.b_topk'] + F['f17.b_qg'] + F['f17.b_oc'];
    var sa = F['f17.a_topk'] + F['f17.a_qg'] + F['f17.a_oc'];
    K.callout(note, 0, 426, 640, st ?
      'The exchanges shrink to ' + F['f17.a_topk'] + ' + ' + F['f17.a_qg'] + ' + ' +
      F['f17.a_oc'] + ' = ' + sa + ' ' + US + '. The paper credits hardware multicast, ' +
      'latency-first exchange patterns, removing in-kernel synchronization and fusing ' +
      'compute into the communication kernels.' :
      'One stream, so each operation waits for the one before it. The three exchanges alone take ' +
      F['f17.b_topk'] + ' + ' + F['f17.b_qg'] + ' + ' + F['f17.b_oc'] + ' = ' + sb + ' of the ' +
      TOT[0] + ' ' + US + '.', { cols: 74 });
  }

  K.switcher(svg, 0, 0, ['before', 'after'], draw, { tint: 'amber' });
  K.foot(svg, 'Bar positions and lengths are exact, extracted from the vector figure; the ' +
    'durations and totals are the paper\'s labels. Derived from them: the sums in the note, and ' +
    DV.f17_saved_us + ' ' + US + ' saved (' + TOT[0] + ' minus ' + TOT[1] + '). After the change, the two short grey bars carry no label, as ' +
    'in the paper.');
};
