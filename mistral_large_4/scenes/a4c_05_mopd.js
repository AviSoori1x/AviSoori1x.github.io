/* S_MOPD: multi-teacher on-policy distillation (§4.5.2) and Figure 15, the student's
   verbosity relative to high reasoning effort through training. Exact values,
   extracted from the vector figure (window.ML4.figread.fig15). */
window.SCENES.S_MOPD = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, FR = (api.D.figread || {}).fig15;
  var svg = K.board(host, { alt: 'Figure 15: student verbosity relative to high reasoning effort during MOPD' });
  K.head(svg, 'Three teachers, one student', 'Figure 15 · verbosity during MOPD · relative to high effort');
  if (!FR || !FR['medium/high']) { K.mono(svg, 0, 40, 'Figure 15 values are not available.', { size: 14 }); return; }

  var MODES = [
    { key: 'medium/high', name: 'medium' },
    { key: 'low/high', name: 'low' }
  ];
  MODES.forEach(function (m) {
    var s = FR[m.key];
    m.pts = s.student.map(function (q) { return [q.pct, q.v]; });
    m.target = s.target[0];
    m.last = m.pts[m.pts.length - 1];
    // the settled level: the mean of the checkpoints in the last fifth of training
    // (80% onward). Any single point is noisy, so the final one is not quoted alone.
    var tail = m.pts.filter(function (q) { return q[0] >= 80; });
    m.mean = tail.reduce(function (acc, q) { return acc + q[1]; }, 0) / tail.length;
    // the drop: the first point within 10% of the target, then the last point still at
    // 0.95 or more of high's length before it
    var j = 0;
    while (j < m.pts.length && m.pts[j][1] > m.target * 1.1) j++;
    var i = j;
    while (i > 0 && m.pts[i][1] < 0.95) i--;
    m.until = m.pts[i][0]; m.by = m.pts[Math.min(j, m.pts.length - 1)][0];
  });

  // the schematic: three teachers feed one student
  var top = CH.body(svg), TBW = 200, TBX = [0, 220, 440];
  var TEACH = [['high teacher', 'high-reasoning RL'], ['medium teacher', 'length-penalized RL'],
    ['low teacher', 'length-penalized RL']];
  var tbox = TEACH.map(function (t, i) {
    var r = K.n('rect', { x: TBX[i], y: 46, width: TBW, height: 50, rx: 9, fill: '#fff', stroke: C.line,
      'stroke-width': 1 });
    top.appendChild(r);
    K.mono(top, TBX[i] + 14, 66, t[0], { size: 13.5, weight: 600, color: C.ink });
    K.mono(top, TBX[i] + 14, 86, t[1], { size: 13, color: C.ink3 });
    var a = K.n('path', { d: 'M' + (TBX[i] + TBW / 2) + ' 98 V118', stroke: C.ink3, 'stroke-width': 1.4 });
    top.appendChild(a);
    return { r: r, a: a };
  });
  K.panel(top, 0, 122, 640, 50, { r: 9 });
  K.mono(top, 14, 142, 'student: starts from the high teacher', { size: 13.5, weight: 600, color: C.ink });
  K.mono(top, 14, 162, 'each mode: per-token KL to that teacher\'s full distribution', { size: 13, color: C.ink3 });

  // Figure 15
  var ax = CH.axes(svg, 46, 214, 494, 160, { x0: 0, x1: 100, y0: 0, y1: 1.25,
    xt: [0, 25, 50, 75, 100], yt: [0, 0.5, 1],
    xf: function (v) { return v + '%'; }, yf: function (v) { return CH.fmt(v, 1); },
    yl: 'verbosity relative to high effort' });
  CH.key(svg, ax.sx(58), ax.sy(1.12), 'student', { color: C.ink2 });
  CH.key(svg, ax.sx(80), ax.sy(1.12), 'target', { color: C.ink2, dash: '5 4', dot: false });
  var plot = CH.body(svg);
  var names = MODES.map(function (m) {
    return K.mono(svg, 552, ax.sy(m.last[1]) + 4.5, m.name, { size: 13, weight: 600, color: C.ink2 });
  });
  var cur = K.n('path', { d: '', stroke: C.ink, 'stroke-width': 1.2, 'stroke-dasharray': '3 3' });
  svg.appendChild(cur);
  var dot = K.n('circle', { r: 6.5, fill: C.amber, stroke: '#fff', 'stroke-width': 2.2 });
  svg.appendChild(dot);
  var out = CH.readout(svg, 0, 452, { size: 13.5 });
  // reserve the callout, so the foot (placed once) never lands on it
  svg.appendChild(K.n('rect', { x: 0, y: 468, width: 640, height: 3 * 19 + 22, fill: 'none' }));
  var note = CH.body(svg);

  var sel = 0, pct = 98;
  function at(m, p) {
    var best = m.pts[0];
    m.pts.forEach(function (q) { if (Math.abs(q[0] - p) < Math.abs(best[0] - p)) best = q; });
    return best;
  }
  function move(p) {
    pct = p;
    var m = MODES[sel], q = at(m, p);
    cur.setAttribute('d', 'M' + ax.sx(q[0]) + ' ' + ax.sy(0) + ' V' + ax.sy(q[1]));
    dot.setAttribute('cx', ax.sx(q[0]));
    dot.setAttribute('cy', ax.sy(q[1]));
    out.set('at ' + q[0] + '% of training: ' + m.name + ' runs at ' + CH.fmt(q[1], 2) +
      ' of high\'s length, target ' + CH.fmt(m.target, 3));
  }

  function draw(s) {
    sel = s;
    CH.clear(plot);
    // the other mode first, in grey, so the selected one sits on top
    [1 - s, s].forEach(function (k) {
      var m = MODES[k], on = k === s, col = on ? C.amber : CH.GREY;
      plot.appendChild(K.n('path', { d: 'M46 ' + ax.sy(m.target) + ' H540', stroke: col,
        'stroke-width': on ? 1.8 : 1.4, 'stroke-dasharray': '5 4', opacity: on ? 1 : 0.8 }));
      CH.line(plot, ax, m.pts, { color: col, w: on ? 2.4 : 1.8, opacity: on ? 1 : 0.8 });
      CH.dots(plot, ax, m.pts, { color: col, r: on ? 2.9 : 2.3, opacity: on ? 1 : 0.8 });
      names[k].setAttribute('fill', on ? C.ink : C.ink3);
    });
    tbox.forEach(function (t, i) {
      var on = i === s + 1;
      t.r.setAttribute('stroke', on ? C.amber : C.line);
      t.r.setAttribute('stroke-width', on ? 2 : 1);
      t.r.setAttribute('fill', on ? K.TINT.amber : '#fff');
      t.a.setAttribute('stroke', on ? C.amber : C.ink3);
      t.a.setAttribute('stroke-width', on ? 2 : 1.4);
    });
    var m = MODES[s];
    CH.clear(note);
    K.callout(note, 0, 468, 640, m.name.charAt(0).toUpperCase() + m.name.slice(1) + ' mode: close ' +
      'to high\'s length until ' + m.until + '% of training, down near its target by ' + m.by +
      '% of training, then hovering just ' + (m.mean < m.target ? 'under' : 'over') + ' it, averaging ' +
      CH.fmt(m.mean, 2) + ' over the last fifth against a target of ' + CH.fmt(m.target, 3) + '.',
      { cols: 74 });
    move(pct);
  }

  CH.slider(svg, 46, 422, 494, { min: 2, max: 98, step: 2, value: 98, label: 'training completion',
    fmt: function (p) { return p + '%'; }, onChange: move });
  K.switcher(svg, 0, 0, ['medium / high', 'low / high'], draw, { tint: 'amber' });
  K.foot(svg, 'The teacher diagram is schematic; the Figure 15 curves are exact values, extracted from ' +
    'the vector figure. Teachers run on separate GPU nodes and send only hidden states; the logits ' +
    'are rebuilt on the trainer.');
};
