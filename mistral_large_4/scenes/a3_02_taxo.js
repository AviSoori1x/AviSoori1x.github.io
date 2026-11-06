/* S_TAXO: Figure 10, the economy and capabilities taxonomies, redrawn as two
   trees. Every branch name is the figure's own; only the branches on a picked
   task's path are opened. The figure traces only the knowledge-work task (its
   three red nodes). For the code and math tasks the economy leaf whose wording
   matches is my reading, drawn dashed and said so in the foot. */
window.SCENES.S_TAXO = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH;
  var svg = K.board(host, { alt: 'Figure 10: the economy taxonomy and the capabilities taxonomy, ' +
    'and an example task sampled from both' });
  K.head(svg, 'Two trees, one task', 'Figure 10 · schematic redraw · one seed drawn from both trees');
  var body = CH.body(svg);

  var ECON = ['Healthcare practitioners', 'Business & finance', 'Management', 'Legal',
    'Office & administrative', 'Technical & math domains'];
  var CAPS = ['Success criteria', 'Underspecification', 'Reasoning', 'Data analysis',
    'Document capabilities', 'User persona', 'User behaviors'];
  // per example: economy [category, child, leaf], capability leaves by category,
  // whether Figure 10 itself draws the link, and the task text
  var EX = [
    { econ: [1, 'Accountants & auditors', 'Month-end close'], caps: { 2: 'Temporal ordering',
      4: 'Cross-document reconciliation' }, traced: true,
      task: 'Prepare the month-end close report from the ledger and the reconciliation notes. ' +
        'Use revision history to resolve conflicting figures.' },
    { econ: [5, 'Distributed systems', 'Repair a retry handler'], caps: {}, traced: false,
      task: 'Fix duplicate charges when a timed-out payment is retried. Preserve the public API ' +
        'and add a regression test that each payment is charged at most once.' },
    { econ: [5, 'Optimization', 'Optimize production'], caps: {}, traced: false,
      task: 'Maximize a linear objective subject to linear constraints. Find the optimal ' +
        'production plan and prove that no feasible plan earns more.' }
  ];
  var TOP = 104, ROW = 26, P0 = 346, CW = 7.8;   // CW: mono advance at 13px

  function node(p, x, y, on, r) {
    p.appendChild(K.n('circle', { cx: x, cy: y, r: r || 4.5, fill: on ? C.amber : '#fff',
      stroke: on ? C.amber : CH.GREY, 'stroke-width': 1.6 }));
  }
  function elbow(p, x, y0, y1, x1, on) {
    p.appendChild(K.n('path', { d: 'M' + x + ' ' + y0 + ' V' + y1 + ' H' + x1, fill: 'none',
      stroke: on ? C.amber : CH.GREY2, 'stroke-width': on ? 2 : 1.3 }));
  }
  // one tree; rows open under the categories on the path. Returns the leaf ends.
  function tree(X, title, ask, cats, open) {
    K.label(body, X, 50, title);
    K.mono(body, X, 69, ask, { size: 12.5, color: C.ink3 });
    var g = K.n('g', {}), lines = K.n('g', {});
    body.appendChild(lines); body.appendChild(g);
    body.appendChild(K.n('circle', { cx: X + 8, cy: 86, r: 6, fill: C.ink }));
    var y = TOP, ends = [], lastCat = TOP;
    cats.forEach(function (c, i) {
      var kids = open[i] || [], on = kids.length > 0;
      node(g, X + 8, y, on);
      K.mono(g, X + 22, y + 4.5, c, { size: 13, color: on ? C.ink : C.ink2, weight: on ? 600 : 400 });
      lastCat = y;
      var py = y, px = X + 18;
      kids.forEach(function (kid, d) {
        y += ROW;
        var kx = X + 30 + d * 22;
        elbow(lines, px, py + 6, y, kx - 5, true);
        node(g, kx, y, true, d === kids.length - 1 ? 5 : 4);
        K.mono(g, kx + 13, y + 4.5, kid, { size: 13, color: C.ink, weight: 600 });
        if (d === kids.length - 1) ends.push([kx + 13 + kid.length * CW + 6, y]);
        py = y; px = kx + 10;
      });
      y += ROW;
    });
    lines.appendChild(K.n('path', { d: 'M' + (X + 8) + ' 86 V' + lastCat, stroke: CH.GREY2,
      'stroke-width': 1.3, fill: 'none' }));
    return { ends: ends, bottom: y };
  }

  function draw(k) {
    CH.clear(body);
    var ex = EX[k], eo = {}, co = {};
    eo[ex.econ[0]] = [ex.econ[1], ex.econ[2]];
    Object.keys(ex.caps).forEach(function (c) { co[c] = [ex.caps[c]]; });
    var E = tree(0, 'economy taxonomy', 'what work is being done?', ECON, eo);
    var Q = tree(340, 'capabilities taxonomy', 'what should the task require?', CAPS, co);

    // the sampled leaves feed the task-authoring agent, routed clear of every label
    var route = K.n('g', {});
    body.appendChild(route);
    function feed(end, gx) {
      route.appendChild(K.n('path', { d: 'M' + end[0] + ' ' + end[1] + ' H' + gx + ' V' + (P0 - 2),
        fill: 'none', stroke: C.amber, 'stroke-width': 2, 'stroke-dasharray': ex.traced ? null : '5 4' }));
      K.arrow(route, gx, P0 - 9, gx, P0 - 1, { color: C.amber, w: 2, headSize: 7 });
    }
    E.ends.forEach(function (e) { feed(e, 318); });
    Q.ends.forEach(function (e) { feed(e, 628); });
    if (!ex.traced) {
      K.mono(body, 340, Q.bottom + 8, 'not traced in Figure 10 for this task', { size: 12.5, color: C.ink3 });
    }

    // the agent writes the task
    body.appendChild(K.n('rect', { x: 0, y: P0, width: 640, height: 112, rx: 9, fill: '#fff',
      stroke: C.line }));
    body.appendChild(K.n('path', { d: 'M0 ' + (P0 + 9) + ' Q0 ' + P0 + ' 9 ' + P0 + ' H631 Q640 ' + P0 +
      ' 640 ' + (P0 + 9) + ' V' + (P0 + 32) + ' H0 Z', fill: C.ink }));
    K.mono(body, 16, P0 + 21, 'task-authoring agent: one workflow plus capabilities', { size: 13,
      color: '#fff', weight: 600 });
    K.wrap(ex.task, 80).forEach(function (ln, i) {
      K.text(body, 16, P0 + 58 + i * 22, ln, { size: 15, color: C.ink });
    });
  }

  K.switcher(svg, 0, 0, ['knowledge work', 'code', 'math'], draw, { tint: 'amber' });
  K.foot(svg, 'A schematic redraw of Figure 10 that opens only the branches on a task\'s path. The ' +
    'figure traces only the knowledge-work task; for code and math the dashed link marks the ' +
    'matching leaf, my reading. The math task is shortened.');
};
