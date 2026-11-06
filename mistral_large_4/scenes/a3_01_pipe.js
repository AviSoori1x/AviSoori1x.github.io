/* S_PIPE: Figure 9, the task generation pipeline, redrawn top to bottom so the
   portrait panel holds it. Choose where a candidate fails: the seed filter
   rejects outright, every later gate sends the candidate back to its seed with
   feedback. A schematic; the figure carries no numbers. */
window.SCENES.S_PIPE = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH;
  var svg = K.board(host, { alt: 'Figure 9: seed, seed filter, cheap checks, expensive checks, ' +
    'difficulty calibration, accepted task, with failures sent back to the seed' });
  K.head(svg, 'Every task runs a gauntlet', 'Figure 9 · schematic redraw · later failures loop back to the seed');
  var body = CH.body(svg);

  // [name, detail, top y]; the gap after the seed filter is wider, because the
  // candidate task is only generated once a seed has passed it
  var ST = [
    ['Seed', 'an environment plus a behavior', 70],
    ['Seed filter', '', 134],
    ['Cheap quality checks', 'LLM judges, polarity checks', 214],
    ['Expensive checks', 'agentic verifier, reward hacking', 278],
    ['Difficulty calibration', 'enough learning signal?', 342],
    ['Accepted task', '', 406]
  ];
  var BW = 290, BH = 44, RX = 590;
  var mid = function (i) { return ST[i][2] + BH / 2; };
  var SAY = [
    'It clears every gate and becomes an accepted task. The successful authoring trajectory ' +
      'also joins the midtraining data: the flywheel.',
    'The seed itself fails the filter, so no candidate is ever generated from it. It is ' +
      'rejected outright, the only exit that does not loop back.',
    'LLM judges and polarity checks catch the problem early, while checking is still cheap. ' +
      'The candidate goes back to its seed with feedback.',
    'The expensive checks, an agentic verifier and a check for reward hacking, find a flaw. ' +
      'The candidate goes back to its seed with feedback.',
    'Every check passes, but the task gives too little learning signal for training. ' +
      'It goes back to its seed with feedback.'
  ];

  // a box with a coloured strip along its top edge, as in the paper's figure
  function box(g, x, y, w, h, strip, o) {
    o = o || {};
    if (o.fill) {
      g.appendChild(K.n('rect', { x: x, y: y, width: w, height: h, rx: 6, fill: o.fill }));
    } else {
      K.panel(g, x, y, w, h, { r: 6, stroke: o.stroke });
    }
    var r = 6;
    g.appendChild(K.n('path', { d: 'M' + x + ' ' + (y + r) + ' Q' + x + ' ' + y + ' ' + (x + r) + ' ' + y +
      ' H' + (x + w - r) + ' Q' + (x + w) + ' ' + y + ' ' + (x + w) + ' ' + (y + r) + ' V' + (y + 4) +
      ' H' + x + ' Z', fill: strip }));
  }

  function draw(f) {
    CH.clear(body);
    // f = 0 means it passes everything; otherwise f is the index of the failing gate
    var stop = f === 0 ? 5 : f;
    var lines = K.n('g', {});
    body.appendChild(lines);

    // passes, straight down
    for (var i = 0; i < 5; i++) {
      var on = i < stop;
      K.arrow(lines, 40, ST[i][2] + BH, 40, ST[i + 1][2] - 1,
        { color: on ? C.amber : CH.GREY2, w: on ? 2.2 : 1.3 });
    }
    K.mono(body, 54, 201, 'pass, then generate a candidate', { size: 12.5, color: stop > 1 ? C.ink2 : C.ink3 });

    // the seed filter's only exit is a rejection
    var rej = f === 1;
    K.arrow(lines, BW, mid(1), 338, mid(1), { color: rej ? C.red : CH.GREY2, w: rej ? 2.2 : 1.3 });
    K.mono(body, 302, mid(1) - 8, 'fail', { size: 12.5, color: rej ? C.ink : C.ink3 });
    var rg = K.n('g', { opacity: rej ? 1 : 0.32 });
    body.appendChild(rg);
    rg.appendChild(K.n('rect', { x: 342, y: mid(1) - 17, width: 132, height: 34, rx: 6, fill: C.red }));
    K.mono(rg, 408, mid(1) + 5, 'Rejected', { size: 13.5, color: '#fff', anchor: 'middle', weight: 600 });

    // every later gate loops back to the seed with feedback
    var back = f >= 2;
    var bc = back ? C.red : CH.GREY2, bw = back ? 2.2 : 1.3;
    [2, 3, 4].forEach(function (k) {
      var me = f === k;
      lines.appendChild(K.n('path', { d: 'M' + BW + ' ' + mid(k) + ' H' + RX, fill: 'none',
        stroke: me ? C.red : CH.GREY2, 'stroke-width': me ? 2.2 : 1.3, 'stroke-dasharray': '5 4' }));
      K.mono(body, 302, mid(k) - 8, 'fail', { size: 12.5, color: me ? C.ink : C.ink3 });
    });
    lines.appendChild(K.n('path', { d: 'M' + RX + ' ' + mid(back ? f : 4) + ' V' + mid(0) + ' H' + (BW + 8),
      fill: 'none', stroke: bc, 'stroke-width': bw, 'stroke-dasharray': '5 4' }));
    if (back) {
      lines.appendChild(K.n('path', { d: 'M' + RX + ' ' + mid(4) + ' V' + mid(f), fill: 'none',
        stroke: CH.GREY2, 'stroke-width': 1.3, 'stroke-dasharray': '5 4' }));
    }
    K.arrow(lines, BW + 9, mid(0), BW + 1, mid(0), { color: bc, w: bw, headSize: 7 });
    K.mono(body, 330, mid(0) - 10, 'back to the seed, with feedback', { size: 12.5,
      color: back ? C.ink : C.ink3, weight: back ? 600 : 400 });

    // the stages themselves
    ST.forEach(function (s, i) {
      var passed = i < stop || (f === 0), failing = i === f && f > 0, last = i === 5;
      var g = K.n('g', { opacity: i > stop ? 0.38 : 1 });
      body.appendChild(g);
      var strip = failing ? C.red : passed ? C.amber : CH.GREY2;
      if (last) {
        box(g, 0, s[2], BW, BH, f === 0 ? C.amber : CH.GREY2, { fill: f === 0 ? C.amber : C.ink });
        K.text(g, 16, s[2] + 28, s[0], { size: 15, weight: 600, color: '#fff' });
        return;
      }
      box(g, 0, s[2], BW, BH, strip, { stroke: failing ? C.red : null });
      if (s[1]) {
        K.text(g, 16, s[2] + 21, s[0], { size: 14.5, weight: 600, color: C.ink });
        K.mono(g, 16, s[2] + 37, s[1], { size: 12.5, color: C.ink3 });
      } else {
        K.text(g, 16, s[2] + 28, s[0], { size: 14.5, weight: 600, color: C.ink });
      }
    });

    // the flywheel: a successful authoring trajectory becomes midtraining data
    var fw = f === 0;
    K.arrow(lines, BW, mid(5), 330, mid(5), { color: fw ? C.amber : CH.GREY2, w: fw ? 2.2 : 1.3 });
    K.mono(body, 340, mid(5) + 5, 'trajectory joins midtraining data', { size: 13,
      color: fw ? C.ink : C.ink3, weight: fw ? 600 : 400 });

    K.callout(body, 0, 472, 640, SAY[f], { cols: 74 });
  }

  K.label(svg, 0, 7, 'where does it fail?');
  var sw = K.switcher(svg, 0, 19, ['nowhere', 'seed filter', 'cheap checks', 'expensive checks',
    'calibration'], draw, { tint: 'amber' });
  K.foot(svg, 'A schematic redraw of Figure 9, turned to run top to bottom. The midtraining arrow ' +
    'and the seed and calibration notes come from the text of §4.1. The paper gives no pass or ' +
    'failure rates.');
  sw.pick(3);
};
