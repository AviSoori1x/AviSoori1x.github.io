/* S_SWE: §4.1.1, a merged pull request turned into a training task. Pick a test
   and it runs on the code before and after the reference change; the two results
   decide its job (fail then pass: verifier; pass both times: regression guard).
   Tests already run keep their results, so the suite fills in as you go. The pull
   request and the test names are illustrative, the rule and both stages are the
   paper's. */
window.SCENES.S_SWE = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH;
  var svg = K.board(host, { alt: 'Tests sorted into verifiers and regression guards by running ' +
    'them before and after the fix, then the two authoring stages' });
  K.head(svg, 'A merged PR becomes an exam', '§4.1.1 · schematic · two runs decide what a test is for');

  // [name, passes before the fix, passes after it]
  var T = [['test_retry_charges_once', false, true], ['test_refund_flow', true, true],
    ['test_timeout_then_retry', false, true], ['test_public_api_unchanged', true, true]];
  var JOB = {
    v: ['verifier', C.amber, 'It fails before the fix and passes after it, so any solution has ' +
      'to make it pass.'],
    g: ['regression guard', C.blue, 'It passes both before and after, so it catches a solution ' +
      'that breaks what already worked.']
  };
  var job = function (t) { return !t[1] && t[2] ? JOB.v : JOB.g; };
  var ran = {}, LW = 316, IX = [230, 282], RX = 340;

  // the pull request the task is cut from
  var mg = K.n('g', { fill: 'none', stroke: C.ink3, 'stroke-width': 1.6 });
  svg.appendChild(mg);
  mg.appendChild(K.n('circle', { cx: 5, cy: -2, r: 3 }));
  mg.appendChild(K.n('circle', { cx: 5, cy: 14, r: 3 }));
  mg.appendChild(K.n('circle', { cx: 16, cy: 6, r: 3 }));
  mg.appendChild(K.n('path', { d: 'M5 1 V11 M5 2 Q5 6 13 6' }));
  K.mono(svg, 28, 11, 'merged pull request: fix duplicate charges on a retried payment',
    { size: 13, color: C.ink2 });
  K.mono(svg, IX[0], 42, 'before', { size: 12.5, color: C.ink3, anchor: 'middle' });
  K.mono(svg, IX[1], 42, 'after', { size: 12.5, color: C.ink3, anchor: 'middle' });

  var marks = CH.body(svg), panel = CH.body(svg);
  function icon(p, x, y, state) {
    if (state == null) {
      p.appendChild(K.n('circle', { cx: x, cy: y, r: 9, fill: 'none', stroke: CH.GREY2,
        'stroke-width': 1.4, 'stroke-dasharray': '3 3' }));
      return;
    }
    p.appendChild(K.n('circle', { cx: x, cy: y, r: 9, fill: state ? C.ink2 : '#fff',
      stroke: state ? C.ink2 : C.ink3, 'stroke-width': 1.5 }));
    p.appendChild(K.n('path', { d: state ? 'M' + (x - 4) + ' ' + y + ' l3 3.5 l5.5 -6.5' :
      'M' + (x - 3.5) + ' ' + (y - 3.5) + ' l7 7 m0 -7 l-7 7', fill: 'none',
      stroke: state ? '#fff' : C.ink2, 'stroke-width': 1.8, 'stroke-linecap': 'round' }));
  }

  function draw(i) {
    ran[i] = true;
    CH.clear(marks); CH.clear(panel);
    T.forEach(function (t, k) {
      var y = 54 + k * 44 + 17, r = ran[k];
      icon(marks, IX[0], y, r ? t[1] : null);
      icon(marks, IX[1], y, r ? t[2] : null);
      if (r) marks.appendChild(K.n('circle', { cx: 302, cy: y, r: 5, fill: job(t)[1] }));
    });
    var t = T[i], j = job(t);
    [['before the fix', t[1]], ['after the fix', t[2]]].forEach(function (run, k) {
      var y = 54 + k * 46;
      K.panel(panel, RX, y, 300, 36, { r: 8 });
      icon(panel, RX + 20, y + 18, run[1]);
      K.mono(panel, RX + 40, y + 23, run[0] + ': ' + (run[1] ? 'passes' : 'fails'),
        { size: 13.5, color: C.ink, weight: 600 });
    });
    K.label(panel, RX, 164, 'so its job is');
    K.big(panel, RX, 198, j[0], { size: 28, color: j[1] });
    K.para(panel, RX, 228, j[2], 37, { size: 13, lh: 19 });
  }
  // neutral selection tint, so amber means only "verifier"
  var pk = CH.pick(svg, 0, 54, T.map(function (t) { return t[0]; }), draw,
    { w: LW, h: 34, gap: 10, tint: 'ink' });
  svg.appendChild(marks);   // keep the result icons above the buttons
  marks.setAttribute('pointer-events', 'none');
  // key for the job dots at the end of each row
  [[JOB.v, 0], [JOB.g, 116]].forEach(function (k) {
    svg.appendChild(K.n('circle', { cx: k[1] + 5, cy: 245.5, r: 5, fill: k[0][1] }));
    K.mono(svg, k[1] + 16, 250, k[0][0], { size: 12.5, color: C.ink2 });
  });

  // what happens around the sorting, in the paper's two stages
  var STAGE = [
    ['stage 1 · authoring agent', ['build it in a container', 'find a reproducible test run',
      'sort the tests by job', 'hide the fix, rewrite the spec'], 'tests cannot be validated: midtraining'],
    ['stage 2 · refinement agent', ['run solvers of varying strength', 'model, spec or test at fault?',
      'revise the spec and the tests', 'verify the revised task again'], 'later RL runs can feed back in']
  ];
  STAGE.forEach(function (s, c) {
    var x = c * 336, y0 = 330;
    K.label(svg, x, y0 - 22, s[0], { color: C.ink2 });
    svg.appendChild(K.n('path', { d: 'M' + (x + 5) + ' ' + y0 + ' V' + (y0 + 84), stroke: CH.GREY2,
      'stroke-width': 1.3 }));
    s[1].forEach(function (st, k) {
      var y = y0 + k * 28, hot = c === 0 && k === 2;
      svg.appendChild(K.n('circle', { cx: x + 5, cy: y, r: hot ? 5 : 4, fill: hot ? C.amber : '#fff',
        stroke: hot ? C.amber : CH.GREY, 'stroke-width': 1.5 }));
      K.mono(svg, x + 20, y + 4.5, st, { size: 13, color: hot ? C.ink : C.ink2, weight: hot ? 600 : 400 });
    });
    K.mono(svg, x, y0 + 124, s[2], { size: 12.5, color: C.ink3 });
  });
  K.arrow(svg, 300, 372, 326, 372, { color: C.ink3 });

  K.foot(svg, 'Schematic: the pull request and the test names are illustrative. The rule for ' +
    'each job and the two stages follow §4.1.1.');
  pk.pick(0);
};
