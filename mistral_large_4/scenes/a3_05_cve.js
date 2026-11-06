/* S_CVE: §4.1.3, vulnerability patching. Top: how a public CVE becomes an
   environment, and the rule a test harness must pass to be kept. Middle: the
   reward rule as a 2 by 2 grid you can click: exploit neutralized or not, against
   existing tests passing or not; only one cell earns positive reward. Bottom: the
   size of the pool (roughly 600 weakness classes is a fact; "tens of thousands"
   of CVEs is the paper's wording, not a number). */
window.SCENES.S_CVE = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var svg = K.board(host, { alt: 'From a public CVE to a containerised environment, and the reward rule: ' +
    'positive only if the exploit is neutralized and existing tests still pass' });
  K.head(svg, 'Kill the exploit, keep the tests', '§4.1.3 · schematic · one outcome of four earns positive reward');

  // how an environment is built
  var CHAIN = [['public CVE', 'a disclosed flaw'], ['upstream fix', 'the reference patch'],
    ['vulnerable code', 'in a container']];
  CHAIN.forEach(function (c, i) {
    var x = i * 222;
    K.panel(svg, x, 0, 196, 46, { r: 8 });
    K.mono(svg, x + 14, 19, c[0], { size: 13.5, color: C.ink, weight: 600 });
    K.mono(svg, x + 14, 37, c[1], { size: 12.5, color: C.ink3 });
    if (i < 2) K.arrow(svg, x + 199, 23, x + 219, 23, { color: C.ink3 });
  });
  K.arrow(svg, 320, 48, 320, 64, { color: C.ink3 });
  K.panel(svg, 0, 66, 640, 50, { r: 8 });
  svg.appendChild(K.n('rect', { x: 0, y: 66, width: 5, height: 50, rx: 2.5, fill: C.ink2 }));
  K.mono(svg, 18, 86, 'test harness: adversarial exploits plus standard inputs', { size: 13.5,
    color: C.ink, weight: 600 });
  K.mono(svg, 18, 106, 'kept only if it fires on the unpatched code and passes after the reference fix',
    { size: 12.5, color: C.ink2 });

  // what the model is given
  K.label(svg, 0, 146, 'the model gets the vulnerable code and');
  var c1 = K.chip(svg, 0, 156, 'a named flaw to fix', { color: C.ink, stroke: C.ink3 });
  K.mono(svg, c1._w + 12, 173, 'or', { size: 12.5, color: C.ink3 });
  K.chip(svg, c1._w + 40, 156, 'only a weakness category', { color: C.ink, stroke: C.ink3 });

  // the reward rule
  var COLS = ['existing tests still pass', 'existing tests break'];
  var ROWS = ['exploit neutralized', 'exploit still works'];
  var SAY = [
    ['The exploit is completely neutralized and every test that passed before still passes. ' +
      'This is the only outcome with positive reward.',
     'The exploit no longer fires, but tests that used to pass now fail: the patch broke the ' +
      'code it was meant to protect. No positive reward.'],
    ['The existing tests still pass, but the exploit still works. The patch must neutralize ' +
      'it completely. No positive reward.',
     'The exploit still works and the patch also broke existing tests: it fails on both counts. ' +
      'No positive reward.']
  ];
  var GX = 196, CW = 214, CHT = 58, GY = 222;
  COLS.forEach(function (s, j) {
    K.mono(svg, GX + j * (CW + 8) + CW / 2, GY - 12, s, { size: 13, color: C.ink2, anchor: 'middle' });
  });
  ROWS.forEach(function (s, r) {
    K.mono(svg, 0, GY + r * (CHT + 8) + CHT / 2 + 5, s, { size: 13, color: C.ink2 });
  });
  var cells = [], say = CH.body(svg);
  function draw(r, j) {
    cells.forEach(function (c) {
      var on = c.r === r && c.j === j, win = c.r === 0 && c.j === 0;
      c.rect.setAttribute('fill', win ? (on ? C.amber : K.TINT.amber) : (on ? K.TINT.ink : '#fff'));
      c.rect.setAttribute('stroke', win ? C.amber : (on ? C.ink2 : C.line));
      c.rect.setAttribute('stroke-width', on ? 2.4 : 1.2);
      c.t.setAttribute('fill', win ? (on ? '#fff' : C.amber) : (on ? C.ink : C.ink3));
    });
    CH.clear(say);
    K.callout(say, 0, GY + 2 * CHT + 28, 640, SAY[r][j], { cols: 74 });
  }
  ROWS.forEach(function (rs, r) {
    COLS.forEach(function (cs, j) {
      var x = GX + j * (CW + 8), y = GY + r * (CHT + 8), win = r === 0 && j === 0;
      var g = K.n('g', { tabindex: 0, role: 'button', 'aria-label': rs + ', ' + cs,
        style: 'cursor:pointer;outline:none' });
      var rect = K.n('rect', { x: x, y: y, width: CW, height: CHT, rx: 9 });
      g.appendChild(rect);
      // the paper says only when reward is positive, not that the rest score zero
      var t = K.mono(g, x + CW / 2, y + CHT / 2 + 6, win ? 'positive reward' : 'no positive reward',
        { size: win ? 16 : 14, anchor: 'middle', weight: 600 });
      var go = function () { draw(r, j); };
      g.addEventListener('click', go);
      g.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); }
      });
      svg.appendChild(g);
      cells.push({ r: r, j: j, rect: rect, t: t });
    });
  });

  // the size of the pool
  // the paper's own hedge sits right above the numeral, so 600 never reads as exact
  var PY = GY + 2 * CHT + 8 + 114;
  K.label(svg, 0, PY - 4, 'roughly', { color: C.ink2 });
  K.big(svg, 0, PY + 50, String(F['cve.classes']), { size: 46, color: C.ink });
  K.mono(svg, 124, PY + 28, 'weakness classes, spanning', { size: 13, color: C.ink2 });
  K.mono(svg, 124, PY + 48, 'tens of thousands of CVEs in major languages', { size: 13, color: C.ink,
    weight: 600 });

  K.foot(svg, 'A schematic of §4.1.3. The paper gives the pool as tens of thousands of CVEs; ' +
    'it reports no reward or solve rates.');
  draw(0, 1);
};
