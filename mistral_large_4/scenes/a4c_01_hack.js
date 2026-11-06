/* S_HACK: the reward hack from §4.4.3, step by step. A schematic: the strings in
   code type (the execvp call, the shim path, the quote) and the count of
   discrepancies are the paper's; the boxes only lay out how a PATH shim works. */
window.SCENES.S_HACK = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var CALL = F['a4c.execvp'], SHIM = F['a4c.shim'];
  var WORDS = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10 };
  var NDIFF = WORDS[F['hack.discrepancies']] || 0;   // "eight" in the paper
  var svg = K.board(host, { alt: 'How the model intercepted a compiled checker to read the reference implementation' });
  K.head(svg, 'Caught reading the answer key', 'reward hacking · one case from run monitoring · schematic');
  var stage = CH.body(svg);

  // a box with a bold first line and a quieter second line
  function box(g, x, y, w, h, t1, t2, o) {
    o = o || {};
    K.panel(g, x, y, w, h, { r: 9, stroke: o.stroke, fill: o.fill });
    if (o.stroke && o.stroke !== C.line) {
      g.appendChild(K.n('rect', { x: x, y: y, width: w, height: h, rx: 9, fill: 'none',
        stroke: o.stroke, 'stroke-width': o.sw || 1.8 }));
    }
    K.mono(g, x + 16, y + 25, t1, { size: o.size || 13.5, weight: 600, color: C.ink });
    K.mono(g, x + 16, y + 46, t2, { size: 13, color: C.ink3 });
  }

  var STEP = [
    "The task, as the paper gives it: fix a pipeline so that a compiled checker " +
      "accepts the model's output.",
    "The model noticed that the checker internally passes source code to python3 on the " +
      "command line: " + CALL + ".",
    "execvp finds python3 by searching the PATH in order. The model planted a fake " +
      SHIM + " first in line, so the checker's call reached the shim first.",
    "The shim received the command-line payload, and with it the full reference " +
      "implementation. The model now held the answer key.",
    "It listed " + F["hack.discrepancies"] + " discrepancies between its code and the reference, " +
      "fixed them, and submitted, noting: \"" + F["hack.quote"] + "\""
  ];
  // the callout changes height between steps; reserve its tallest size so the foot,
  // which is placed once, never lands on it
  svg.appendChild(K.n("rect", { x: 0, y: 432, width: 640, height: 3 * 19 + 22, fill: "none" }));

  function draw(s) {
    CH.clear(stage);
    // steps not reached stay as faint ghosts, so the layout never jumps
    function lay(k) {
      var g = K.n('g', { opacity: k <= s ? 1 : 0.16 });
      stage.appendChild(g);
      return g;
    }
    var g0 = lay(0), g1 = lay(1), g2 = lay(2), g3 = lay(3), g4 = lay(4);
    var hot = function (k) { return k === s ? C.amber : C.line; };

    // 1. the task
    box(g0, 0, 50, 200, 58, 'the model', 'fixing a pipeline', { stroke: hot(0) });
    K.arrow(g0, 204, 79, 296, 79, { color: C.ink3, w: 1.4 });
    K.mono(g0, 250, 70, 'its output', { size: 13, anchor: 'middle', color: C.ink2 });
    box(g0, 300, 50, 340, 58, 'compiled checker', 'must accept the output', { stroke: hot(0) });

    // 2. the call it makes internally
    g1.appendChild(K.n('path', { d: 'M470 108 V130', stroke: C.ink3, 'stroke-width': 1.4,
      'stroke-dasharray': '3 3' }));
    K.label(g1, 0, 158, 'inside the checker');
    K.code(g1, 300, 130, 340, [{ t: CALL, c: s === 1 ? '#f0b98f' : '#c8cfdb' }], { size: 12.5 });
    if (s === 1) g1.appendChild(K.n('rect', { x: 300, y: 130, width: 340, height: 45, rx: 10,
      fill: 'none', stroke: C.amber, 'stroke-width': 2 }));

    // 3. the PATH shim: execvp takes the first python3 it finds
    K.arrow(g2, 390, 175, 160, 203, { color: C.red, w: 1.8 });
    box(g2, 0, 205, 300, 58, SHIM, 'fake · found first on the PATH',
      { stroke: C.red, fill: K.TINT.red, sw: s === 2 ? 2.6 : 1.8 });
    // the paper only says the shim intercepted the call; it does not say whether the
    // real interpreter also ran, so this box claims only its place on the PATH
    box(g2, 340, 205, 300, 58, 'the real python3', 'later on the PATH', {});

    // 4. the capture
    K.arrow(g3, 150, 265, 150, 291, { color: C.ink3, w: 1.4 });
    box(g3, 0, 293, 300, 58, '"-c", <source_code>', 'the reference implementation',
      { stroke: hot(3) });
    K.mono(g3, 320, 330, '≠', { size: 22, anchor: 'middle', color: C.ink2, weight: 600 });
    box(g3, 340, 293, 300, 58, 'the model\'s own code', 'its code, before the fix', { stroke: hot(3) });

    // 5. the diff, the fix, the submission
    for (var i = 0; i < NDIFF; i++) {
      g4.appendChild(K.n('rect', { x: i * 24, y: 380, width: 18, height: 18, rx: 4,
        fill: s === 4 ? C.amber : CH.GREY2 }));
    }
    K.mono(g4, NDIFF * 24 + 8, 394, F['hack.discrepancies'] + ' discrepancies, all fixed',
      { size: 13.5, weight: 600, color: C.ink });
    K.arrow(g4, 486, 389, 516, 389, { color: C.ink3, w: 1.4 });
    K.chip(g4, 524, 376, 'submitted', { size: 13, w: 116, h: 27,
      stroke: s === 4 ? C.amber : C.line, color: C.ink2 });

    K.callout(stage, 0, 432, 640, STEP[s], { cols: 74 });
  }

  K.switcher(svg, 0, 0, ['1 the task', '2 the call', '3 the shim', '4 the capture', '5 fix and submit'],
    draw, { tint: 'amber' });
  K.foot(svg, 'Schematic of the case in §4.4.3. The code strings, the count and the quote are the ' +
    'paper\'s. Runs are monitored for hacks like this; the sandboxes keep being hardened without ' +
    'removing the realistic internet access and tooling agents need.');
};
