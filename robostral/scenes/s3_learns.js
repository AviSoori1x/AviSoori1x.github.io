/* How it learns. Simulation, the prefix-tree mask, and the RL pass. */
window.SCENES = window.SCENES || {};

(function () {
  var K = window.KIT, C = K.C, T = K.TINT, K3 = window.K3;

  /* ---- 13 the simulated corpus ---- */
  window.SCENES.S_SIM = function (root, api) {
    var f = api.RN.facts;
    K3.head(root, 'All of it in simulation',
      f.trajectories_m + ' million trajectories across ' + f.scenes_k + ' thousand scenes');

    var handle = K3.mount(root, { aspect: '16 / 9' }, function (ctx) {
      var THREE = ctx.THREE;
      var w = ctx.world(THREE, 5);
      ctx.scene.add(w.group);

      // sample routes across the navigable area, seeded so the figure is stable
      var rand = ctx.rng(99), routes = [], PAL = [ctx.C.blue, ctx.C.teal, ctx.C.purple, ctx.C.amber];
      function free() {
        return new THREE.Vector3((rand() - 0.5) * 22, 0.06, (rand() - 0.5) * 15);
      }
      for (var i = 0; i < 34; i++) {
        var a = free(), b = free();
        if (a.distanceTo(b) < 9) { i--; continue; }
        var mid = new THREE.Vector3((a.x + b.x) / 2 + (rand() - 0.5) * 5, 0.06,
          (a.z + b.z) / 2 + (rand() - 0.5) * 4);
        var m = ctx.route(THREE, [a, mid, b], PAL[i % PAL.length]);
        m.material.transparent = true;
        m.material.opacity = 0;
        ctx.scene.add(m);
        routes.push(m);
      }

      var cam = ctx.thirdPerson(THREE, 38, [1, 2]);
      return {
        camera: cam,
        frame: function (t, c) {
          var a = t * 0.05;
          cam.position.set(Math.sin(a) * 5, 30, Math.cos(a) * 5 + 2);
          cam.lookAt(0, 0, 0);
          cam.updateMatrixWorld();
          routes.forEach(function (m, i) {
            var due = (t * 1.6 - i * 0.28) % 14;
            m.material.opacity = due > 0 && due < 9 ? Math.min(0.85, due * 1.6) * (due > 7 ? (9 - due) / 2 : 1) : 0;
          });
          c.renderer.render(c.scene, cam);
        }
      };
    });

    K3.note(root, 'Offices, homes, commercial spaces and outdoors, varied on layout complexity, ' +
      'object density, lighting and architectural style. Collecting this on real hardware is not ' +
      'a budget problem, it is prohibitively costly, which is the constraint the recipe is built ' +
      'to remove.', 'blue');
    return handle;
  };

  /* ---- 14 farthest point sampling ---- */
  window.SCENES.S_FPS = function (root, api) {
    var s = K.board(root, { alt: 'Farthest point sampling spreading start and goal positions out.' });
    K.head(s, 'Spread the endpoints on purpose',
      'each new pick is pushed away from everything already chosen');

    var X = 40, Y = 30, Wd = 560, Hd = 330;
    K.panel(s, X, Y, Wd, Hd, { fill: 'rgba(255,255,255,.92)' });
    K.label(s, X, Y - 12, 'navigable area');

    // a fixed cloud of candidate positions, chosen once
    var seed = 20250831, pts = [];
    function rnd() { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }
    for (var i = 0; i < 260; i++) {
      var px = X + 18 + rnd() * (Wd - 36), py = Y + 18 + rnd() * (Hd - 36);
      // carve a corridor so the cloud reads as a floor plan rather than a blob
      if (Math.abs(py - (Y + Hd * 0.52)) < 16 && rnd() < 0.55) continue;
      pts.push([px, py]);
    }
    var dots = pts.map(function (p) {
      var c = K.n('circle', { cx: p[0], cy: p[1], r: 2.1, fill: 'rgba(31,37,48,.20)' });
      s.appendChild(c); return c;
    });

    var picked = [], marks = K.n('g', {}); s.appendChild(marks);
    var links = K.n('g', {}); s.appendChild(links);
    K.label(s, X, Y + Hd + 26, 'picked so far');
    var counter = K.mono(s, X, Y + Hd + 60, '0', { size: 26, color: C.amber, weight: 700 });

    function step() {
      if (!picked.length) { picked.push(0); return; }
      var best = -1, bd = -1;
      for (var i = 0; i < pts.length; i++) {
        if (picked.indexOf(i) >= 0) continue;
        var d = Infinity;
        for (var j = 0; j < picked.length; j++) {
          var dx = pts[i][0] - pts[picked[j]][0], dy = pts[i][1] - pts[picked[j]][1];
          d = Math.min(d, dx * dx + dy * dy);
        }
        if (d > bd) { bd = d; best = i; }
      }
      if (best >= 0) picked.push(best);
    }

    function render() {
      marks.innerHTML = ''; links.innerHTML = '';
      picked.forEach(function (i, k) {
        dots[i].setAttribute('r', 0);
        marks.appendChild(K.n('circle', { cx: pts[i][0], cy: pts[i][1], r: 6.5,
          fill: k % 2 ? C.teal : C.amber }));
        marks.appendChild(K.n('circle', { cx: pts[i][0], cy: pts[i][1], r: 12, fill: 'none',
          stroke: k % 2 ? C.teal : C.amber, 'stroke-opacity': .35 }));
        if (k % 2 === 1) {
          var a = pts[picked[k - 1]], b = pts[i];
          links.appendChild(K.n('path', {
            d: 'M' + a[0] + ' ' + a[1] + ' Q' + ((a[0] + b[0]) / 2 + 26) + ' ' +
               ((a[1] + b[1]) / 2 - 22) + ' ' + b[0] + ' ' + b[1],
            fill: 'none', stroke: C.blue, 'stroke-width': 1.6, 'stroke-opacity': .55 }));
        }
      });
      counter.textContent = String(picked.length);
    }

    K.callout(s, 0, 448, 640,
      'The paper uses farthest point sampling to ensure diverse start and goal positions, giving ' +
      'trajectories of various lengths, sometimes spanning multiple floors.', { cols: 66 });

    var last = -1;
    return {
      tick: function (t) {
        var n = Math.floor((t * 1.4) % 17);
        if (n === last) return;
        if (n < last) { picked = []; dots.forEach(function (d) { d.setAttribute('r', 2.1); }); }
        while (picked.length < n) step();
        last = n;
        render();
      }
    };
  };

  /* ---- 15 the quadratic bill ---- */
  window.SCENES.S_NAIVE = function (root, api) {
    var s = K.board(root, { alt: 'Per-timestep training re-encodes every frame.' });
    K.head(s, 'The obvious way pays twice for everything',
      'one sample per timestep, each holding the whole history');

    K.label(s, 0, 14, 'per-timestep samples, an episode of 6 steps');
    var y0 = 30, cell = 18, gap = 3;
    for (var t = 1; t <= 6; t++) {
      var y = y0 + (t - 1) * 30;
      K.mono(s, 0, y + 13, 't=' + t, { size: 11, color: C.ink3 });
      s.appendChild(K.n('rect', { x: 36, y: y, width: cell, height: cell, rx: 3,
        fill: C.ink, opacity: .8 }));
      for (var k = 0; k < t; k++) {
        s.appendChild(K.n('rect', { x: 36 + (k + 1) * (cell + gap), y: y,
          width: cell, height: cell, rx: 3, fill: C.blue,
          opacity: k === t - 1 ? .95 : .30 }));
      }
      K.mono(s, 36 + (t + 1) * (cell + gap) + 10, y + 13, t + ' frame' + (t > 1 ? 's' : ''),
        { size: 10.5, color: C.ink3 });
    }
    K.mono(s, 36, y0 + 6 * 30 + 12, 'solid = encoded for the first time, faded = encoded again',
      { size: 11, color: C.ink3 });

    var totalG = K.n('g', {}); s.appendChild(totalG);
    K.label(s, 0, 268, 'frame encodings for an episode of length T');
    K.switcher(s, 0, 282, ['T = 8', 'T = 20', 'T = 43'], function (i) {
      draw([8, 20, 43][i]);
    }, { tint: 'red' });

    function draw(Tn) {
      totalG.innerHTML = '';
      var naive = Tn * (Tn + 1) / 2, packed = Tn;
      var mx = naive;
      [['per timestep', naive, C.red], ['one sequence', packed, C.teal]].forEach(function (r, i) {
        var y = 330 + i * 60;
        K.mono(totalG, 0, y + 4, r[0], { size: 12.5, color: r[2] });
        K.bar(totalG, 0, y + 14, 470, 16, r[1] / mx, { color: r[2] });
        K.mono(totalG, 486, y + 27, String(r[1]), { size: 15, color: r[2], weight: 700 });
      });
      K.mono(totalG, 0, 466, 'ratio ' + (naive / packed).toFixed(1) + '× at T = ' + Tn,
        { size: 13, color: C.ink2 });
    }
    draw(20);

    K.foot(s, 'Counted in frame encodings, which is the term that grows. The paper reports the ' +
      'measured saving on its own data as 22 times fewer training tokens.');
  };

  /* ---- 16 pack the episode ---- */
  window.SCENES.S_PACK = function (root, api) {
    var s = K.board(root, { alt: 'The whole episode as one flat sequence.' });
    K.head(s, 'Encode the episode once', 'instruction, then observations interleaved with actions');

    K.label(s, 0, 14, 'one training sequence');
    var x = 0, y = 30;
    function blk(w, lab, col, tint) {
      s.appendChild(K.n('rect', { x: x, y: y, width: w, height: 46, rx: 6,
        fill: tint, stroke: col }));
      K.mono(s, x + w / 2, y + 29, lab, { size: 12.5, color: col, anchor: 'middle' });
      x += w + 6;
    }
    blk(64, 'I', C.purple, T.purple);
    for (var i = 0; i < 5; i++) {
      blk(72, 'O' + i, C.blue, T.blue);
      blk(40, 'a' + i, C.amber, T.amber);
    }

    K.mono(s, 0, 100, 'every frame encoded exactly once, all losses from a single forward pass',
      { size: 12.5, color: C.ink3 });

    K.label(s, 0, 142, 'what that changes');
    [['token cost', 'O(T²) → O(T)', C.teal],
     ['prediction targets', 'unchanged, none discarded', C.teal],
     ['redundancy removed', 'the repeated prefixes, and only those', C.teal]
    ].forEach(function (r, j) {
      var yy = 156 + j * 56;
      K.panel(s, 0, yy, 640, 46, { flat: true, fill: T.teal, stroke: C.teal });
      K.mono(s, 18, yy + 28, r[0], { size: 12, color: C.ink3 });
      K.mono(s, 210, yy + 28, r[1], { size: 13, color: r[2], weight: 500 });
    });

    K.callout(s, 0, 340, 640,
      'The idea comes from amortising one shared context across many independent questions. ' +
      'Here it is pushed onto the nested structure of a trajectory, where each step\'s history ' +
      'is a prefix of the next one\'s.', { cols: 66, color: C.blue, tint: T.blue });
  };

  /* ---- 17 the leak ---- */
  window.SCENES.S_LEAK = function (root, api) {
    var s = K.board(root, { alt: 'Causal attention lets the model read earlier ground-truth actions.' });
    K.head(s, 'Flattening it opens a hole', 'the answers are now sitting in the context');

    K.label(s, 0, 14, 'what an ordinary causal mask allows at step 3');
    var x = 0, y = 28, boxes = [];
    function blk(w, lab, col, tint, leak) {
      s.appendChild(K.n('rect', { x: x, y: y, width: w, height: 44, rx: 6,
        fill: leak ? T.red : tint, stroke: leak ? C.red : col,
        'stroke-width': leak ? 1.8 : 1 }));
      K.mono(s, x + w / 2, y + 27, lab, { size: 12.5, color: leak ? C.red : col, anchor: 'middle' });
      x += w + 6;
    }
    blk(60, 'I', C.purple, T.purple);
    blk(70, 'O0', C.blue, T.blue);
    blk(40, 'a0', C.amber, T.amber, true);
    blk(70, 'O1', C.blue, T.blue);
    blk(40, 'a1', C.amber, T.amber, true);
    blk(70, 'O2', C.blue, T.blue);
    blk(40, 'a2', C.ink3, T.ink);
    K.mono(s, 0, 92, 'the two red blocks are ground truth the model will not have on a robot',
      { size: 12, color: C.red });

    K.label(s, 0, 134, 'why it matters more here than usual');
    K.panel(s, 0, 148, 640, 118, { stroke: C.red, fill: T.red });
    K.para(s, 18, 176, 'In pointing mode the target is the furthest visible waypoint, so previous ' +
      'ground-truth actions are often very informative of the next one. A model allowed to see ' +
      'them would lean on information it does not have at deployment.', 62,
      { size: 13, color: C.red });

    K.panel(s, 0, 286, 640, 92, { flat: true, fill: T.ink, stroke: C.line });
    K.mono(s, 18, 314, 'available in training', { size: 14, color: C.ink2, weight: 500 });
    K.mono(s, 18, 340, 'absent at deployment', { size: 14, color: C.red, weight: 500 });
    K.mono(s, 210, 314, 'the ground-truth actions sit in the context', { size: 12, color: C.ink3 });
    K.mono(s, 210, 340, 'and a robot never has them', { size: 12, color: C.ink3 });

    K.callout(s, 0, 400, 640,
      'The fix is not to change the data. It is to change what the model is allowed to look at.',
      { cols: 66 });
  };

  /* ---- 18 the prefix-tree attention mask ---- */
  window.SCENES.S_TREE = function (root, api) {
    var s = K.board(root, { alt: 'The prefix-tree attention mask compared with a causal one.' });
    K.head(s, 'Close it with the mask', 'trunk is shared, branches cannot see each other');

    // token layout: I, then per step an observation block and an action block
    var STEPS = 5, IT = 2, OT = 3, AT = 1;
    var toks = [];
    for (var q = 0; q < IT; q++) toks.push({ kind: 'I', branch: -1 });
    for (var t = 0; t < STEPS; t++) {
      for (var o = 0; o < OT; o++) toks.push({ kind: 'O', branch: -1, step: t });
      for (var a = 0; a < AT; a++) toks.push({ kind: 'a', branch: t, step: t });
    }
    var N = toks.length, cell = 15, GX = 130, GY = 34;

    K.label(s, GX, GY - 28, 'key  →');
    K.label(s, 0, GY + 8, 'query');
    K.label(s, 0, GY + 24, '↓');

    var grid = K.n('g', {}); s.appendChild(grid);
    var legendG = K.n('g', {}); s.appendChild(legendG);

    function allowedCausal(i, j) { return j < i; }
    function allowedTree(i, j) {
      if (j >= i) return false;
      if (toks[j].branch === -1) return true;              // trunk: instruction and observations
      return toks[j].branch === toks[i].branch;            // same branch only
    }

    function draw(mode) {
      grid.innerHTML = '';
      var leaks = 0;
      for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
          var cz = allowedCausal(i, j), tz = allowedTree(i, j);
          var on = mode === 0 ? cz : tz;
          var leak = cz && !tz;
          if (leak && mode === 0) leaks++;
          var fill = 'rgba(31,37,48,.045)';
          if (on) {
            fill = toks[j].kind === 'a'
              ? (mode === 0 && leak ? 'rgba(163,45,45,.72)' : 'rgba(180,85,26,.72)')
              : (toks[j].kind === 'I' ? 'rgba(107,63,168,.55)' : 'rgba(28,110,140,.55)');
          }
          grid.appendChild(K.n('rect', {
            x: GX + j * cell, y: GY + i * cell, width: cell - 1.4, height: cell - 1.4,
            rx: 2, fill: fill }));
        }
      }
      // axis ticks naming the blocks
      var px = GX;
      function tick(n, lab, col) {
        K.mono(grid, px + n * cell / 2, GY - 6, lab,
          { size: 9.5, color: col, anchor: 'middle' });
        px += n * cell;
      }
      tick(IT, 'I', C.purple);
      for (var t2 = 0; t2 < STEPS; t2++) { tick(OT, 'O' + t2, C.blue); tick(AT, 'a' + t2, C.amber); }

      legendG.innerHTML = '';
      var msg = mode === 0
        ? leaks + ' cells let a step read another step\'s ground-truth action'
        : 'no cell reads another step\'s action, and every step still sees every frame up to its own';
      K.panel(legendG, 0, BASE + 46, 640, 60, { flat: true,
        fill: mode === 0 ? T.red : T.teal, stroke: mode === 0 ? C.red : C.teal });
      K.mono(legendG, 18, BASE + 82, msg,
        { size: 13, color: mode === 0 ? C.red : C.teal });
    }

    // everything below the grid hangs off one baseline, so the switcher cannot
    // land back on top of the matrix when the token count changes
    var BASE = GY + N * cell + 10;
    K.switcher(s, 0, BASE, ['causal', 'prefix tree'],
      function (i) { draw(i); }, { tint: 'teal' });
    draw(0);

    K.callout(s, 0, BASE + 122, 640,
      'The rule: a token may attend to an earlier token only if that token is in the trunk, or ' +
      'in the same branch. Observations are trunk. Actions are branch.', { cols: 66 });
    K.foot(s, 'Drawn with ' + STEPS + ' steps and small blocks so the structure is visible. The ' +
      'paper states this gives provably the same training signal as the per-timestep samples.');
  };

  /* ---- 19 what it bought ---- */
  window.SCENES.S_22X = function (root, api) {
    var f = api.RN.facts;
    var s = K.board(root, { alt: 'Token saving and the training time it converts to.' });
    K.head(s, 'Twenty two times fewer tokens', 'and no prediction target given up');

    K.big(s, 0, 92, f.token_saving_x + '×', { size: 96, color: C.teal });
    K.mono(s, 0, 132, 'fewer training tokens, same targets', { size: 13.5, color: C.ink3 });

    K.panel(s, 0, 156, 640, 96, { stroke: C.teal, fill: T.teal });
    K.mono(s, 18, 188, 'months', { size: 22, color: C.ink3, weight: 700 });
    K.arrow(s, 118, 182, 188, 182, { color: C.teal, w: 2, headSize: 8 });
    K.mono(s, 202, 188, 'days', { size: 22, color: C.teal, weight: 700 });
    K.mono(s, 18, 220, 'the same run, after the redundant prefixes stop being re-encoded',
      { size: 12, color: C.ink3 });

    K.label(s, 0, 278, 'where it matters most');
    K.panel(s, 0, 292, 310, 130, { flat: true, fill: T.ink, stroke: C.line });
    K.mono(s, 18, 320, 'R2R-CE', { size: 15, color: C.ink, weight: 500 });
    K.para(s, 18, 344, 'shorter instructions, shorter routes', 34, { size: 12, color: C.ink3 });
    K.panel(s, 330, 292, 310, 130, { fill: T.amber, stroke: C.amber });
    K.mono(s, 348, 320, 'RxR-CE', { size: 15, color: C.amber, weight: 500 });
    K.para(s, 348, 344, 'considerably longer on both, so the quadratic term bites hardest here',
      34, { size: 12, color: C.amber });

    K.callout(s, 0, 442, 640,
      'The benchmark with the longest episodes is the one this rescues, which is why the paper ' +
      'calls the technique vital for RxR-CE specifically.', { cols: 66 });
  };

  /* ---- 20 where imitation runs out ---- */
  window.SCENES.S_EXPOSURE = function (root, api) {
    var s = K.board(root, { alt: 'Shortest-path demonstrations contain no recoveries.' });
    K.head(s, 'A shortest path never contains a mistake',
      'so a model trained only on them has never seen one');

    K.label(s, 0, 14, 'what the demonstrations look like');
    K.panel(s, 0, 26, 640, 132, { flat: true, fill: 'rgba(255,255,255,.9)' });
    s.appendChild(K.n('path', { d: 'M40 130 C 160 120, 300 70, 590 58', fill: 'none',
      stroke: C.teal, 'stroke-width': 3 }));
    s.appendChild(K.n('circle', { cx: 40, cy: 130, r: 6, fill: C.teal }));
    s.appendChild(K.n('circle', { cx: 590, cy: 58, r: 6, fill: C.teal }));
    K.mono(s, 40, 152, 'expert, optimal, never wrong', { size: 12, color: C.teal });

    K.label(s, 0, 190, 'what happens on a robot');
    K.panel(s, 0, 202, 640, 168, { stroke: C.red, fill: T.red });
    s.appendChild(K.n('path', { d: 'M40 330 C 130 320, 200 290, 250 270', fill: 'none',
      stroke: C.ink3, 'stroke-width': 3 }));
    s.appendChild(K.n('path', { d: 'M250 270 C 320 250, 360 300, 420 330 S 520 350, 580 300',
      fill: 'none', stroke: C.red, 'stroke-width': 3, 'stroke-dasharray': '7 5' }));
    s.appendChild(K.n('circle', { cx: 250, cy: 270, r: 6, fill: C.red }));
    K.mono(s, 258, 258, 'a small error', { size: 12, color: C.red });
    K.mono(s, 400, 368, 'and now it is somewhere the data never went', { size: 12, color: C.red });

    [['exposure bias', 'trained on its own predictions never, only on the expert'],
     ['covariate shift', 'the states it visits are not the states it learned from']
    ].forEach(function (r, i) {
      var y = 392 + i * 46;
      K.mono(s, 0, y + 16, r[0], { size: 13, color: C.ink, weight: 500 });
      K.mono(s, 170, y + 16, r[1], { size: 12, color: C.ink3 });
    });

    K.callout(s, 0, 484, 640,
      'The shortest-path demonstrations do not show information-seeking or recovery, which is what ' +
      'leaves the policy exposed in the states it reaches after an error.',
      { cols: 66 });
  };

  /* ---- 21 the reward ---- */
  window.SCENES.S_CISPO = function (root, api) {
    var f = api.RN.facts;
    var s = K.board(root, { alt: 'The clipped distance reward.' });
    K.head(s, 'A reward that stops caring once you are close',
      'r = −max(' + f.reward_clip_m + ', distance to goal)');

    var X = 60, Y = 40, Wd = 520, Hd = 240, DMAX = 10;
    K.panel(s, X, Y, Wd, Hd, { fill: 'rgba(255,255,255,.94)' });
    function px(d) { return X + (d / DMAX) * Wd; }
    function py(r) { return Y + Hd - ((r + DMAX) / DMAX) * Hd; }

    var d = 'M';
    for (var i = 0; i <= 100; i++) {
      var dd = i / 100 * DMAX, r = -Math.max(f.reward_clip_m, dd);
      d += (i ? ' L' : '') + px(dd) + ' ' + py(r);
    }
    s.appendChild(K.n('path', { d: d, fill: 'none', stroke: C.blue, 'stroke-width': 3 }));
    s.appendChild(K.n('rect', { x: X, y: Y, width: px(f.reward_clip_m) - X, height: Hd,
      fill: T.teal }));
    K.mono(s, X + 10, Y + 26, 'flat inside ' + f.reward_clip_m + ' m', { size: 12, color: C.teal });
    K.mono(s, X + 10, Y + 44, 'nothing left to gain by shuffling', { size: 11, color: C.teal });

    K.mono(s, X, Y + Hd + 20, '0 m', { size: 11, color: C.ink3 });
    K.mono(s, X + Wd - 30, Y + Hd + 20, DMAX + ' m', { size: 11, color: C.ink3 });
    K.label(s, X, Y + Hd + 42, 'geodesic distance from the goal when the episode ends');

    var dot = K.n('circle', { cx: px(6), cy: py(-6), r: 7, fill: C.red });
    s.appendChild(dot);
    var lab = K.mono(s, 0, 0, '', { size: 13, color: C.red, weight: 500 });

    K.label(s, 0, 340, 'what the clip is for');
    K.panel(s, 0, 352, 640, 88, { stroke: C.blue, fill: T.blue });
    K.para(s, 18, 380, 'Without it, an agent can keep collecting reward by manoeuvring near the ' +
      'target forever. With it, once you are close the only thing left to gain is ending the ' +
      'episode, so the model learns to emit STOP.', 62, { size: 13, color: C.blue });

    K.foot(s, 'CISPO with group-relative advantage estimation, on top of the supervised model. ' +
      'Distance is geodesic, from a path finder, not straight line.');

    return {
      tick: function (t) {
        var dd = 0.4 + (1 + Math.sin(t * 0.6)) / 2 * 9.2;
        var r = -Math.max(f.reward_clip_m, dd);
        dot.setAttribute('cx', px(dd)); dot.setAttribute('cy', py(r));
        lab.setAttribute('x', Math.min(px(dd) + 14, X + Wd - 120));
        lab.setAttribute('y', py(r) - 12);
        lab.textContent = dd.toFixed(1) + ' m  →  r = ' + r.toFixed(1);
      }
    };
  };

  /* ---- 22 the hard subset ---- */
  window.SCENES.S_HARD = function (root, api) {
    var f = api.RN.facts;
    var s = K.board(root, { alt: 'Curating a hard subset by rolling out the supervised policy.' });
    K.head(s, 'Only the episodes it already fails',
      f.hard_tasks_k + ' thousand tasks the supervised policy could not reliably reach the goal on');

    K.label(s, 0, 14, 'curation');
    var STEPS = [
      ['roll out the supervised policy over the task pool', C.ink3],
      ['drop everything it already solves reliably', C.ink3],
      ['keep the tangled layouts, ambiguous instructions, long horizons', C.amber]
    ];
    STEPS.forEach(function (r, i) {
      var y = 28 + i * 58;
      K.panel(s, 0, y, 640, 48, { flat: i < 2, fill: i === 2 ? T.amber : 'rgba(255,255,255,.9)',
        stroke: i === 2 ? C.amber : C.line });
      K.mono(s, 18, y + 20, String(i + 1), { size: 11, color: C.ink3 });
      K.mono(s, 44, y + 29, r[0], { size: 13, color: r[1] });
    });

    K.big(s, 0, 268, f.hard_tasks_k + 'k', { size: 62, color: C.amber });
    K.mono(s, 0, 300, 'tasks kept for reinforcement learning', { size: 12.5, color: C.ink3 });

    K.label(s, 0, 330, 'then packed by scene, not shuffled');
    var gx = 0;
    [C.blue, C.blue, C.blue, C.teal, C.teal, C.purple, C.purple, C.purple, C.purple, C.amber]
      .forEach(function (col, i) {
        s.appendChild(K.n('rect', { x: gx, y: 344, width: 56, height: 34, rx: 5,
          fill: col, opacity: .85 }));
        gx += 62;
      });
    K.mono(s, 0, 398, 'consecutive rollouts share a building, which is an implicit visual curriculum',
      { size: 12, color: C.ink3 });

    K.callout(s, 0, 424, 640,
      'Empirically the scene-contiguous ordering beats random shuffling, and the curation focuses ' +
      'RL compute on the cases where the policy most needs improvement.', { cols: 66 });
  };

  /* ---- 23 three workloads ---- */
  window.SCENES.S_SYSTEMS = function (root, api) {
    var s = K.board(root, { alt: 'Three GPU-bound workloads sharing one cluster.' });
    K.head(s, 'A systems engineering problem',
      'three GPU-bound workloads that have to run at once without stalling');

    var W = [
      { n: 'simulator', d: 'physics and rendering for hundreds of parallel environments',
        c: C.purple, note: 'the awkward one: rendering competes with learning for the same memory' },
      { n: 'action generator', d: 'distributed inference producing actions autoregressively',
        c: C.blue, note: 'must not stall waiting for frames' },
      { n: 'trainer', d: 'distributed weight updates across the training ranks',
        c: C.teal, note: 'must not stall waiting for rollouts' }
    ];
    W.forEach(function (w, i) {
      var y = 16 + i * 128;
      K.panel(s, 0, y, 640, 104, { stroke: w.c, fill: i === 0 ? T.purple : (i === 1 ? T.blue : T.teal) });
      K.mono(s, 20, y + 32, w.n, { size: 15, color: w.c, weight: 500 });
      K.para(s, 20, y + 56, w.d, 60, { size: 12.5, color: C.ink2 });
      K.mono(s, 20, y + 90, w.note, { size: 11.5, color: C.ink3 });
    });

    // the loop arrows, drawn outside the stack so they read as a cycle
    K.arrow(s, 620, 120, 620, 144, { color: C.line, w: 1.4 });
    K.arrow(s, 620, 248, 620, 272, { color: C.line, w: 1.4 });

    K.callout(s, 0, 408, 640,
      'The paper says allocation had to be balanced carefully across the cluster so the generators ' +
      'and the training ranks were not left waiting on simulator rendering.', { cols: 66 });
  };
})();
