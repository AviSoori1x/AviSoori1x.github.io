/* Does it hold up. The benchmark table, the RL delta, and the honest caveats. */
window.SCENES = window.SCENES || {};

(function () {
  var K = window.KIT, C = K.C, T = K.TINT;

  function groupColor(g) {
    return g === 'ours' ? C.red : (g === 'single' ? C.blue : C.ink3);
  }

  /* ---- 24 the four metrics ---- */
  window.SCENES.S_METRICS = function (root, api) {
    var M = api.RN.metrics, f = api.RN.facts, o = api.RN.ours;
    var s = K.board(root, { alt: 'What the four navigation metrics measure.' });
    K.head(s, 'What the columns mean', 'where it stopped, whether it ever arrived, and how direct');

    var ROWS = [
      { k: 'NE', dir: 'lower', c: C.purple, d: M.NE, v: o.r2r.NE.toFixed(2) + ' m' },
      { k: 'SR', dir: 'higher', c: C.teal, d: M.SR, v: o.r2r.SR.toFixed(1) + '%' },
      { k: 'OS', dir: 'higher', c: C.blue, d: M.OS, v: o.r2r.OS.toFixed(1) + '%' },
      { k: 'SPL', dir: 'higher', c: C.amber, d: M.SPL, v: o.r2r.SPL.toFixed(1) + '%' }
    ];
    ROWS.forEach(function (r, i) {
      var y = 14 + i * 96;
      K.panel(s, 0, y, 640, 82, { stroke: r.c, fill: 'rgba(255,255,255,.9)' });
      K.mono(s, 20, y + 34, r.k, { size: 19, color: r.c, weight: 700 });
      K.mono(s, 20, y + 56, r.dir + ' is better', { size: 10.5, color: C.ink3 });
      K.para(s, 120, y + 28, r.d, 52, { size: 12, color: C.ink2 });
      K.mono(s, 620, y + 34, r.v, { size: 17, color: r.c, weight: 700, anchor: 'end' });
      K.label(s, 620, y + 54, 'ours, r2r-ce', { size: 9, color: C.ink3, anchor: 'end' });
    });

    K.callout(s, 0, 402, 640,
      'The gap between oracle success and success rate is the share of episodes that came within ' +
      f.success_radius_m + ' m of the goal but did not stop within it. Ours is ' +
      (o.r2r.OS - o.r2r.SR).toFixed(1) + ' points.', { cols: 66, color: C.blue, tint: T.blue });
  };

  /* ---- 25 the whole table, live ---- */
  window.SCENES.S_TABLE = function (root, api) {
    var RN = api.RN, rows = RN.table1.slice();
    var s = K.board(root, { alt: 'Every baseline in the paper table, sortable by metric.' });
    K.head(s, 'Every row in the table', 'colour is the sensing class, not the score');

    var bench = 'r2r', metric = 'SR';
    var body = K.n('g', {}); s.appendChild(body);

    K.label(s, 0, 12, 'benchmark');
    var sw1 = K.switcher(s, 0, 24, ['R2R-CE', 'RxR-CE'], function (i) {
      bench = i ? 'rxr' : 'r2r';
      if (bench === 'rxr' && metric === 'OS') metric = 'SR';
      draw();
    }, { tint: 'blue' });
    K.label(s, 300, 12, 'metric');
    K.switcher(s, 300, 24, ['SR', 'SPL', 'NE', 'OS'], function (i) {
      metric = ['SR', 'SPL', 'NE', 'OS'][i];
      draw();
    }, { tint: 'amber' });

    function draw() {
      body.innerHTML = '';
      var lower = metric === 'NE';
      var have = rows.filter(function (r) { return r[bench][metric] != null; });
      if (bench === 'rxr' && metric === 'OS') {
        K.panel(body, 0, 90, 640, 70, { flat: true, fill: T.ink, stroke: C.line });
        K.mono(body, 18, 122, 'The paper does not report oracle success for RxR-CE.',
          { size: 13, color: C.ink3 });
        K.mono(body, 18, 144, 'Pick another metric.', { size: 12, color: C.ink3 });
        return;
      }
      have.sort(function (a, b) {
        return lower ? a[bench][metric] - b[bench][metric] : b[bench][metric] - a[bench][metric];
      });
      var mx = Math.max.apply(null, have.map(function (r) { return r[bench][metric]; }));
      var unit = metric === 'NE' ? ' m' : '%';
      have.forEach(function (r, i) {
        var y = 78 + i * 26, col = groupColor(r.group);
        var frac = lower ? (1 - (r[bench][metric] / mx) * 0.72) : r[bench][metric] / mx;
        K.mono(body, 0, y + 11, r.short, { size: 11, color: r.ours ? C.red : C.ink2,
          weight: r.ours ? 700 : 400 });
        K.bar(body, 208, y + 1, 340, 13, frac, { color: col });
        K.mono(body, 640, y + 11, r[bench][metric].toFixed(metric === 'NE' ? 2 : 1) + unit,
          { size: 11.5, color: col, anchor: 'end', weight: r.ours ? 700 : 400 });
      });

      var ly = 78 + have.length * 26 + 14;
      [['ours, one RGB camera', C.red], ['one RGB camera', C.blue],
       ['depth, LiDAR or several cameras', C.ink3]].forEach(function (l, i) {
        body.appendChild(K.n('rect', { x: i * 210, y: ly, width: 11, height: 11, rx: 3, fill: l[1] }));
        K.mono(body, i * 210 + 18, ly + 10, l[0], { size: 10.5, color: C.ink3 });
      });
      if (lower) K.mono(body, 0, ly + 32, 'bars inverted so longer still means better',
        { size: 10.5, color: C.ink3 });
    }
    draw();
    K.foot(s, 'Table 1, validation unseen on both benchmarks, RxR-CE english-only. Three systems ' +
      'appear in both sensing groups and are suffixed accordingly. Parsed from the paper rather ' +
      'than retyped, so the ordering is whatever the numbers say.');
  };

  /* ---- 26 the headline ---- */
  window.SCENES.S_SENSORGAP = function (root, api) {
    var h = api.RN.headline;
    var s = K.board(root, { alt: 'The margin over the best single-camera and best depth systems.' });
    K.head(s, 'It beats the rigs without the rig', 'R2R-CE validation unseen, success rate');

    var ROWS = [
      { n: h.best_single_model, v: h.best_single_sr, c: C.blue, tag: 'best single camera' },
      { n: h.best_depth_model, v: h.best_depth_sr, c: C.ink3, tag: 'best with depth or multi-camera' },
      { n: 'Robostral Navigate', v: h.r2r_sr, c: C.red, tag: 'one RGB camera' }
    ];
    var mx = 100;
    ROWS.forEach(function (r, i) {
      var y = 30 + i * 108;
      K.mono(s, 0, y, r.n, { size: 14, color: r.c, weight: r.c === C.red ? 700 : 400 });
      K.label(s, 0, y + 18, r.tag, { size: 9.5 });
      K.bar(s, 0, y + 30, 520, 26, r.v / mx, { color: r.c });
      K.mono(s, 640, y + 50, r.v.toFixed(1) + '%', { size: 24, color: r.c, weight: 700,
        anchor: 'end' });
    });

    K.label(s, 0, 366, 'the two margins');
    [[h.gain_vs_single, 'over the best other single-camera system', C.blue],
     [h.gain_vs_depth, 'over the best system that gets depth or several cameras', C.ink]
    ].forEach(function (r, i) {
      var y = 380 + i * 58;
      K.panel(s, 0, y, 640, 48, { flat: true, fill: i ? T.red : T.blue,
        stroke: i ? C.red : C.blue });
      K.mono(s, 18, y + 30, '+' + r[0].toFixed(1), { size: 18, color: i ? C.red : C.blue,
        weight: 700 });
      K.mono(s, 84, y + 30, r[1], { size: 12.5, color: i ? C.red : C.blue });
    });

    K.callout(s, 0, 500, 640,
      'The second margin carries the argument: a minimal-sensor recipe outperforming approaches ' +
      'that leverage privileged sensing.', { cols: 66 });
  };

  /* ---- 27 what RL bought ---- */
  window.SCENES.S_RLGAIN = function (root, api) {
    var rl = api.RN.rl;
    var s = K.board(root, { alt: 'Success rate before and after the reinforcement learning pass.' });
    K.head(s, 'The last few points came from RL', 'supervised baseline against the post-RL checkpoint');

    var SETS = [
      { n: 'R2R-CE, seen', a: rl.r2r_sft_seen, b: rl.r2r_rl_seen, g: rl.r2r_gain_seen },
      { n: 'R2R-CE, unseen', a: rl.r2r_sft_unseen, b: rl.r2r_rl_unseen, g: rl.r2r_gain_unseen },
      { n: 'RxR-CE, unseen', a: rl.rxr_sft, b: rl.rxr_rl, g: rl.rxr_gain }
    ];
    var X0 = 0, XW = 470, lo = 68, hi = 82;
    function px(v) { return X0 + ((v - lo) / (hi - lo)) * XW; }

    SETS.forEach(function (S, i) {
      var y = 40 + i * 116;
      K.mono(s, 0, y - 8, S.n, { size: 13, color: C.ink, weight: 500 });
      s.appendChild(K.n('line', { x1: X0, y1: y + 22, x2: X0 + XW, y2: y + 22,
        stroke: C.line, 'stroke-width': 2 }));
      s.appendChild(K.n('line', { x1: px(S.a), y1: y + 22, x2: px(S.b), y2: y + 22,
        stroke: C.teal, 'stroke-width': 6 }));
      s.appendChild(K.n('circle', { cx: px(S.a), cy: y + 22, r: 7, fill: '#fff',
        stroke: C.ink3, 'stroke-width': 2 }));
      s.appendChild(K.n('circle', { cx: px(S.b), cy: y + 22, r: 8, fill: C.teal }));
      K.mono(s, px(S.a), y + 48, S.a.toFixed(2), { size: 11, color: C.ink3, anchor: 'middle' });
      K.mono(s, px(S.b), y + 48, S.b.toFixed(2), { size: 11.5, color: C.teal, anchor: 'middle',
        weight: 700 });
      K.mono(s, 640, y + 28, '+' + S.g.toFixed(2), { size: 20, color: C.teal, weight: 700,
        anchor: 'end' });
    });

    K.mono(s, X0, 400, 'supervised', { size: 10.5, color: C.ink3 });
    K.mono(s, X0 + XW, 400, 'after RL', { size: 10.5, color: C.teal, anchor: 'end' });

    K.panel(s, 0, 414, 640, 74, { flat: true, fill: T.amber, stroke: C.amber });
    K.mono(s, 18, 442, 'peak during the run: ' + api.RN.rl.r2r_peak_seen.toFixed(2) +
      '% seen, ' + api.RN.rl.r2r_peak_unseen.toFixed(2) + '% unseen',
      { size: 13, color: C.amber });
    K.mono(s, 18, 466, 'higher than the reported checkpoint on the seen split, worth noticing',
      { size: 11.5, color: C.amber });

    K.foot(s, 'The paper attributes the gain to information-seeking and recovery, which ' +
      'shortest-path imitation cannot demonstrate.');
  };

  /* ---- 28 the one it loses ---- */
  window.SCENES.S_RXR = function (root, api) {
    var RN = api.RN, o = RN.ours;
    var rival = RN.table1.filter(function (r) {
      return r.group === 'depth' && r.model.indexOf('Qwen-RobotNav-8B') === 0;
    })[0] || RN.table1.filter(function (r) { return r.group === 'depth'; })
      .sort(function (a, b) { return b.rxr.SR - a.rxr.SR; })[0];
    var bestSingle = RN.table1.filter(function (r) { return r.group === 'single'; })
      .sort(function (a, b) { return b.rxr.SR - a.rxr.SR; })[0];

    var s = K.board(root, { alt: 'RxR-CE, where the model wins and where it does not.' });
    K.head(s, 'Third on success, first on efficiency',
      'RxR-CE validation unseen, english-only, against the best depth system');

    var CMP = [
      { k: 'SR', ours: o.rxr.SR, them: rival.rxr.SR, unit: '%', hi: true },
      { k: 'SPL', ours: o.rxr.SPL, them: rival.rxr.SPL, unit: '%', hi: true },
      { k: 'NE', ours: o.rxr.NE, them: rival.rxr.NE, unit: ' m', hi: false }
    ];
    K.label(s, 0, 12, 'ours vs ' + rival.short + ', which gets depth');
    CMP.forEach(function (r, i) {
      var y = 28 + i * 104;
      var win = r.hi ? r.ours > r.them : r.ours < r.them;
      var col = win ? C.teal : C.red;
      K.panel(s, 0, y, 640, 88, { stroke: col, fill: win ? T.teal : T.red });
      K.mono(s, 20, y + 34, r.k, { size: 18, color: col, weight: 700 });
      K.mono(s, 20, y + 58, win ? 'we win' : 'we do not', { size: 11, color: col });
      K.mono(s, 130, y + 30, 'ours', { size: 11, color: C.ink3 });
      K.mono(s, 130, y + 56, r.ours.toFixed(r.k === 'NE' ? 2 : 1) + r.unit,
        { size: 20, color: col, weight: 700 });
      K.mono(s, 300, y + 30, 'theirs', { size: 11, color: C.ink3 });
      K.mono(s, 300, y + 56, r.them.toFixed(r.k === 'NE' ? 2 : 1) + r.unit,
        { size: 20, color: C.ink3, weight: 700 });
      K.mono(s, 620, y + 52, (r.ours > r.them ? '+' : '') +
        (r.ours - r.them).toFixed(r.k === 'NE' ? 2 : 1), { size: 17, color: col,
        weight: 700, anchor: 'end' });
    });

    var above = RN.table1.filter(function (x) { return x.rxr.SR > o.rxr.SR; });
    K.panel(s, 0, 344, 640, 74, { flat: true, fill: T.blue, stroke: C.blue });
    K.mono(s, 18, 372, 'among single-camera systems it is first: ' + o.rxr.SR.toFixed(1) +
      '% against ' + bestSingle.rxr.SR.toFixed(1) + '%', { size: 13, color: C.blue });
    K.mono(s, 18, 396, 'the ' + (above.length === 1 ? 'one system' : above.length + ' systems') +
      ' above it all get depth or several cameras', { size: 11.5, color: C.blue });

    K.callout(s, 0, 434, 640,
      'The paper compares itself only to the higher of the two, and calls that result competitive: ' +
      o.rxr.SR.toFixed(1) + '% against ' + rival.rxr.SR.toFixed(1) + '%.', { cols: 66 });
  };

  /* ---- 29 scope ---- */
  window.SCENES.S_LIMITS = function (root, api) {
    var s = K.board(root, { alt: 'Three things the benchmark numbers do not cover.' });
    K.head(s, 'Read the scope before the score', 'three things the numbers do not cover');

    var L = [
      { t: 'the benchmark measures the waypoints, not the stack',
        d: 'evaluation runs a Habitat pathfinder between predicted waypoints, so the diffusion ' +
           'policy and controller that run on the real robot are not what is being scored' },
      { t: 'cross-embodiment is shown on two mobile robot platforms',
        d: 'the abstract argues the recipe reaches wheeled, legged and aerial robots. What the ' +
           'paper demonstrates is two substantially different mobile robot platforms, a Galaxea ' +
           'R1 and a Hiwonder JetAuto' },
      { t: 'training is entirely simulated',
        d: 'which is the point of the recipe, but it means the real-world evidence is the two ' +
           'deployments rather than the benchmark rows' }
    ];
    L.forEach(function (l, i) {
      var y = 14 + i * 148;
      K.panel(s, 0, y, 640, 126, { stroke: C.amber, fill: T.amber });
      K.mono(s, 20, y + 34, String(i + 1), { size: 13, color: C.amber, weight: 700 });
      K.para(s, 46, y + 34, l.t, 56, { size: 13.5, color: C.amber, lh: 19 });
      K.para(s, 46, y + 74, l.d, 62, { size: 12, color: C.ink2, lh: 17 });
    });

    K.callout(s, 0, 462, 640,
      'None of this undoes the result. It means the result is about the recipe, which is what ' +
      'the paper set out to argue.', { cols: 66, color: C.teal, tint: T.teal });
  };

  /* ---- 30 the close ---- */
  window.SCENES.S_END = function (root, api) {
    var h = api.RN.headline, f = api.RN.facts;
    var s = K.board(root, { alt: 'The whole recipe in four steps.' });
    K.head(s, 'The recipe, end to end', 'four decisions, each one traceable to the same objective');

    var STEPS = [
      ['one RGB camera', 'the most universally available sensor', C.blue],
      ['point at a pixel', 'with a metric fallback when the goal is out of frame', C.teal],
      ['pack the episode', f.token_saving_x + '× fewer tokens, months to days', C.purple],
      ['then online RL', 'exploration and recovery, which shortest paths never show', C.amber]
    ];
    STEPS.forEach(function (st, i) {
      var y = 12 + i * 84;
      K.panel(s, 0, y, 640, 70, { stroke: st[2], fill: 'rgba(255,255,255,.9)' });
      s.appendChild(K.n('rect', { x: 0, y: y, width: 5, height: 70, rx: 2.5, fill: st[2] }));
      K.mono(s, 24, y + 30, st[0], { size: 15, color: st[2], weight: 500 });
      K.mono(s, 24, y + 52, st[1], { size: 12, color: C.ink3 });
      if (i < 3) K.arrow(s, 320, y + 70, 320, y + 84, { color: C.line, w: 1.3, headSize: 5 });
    });

    K.panel(s, 0, 356, 310, 96, { stroke: C.red, fill: T.red });
    K.big(s, 20, 412, h.r2r_sr.toFixed(1) + '%', { size: 40, color: C.red });
    K.mono(s, 20, 434, 'R2R-CE success, validation unseen', { size: 11, color: C.red });
    K.panel(s, 330, 356, 310, 96, { stroke: C.red, fill: T.red });
    K.big(s, 350, 412, h.rxr_sr.toFixed(1) + '%', { size: 40, color: C.red });
    K.mono(s, 350, 434, 'RxR-CE success, validation unseen', { size: 11, color: C.red });

    K.callout(s, 0, 472, 640,
      'The paper leads R2R-CE on every reported metric, and takes RxR-CE on path efficiency and ' +
      'navigation error while placing third on its success rate.', { cols: 66 });
  };
})();
