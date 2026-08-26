/* Act III, training and merging. Purple. */
window.SCENES = window.SCENES || {};

(function () {
  var K = window.KIT, C = K.C, T = K.TINT, P = C.purple;
  var VSET = 'ablation validation sets, not the headline benchmarks';

  function f1of(rows, name, which) {
    for (var i = 0; i < rows.length; i++) if (rows[i].name === name) return rows[i][which][3];
    return null;
  }

  /* 22 the actual architecture, end to end */
  window.SCENES.S_BASE = function (root, api) {
    var s = K.board(root, { alt: 'Architecture from input through to the two output logits.' });
    var A_ = api.SS.arch || { lm: {}, vision: {} };
    K.head(s, 'What the checkpoint actually is',
           'the 3B is the language model, not the artifact');
    var Wd = 640;

    /* --- inputs --- */
    K.label(s, 0, 18, 'input');
    // image tile, drawn as a neutral glyph
    K.panel(s, 0, 30, 108, 84, { flat: true, fill: 'rgba(31,37,48,.04)' });
    s.appendChild(K.n('circle', { cx: 34, cy: 58, r: 9, fill: 'none', stroke: C.ink3 }));
    s.appendChild(K.n('path', { d: 'M12 100 L44 66 L66 88 L82 74 L96 100 Z',
      fill: 'none', stroke: C.ink3, 'stroke-width': 1.4 }));
    K.label(s, 0, 128, 'image, optional', { size: 9 });

    K.panel(s, 124, 30, 236, 84);
    K.mono(s, 138, 52, '<Instruct>', { size: 11.5, color: C.ink3 });
    K.mono(s, 138, 72, '<Query>', { size: 11.5, color: P, weight: 700 });
    K.mono(s, 138, 92, '<Document>', { size: 11.5, color: C.ink3 });
    K.label(s, 124, 128, 'the prompt, where the policy lives', { size: 9 });

    /* --- encoder --- */
    K.arrow(s, 54, 120, 54, 168, { color: C.line });
    K.panel(s, 0, 176, 108, 56, { stroke: C.line });
    K.mono(s, 14, 200, 'Pixtral', { size: 12, color: C.ink });
    K.label(s, 14, 218, 'vision encoder', { size: 8.6 });
    K.label(s, 14, 232, A_.vision.dim + 'd  ' + A_.vision.layers + ' layers', { size: 8.2 });
    K.label(s, 0, 248, 'frozen', { color: C.ink3, size: 9 });

    K.arrow(s, 108, 204, 146, 204, { color: C.line });
    K.arrow(s, 240, 120, 240, 190, { color: C.line });

    /* --- the language model --- */
    K.panel(s, 150, 176, 300, 132, { fill: T.purple, stroke: P });
    K.mono(s, 166, 202, 'Ministral-3-3B', { size: 15, color: C.ink, weight: 700 });
    K.label(s, 166, 220, A_.lm.dim + 'd  ' + A_.lm.layers + ' layers  ' + A_.lm.heads + 'h/' + A_.lm.kvHeads + 'kv', { color: P });
    for (var i = 0; i < 9; i++) {
      s.appendChild(K.n('rect', { x: 166 + i * 31, y: 238, width: 25, height: 40, rx: 3,
        fill: 'rgba(107,63,168,' + (0.16 + i * 0.045) + ')' }));
    }
    K.label(s, 166, 296, 'lora adapters here, and only here', { color: P, size: 9 });

    /* --- the head --- */
    K.arrow(s, 452, 240, 486, 240, { color: C.line });
    K.panel(s, 494, 176, 146, 132, { stroke: C.line });
    K.label(s, 508, 200, 'output head', { color: C.ink3 });
    K.mono(s, 508, 226, 'z_yes', { size: 12.5, color: C.red });
    K.bar(s, 560, 216, 66, 11, 0.82, { color: C.red });
    K.mono(s, 508, 254, 'z_no', { size: 12.5, color: C.teal });
    K.bar(s, 560, 244, 66, 11, 0.26, { color: C.teal });
    K.label(s, 508, 288, 'softmax, two tokens', { size: 8.6 });

    /* --- what ships --- */
    K.label(s, 0, 348, 'and the checkpoint that ships is a merge, not a run');
    var mix = [['PG', 0.6, P], ['P', 0.3, C.ink3], ['Instruct', 0.1, C.amber]], cx = 0;
    mix.forEach(function (m) {
      var w = m[1] * Wd;
      s.appendChild(K.n('rect', { x: cx, y: 360, width: w - 3, height: 32, rx: 5, fill: m[2] }));
      K.mono(s, cx + 12, 381, m[0] + ' ' + m[1], { size: 12.5, color: '#fff', weight: 700 });
      cx += w;
    });
    K.callout(s, 0, 406, Wd,
      'The 3B counts the language model, about ' + A_.lm.approxB + 'B from params.json. The vision '
      + 'encoder adds roughly ' + A_.vision.approxM + 'M more, so the checkpoint is about '
      + A_.totalApproxB + 'B.',
      { color: P, tint: T.purple, cols: 70 });
    K.foot(s, 'Layer and dimension counts are read from params.json in the Hugging Face repo. Parameter totals are computed from those shapes, so they are approximate. Section 6.1 says LoRA is applied to the language model parameters, 6.2 gives the merge.');
  };

  /* 23 LoRA against full SFT */
  window.SCENES.S_LORA = function (root, api) {
    var s = K.board(root, { alt: 'LoRA against full SFT on two validation sets.' });
    K.head(s, 'LoRA, or full SFT', 'close enough that efficiency decided it');
    var L = api.SS.loraVsSft || {}, cols = L.cols || [];
    var sets = [['Aegis v2 validation', L.aegis || []],
                ['fine-grained taxonomy validation', L.taxonomy || []]];
    K.label(s, 0, 14, 'lora against full sft');
    sets.forEach(function (set, si) {
      var y = 38 + si * 168;
      K.label(s, 0, y, set[0], { color: P });
      set[1].forEach(function (r, ri) {
        var ry = y + 26 + ri * 58;
        K.mono(s, 0, ry + 18, r.name, { size: 13, color: C.ink });
        cols.forEach(function (c, ci) {
          var x = 150 + ci * 124;
          K.label(s, x, ry, c, { size: 9 });
          K.mono(s, x, ry + 20, r.vals[ci].toFixed(1), { size: 15,
            color: ci === 3 ? C.ink : C.ink3, weight: ci === 3 ? 700 : 400 });
        });
      });
      var g = set[1][1].vals[3] - set[1][0].vals[3];
      K.text(s, 0, y + 148, (g > 0 ? 'Full SFT ahead by ' : 'LoRA ahead by ')
        + Math.abs(g).toFixed(1) + ' F1', { size: 13.5, color: C.ink3 });
    });
    K.text(s, 0, 388, 'The report calls the overall difference insignificant and takes LoRA for training efficiency.',
      { size: 14, color: C.ink2 });
    K.foot(s, 'Both are ' + VSET + '.');
  };

  /* 24 two checkpoints pulling apart */
  window.SCENES.S_TWOCKPT = function (root, api) {
    var s = K.board(root, { alt: 'P and PG are strong on opposite validation sets.' });
    K.head(s, 'Two checkpoints, pulling apart', 'each one strong exactly where the other is weak');
    var rows = (api.SS.merge || {}).rows || [];
    var pairs = [
      ['Aegis v2 validation', f1of(rows, 'P', 'aegis'), f1of(rows, 'PG', 'aegis')],
      ['fine-grained taxonomy validation', f1of(rows, 'P', 'taxonomy'), f1of(rows, 'PG', 'taxonomy')]
    ];
    K.label(s, 0, 14, 'the two candidate checkpoints');
    K.panel(s, 0, 28, 300, 58, { fill: T.purple, stroke: P });
    K.mono(s, 18, 52, 'P', { size: 17, color: P, weight: 700 });
    K.label(s, 40, 52, 'public data only', { color: C.ink3 });
    K.panel(s, 340, 28, 300, 58, { fill: T.purple, stroke: P });
    K.mono(s, 358, 52, 'PG', { size: 17, color: P, weight: 700 });
    K.label(s, 388, 52, 'public plus generated', { color: C.ink3 });

    pairs.forEach(function (p, i) {
      var y = 130 + i * 130, lo = Math.min(p[1], p[2]) - 6, hi = Math.max(p[1], p[2]) + 4;
      K.label(s, 0, y, p[0], { color: C.ink3 });
      [['P', p[1], 0], ['PG', p[2], 1]].forEach(function (b, bi) {
        var by = y + 18 + bi * 40;
        K.mono(s, 0, by + 16, b[0], { size: 13, color: C.ink });
        var frac = (b[1] - lo) / (hi - lo);
        K.bar(s, 42, by + 4, 500, 16, frac, { color: bi ? P : C.ink3 });
        K.mono(s, 640, by + 17, b[1].toFixed(1), { size: 15, color: C.ink, anchor: 'end' });
      });
      var lead = Math.abs(p[1] - p[2]).toFixed(1);
      K.text(s, 42, y + 112, (p[1] > p[2] ? 'P' : 'PG') + ' ahead by ' + lead + ' F1',
        { size: 13.5, color: C.ink3 });
    });
    K.text(s, 0, 404, 'Strong in opposite places, so neither one is the model you ship.',
      { size: 15, color: C.ink2 });
    K.foot(s, 'Bars are scaled to the local range so the gap is legible. Both are ' + VSET + '.');
  };

  /* 25 the five measured recipes */
  window.SCENES.S_MERGE = function (root, api) {
    var s = K.board(root, { alt: 'Five measured merge recipes on two validation axes.' });
    K.head(s, 'Five recipes, and only five', 'spherical interpolation between the two checkpoints');
    var M = api.SS.merge || {}, rows = M.rows || [], cols = M.cols || [];
    var FINAL = '0.6PG+0.3P+0.1I';
    var xs = rows.map(function (r) { return r.aegis[3]; });
    var ys = rows.map(function (r) { return r.taxonomy[3]; });
    var x0 = Math.min.apply(null, xs) - 0.7, x1 = Math.max.apply(null, xs) + 0.7;
    var y0 = Math.min.apply(null, ys) - 4, y1 = Math.max.apply(null, ys) + 4;
    var L = 54, Rt = 640, Tp = 34, B = 250;
    function px(v) { return L + (v - x0) / (x1 - x0) * (Rt - L); }
    function py(v) { return B - (v - y0) / (y1 - y0) * (B - Tp); }

    K.label(s, 0, 14, 'five measured recipes, nothing in between');
    s.appendChild(K.n('rect', { x: L, y: Tp, width: Rt - L, height: B - Tp, fill: '#fff', stroke: C.line }));
    K.label(s, L, B + 18, 'aegis v2 validation f1', { size: 9.5 });
    K.label(s, 0, Tp + 6, 'taxonomy', { size: 9.5 });
    K.label(s, 0, Tp + 18, 'validation f1', { size: 9.5 });

    var dots = [], readout = K.n('g', {});
    rows.forEach(function (r, i) {
      var isF = r.name === FINAL;
      var g = K.n('g', { cursor: 'pointer', tabindex: 0, role: 'button', 'aria-label': r.name });
      g.appendChild(K.n('circle', { cx: px(r.aegis[3]), cy: py(r.taxonomy[3]), r: isF ? 8 : 5.5,
        fill: isF ? P : '#fff', stroke: P, 'stroke-width': 1.6 }));
      var lab = K.mono(g, px(r.aegis[3]) + (i === 4 ? -12 : 12),
        py(r.taxonomy[3]) + (i === 4 ? -12 : 4), r.name,
        { size: 11, color: C.ink2, anchor: i === 4 ? 'end' : 'start' });
      s.appendChild(g);
      dots.push(g);
      g.addEventListener('click', function () { pick(i); });
      g.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(i); }
      });
    });
    s.appendChild(readout);

    function pick(i) {
      readout.innerHTML = '';
      var r = rows[i];
      dots.forEach(function (g, j) {
        g.firstChild.setAttribute('fill', j === i ? P : (rows[j].name === FINAL ? P : '#fff'));
      });
      K.mono(readout, 0, 296, r.name, { size: 16, color: C.ink, weight: 700 });
      if (r.name === FINAL) K.label(readout, 0, 314, 'the shipped model', { color: P });
      [['aegis v2 validation', r.aegis], ['taxonomy validation', r.taxonomy]].forEach(function (set, si) {
        var y = 344 + si * 54;
        K.label(readout, 0, y, set[0], { color: C.ink3 });
        cols.forEach(function (c, ci) {
          K.label(readout, ci * 116, y + 16, c, { size: 9 });
          K.mono(readout, ci * 116, y + 34, set[1][ci].toFixed(1),
            { size: 14, color: ci === 3 ? P : C.ink3, weight: ci === 3 ? 700 : 400 });
        });
      });
    }
    pick(rows.findIndex ? rows.findIndex(function (r) { return r.name === FINAL; }) : 4);
    K.foot(s, 'Only these five recipes were measured, so the plot does not interpolate. Both axes are '
      + VSET + '.');
  };

  /* 26 what each stage buys */
  window.SCENES.S_STAGES = function (root, api) {
    var s = K.board(root, { alt: 'What each training stage adds.' });
    K.head(s, 'What each stage actually buys', 'fine-grained taxonomy validation set');
    var A = api.SS.stageAblation || {}, cols = A.cols || [], rws = A.rows || [];
    var mrow = ((api.SS.merge || {}).rows || []).filter(function (r) {
      return r.name === '0.6PG+0.3P+0.1I'; })[0];
    var names = ['base, no safety training', 'plus public data', 'plus generated data', 'after slerp merge'];
    var body = K.n('g', {}); s.appendChild(body);

    K.label(s, 0, 14, 'fine-grained taxonomy validation set');
    K.switcher(s, 0, 26, cols, function (mi) {
      body.innerHTML = '';
      var vals = rws.map(function (r) { return r.vals[mi]; });
      if (mrow) vals.push(mrow.taxonomy[mi]);
      var max = Math.max.apply(null, vals.concat([100]));
      vals.forEach(function (v, i) {
        var y = 84 + i * 76;
        K.mono(body, 0, y - 6, names[i], { size: 12.5, color: i === 3 ? P : C.ink2 });
        K.bar(body, 0, y + 4, 540, 26, v / max, { color: i === 3 ? P : C.ink3 });
        K.mono(body, 640, y + 24, v.toFixed(1), { size: 17, color: C.ink, anchor: 'end' });
        if (i) {
          var d = v - vals[i - 1];
          K.mono(body, 0, y + 52, (d >= 0 ? '+' : '') + d.toFixed(1),
            { size: 12, color: d >= 0 ? C.teal : C.red });
        }
      });
      if (mi === 2) {
        K.text(body, 0, 400, 'Zero recall means the base model never predicts a violation at all.',
          { size: 14, color: C.red });
      } else if (mi === 0) {
        K.text(body, 0, 400, 'The 37.8 accuracy is not partial competence, it is what you get for always answering no.',
          { size: 14, color: C.red });
      }
      s.appendChild(body);
    }, { tint: 'purple' });
    K.foot(s, 'Rows one to three are single checkpoints without merging. The final bar is the merged model on the same validation set, '
      + 'which is not the adaptability benchmark quoted elsewhere in this guide.');
  };
})();
