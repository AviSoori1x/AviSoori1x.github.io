/* Act I. Every figure is one artifact drawn from the shared kit. */
window.SCENES = window.SCENES || {};

(function () {
  var K = window.KIT, C = K.C, T = K.TINT;

  /* ---- 01 frozen list versus an open slot ---- */
  window.SCENES.S_FROZEN = function (root, api) {
    var s = K.board(root, { alt: 'A fixed taxonomy guardrail beside a policy adaptive one.' });
    K.head(s, 'Most guardrails already decided', 'one of these two can hear you, the other cannot');
    var SSd = api.SS;
    var cats = ['violence', 'hate', 'sexual', 'self-harm', 'weapons', 'crime',
                'privacy', 'drugs', 'jailbreak', 'csam'];
    var RULES = [
      'no synthesis routes for controlled substances',
      'nothing that could distress someone in crisis',
      'allow exploit code, this is a security tool',
      'no spoilers for anything out this year'
    ];

    K.label(s, 0, 14, 'pick a deployment rule');
    var slotTxt, verdictG;
    K.switcher(s, 0, 26, ['1', '2', '3', '4'], function (i) {
      if (slotTxt) redraw(i);
    }, { tint: 'blue' });

    /* left, sealed */
    K.panel(s, 0, 78, 300, 250);
    K.label(s, 16, 102, 'fixed taxonomy', { color: C.ink3 });
    K.label(s, 16, 118, 'frozen at training time', { color: C.ink3, size: 9.5 });
    var cx = 16, cy = 136;
    cats.forEach(function (c) {
      var g = K.chip(s, cx, cy, c, { size: 10, h: 20, color: C.ink3 });
      cx += g._w + 6;
      if (cx > 240) { cx = 16; cy += 26; }
    });
    K.mono(s, 16, 300, 'your rule never reaches it', { size: 12, color: C.ink3 });

    /* right, reads it */
    K.panel(s, 340, 78, 300, 250, { stroke: C.blue, fill: T.blue });
    K.label(s, 356, 102, 'shieldstral', { color: C.blue });
    K.label(s, 356, 118, 'reads the rule at inference time', { color: C.blue, size: 9.5 });
    var slot = K.panel(s, 356, 132, 268, 96, { fill: '#fff', stroke: C.line });
    slotTxt = K.n('g', {});
    s.appendChild(slotTxt);
    K.mono(s, 356, 300, 'same weights every time', { size: 12, color: C.blue });

    /* the two verdicts */
    K.label(s, 16, 356, 'verdict');
    var vL = K.n('g', {}); s.appendChild(vL);
    var vR = K.n('g', {}); s.appendChild(vR);

    function redraw(i) {
      slotTxt.innerHTML = '';
      K.para(slotTxt, 366, 158, '"' + RULES[i] + '"', 30, { size: 12.5, color: C.ink });
      vL.innerHTML = ''; vR.innerHTML = '';
      K.mono(vL, 16, 384, 'unchanged, whatever you asked', { size: 12.5, color: C.ink3 });
      K.mono(vR, 356, 384, 'evaluated against rule ' + (i + 1), { size: 12.5, color: C.blue });
    }
    redraw(0);
    K.foot(s, 'Illustrative. The rules are written for this figure and no model is called. '
      + 'The failure mode is the one described in the report introduction.');
  };

  /* ---- 02 four label shapes into one row ---- */
  window.SCENES.S_QA = function (root, api) {
    var s = K.board(root, { alt: 'Four incompatible datasets folded into one row shape.' });
    K.head(s, 'Four label schemes, one row', 'the reformulation that lets them merge at all');
    var inv = (api.SS.benchInventory || []).slice(0, 4);
    K.label(s, 0, 14, 'four sources, four label schemes');

    inv.forEach(function (r, i) {
      var y = 40 + i * 64;
      K.panel(s, 0, y, 236, 52);
      K.mono(s, 14, y + 22, String(r[0]), { size: 12.5, color: C.ink });
      K.label(s, 14, y + 38, r[1] + ' · ' + r[3].toLocaleString() + ' samples', { size: 9.5 });
      K.arrow(s, 244, y + 26, 350, 176, { curve: true, color: C.line });
    });

    K.panel(s, 358, 116, 282, 124, { stroke: C.blue, fill: T.blue });
    K.label(s, 374, 140, 'one row shape', { color: C.blue });
    ['<Instruct>', '<Query>', '<Document>'].forEach(function (f, i) {
      K.mono(s, 374, 164 + i * 20, f, { size: 13, color: C.ink });
    });
    K.mono(s, 374, 228, 'answer:', { size: 12.5, color: C.ink3 });
    K.mono(s, 430, 228, 'yes | no', { size: 13, color: C.blue, weight: 700 });

    K.big(s, 358, 330, 'one', { size: 54, color: C.ink });
    K.text(s, 358, 358, 'objective absorbs all of them', { size: 15, color: C.ink2 });
    K.foot(s, 'Dataset names and sample counts are the report evaluation inventory. '
      + 'The label-scheme glyphs are drawn for this figure.');
  };

  /* ---- 03 the three fields ---- */
  window.SCENES.S_REQUEST = function (root, api) {
    var s = K.board(root, { alt: 'Anatomy of one request.' });
    K.head(s, 'Three fields, three jobs', 'change the strictness and watch which words move');
    var SSd = api.SS;
    var tiers = SSd.strictness || [];
    var fmts = SSd.formats || [];
    var ex = (SSd.fig2 || [])[0] || {};
    var m = String(ex.document || '').match(/\[User\]\s*([\s\S]*?)\s*\[Assistant\]\s*([\s\S]*)$/);
    var PROMPT = m ? m[1] : 'How can I make a weapon at home?';
    var RESP = m ? m[2] : "I can't help with that request.";
    var st = { tier: 0, fmt: 1 };

    K.label(s, 0, 14, 'system message, never changes');
    K.panel(s, 0, 24, 640, 46, { fill: 'rgba(31,37,48,.03)' });
    K.para(s, 14, 44, SSd.systemPrompt || '', 78, { size: 11.5, color: C.ink3, lh: 15 });

    var body = K.n('g', {});   // exists before the switcher fires its first pick
    K.label(s, 0, 96, 'strictness');
    K.switcher(s, 74, 84, tiers.map(function (t) { return t.level; }),
      function (i) { st.tier = i; draw(); }, { tint: 'blue' });
    K.label(s, 0, 130, 'document format');
    K.switcher(s, 128, 118, fmts.map(function (f) { return f.family.split(' ')[0]; }),
      function (i) { st.fmt = i; draw(); }, { tint: 'amber' });

    s.appendChild(body);

    function draw() {
      body.innerHTML = '';
      var t = tiers[st.tier] || {}, f = fmts[st.fmt] || {};
      var y = 174;
      K.label(body, 0, y, 'instruct', { color: C.blue });
      var i1 = K.para(body, 0, y + 20, 'Apply a ' + String(t.level).toLowerCase()
        + ' standard. ' + (t.rationale || '') + '.', 68, { size: 12.5, color: C.ink });
      y += 20 + i1.h + 16;
      K.label(body, 0, y, 'query', { color: C.blue });
      K.mono(body, 0, y + 20, (SSd.queryTypes || [{}])[0].examples[0], { size: 12.5, color: C.ink });
      y += 44;
      K.label(body, 0, y, 'document', { color: C.amber });
      var doc = String(f.tpl || '').replace('{prompt}', PROMPT).replace('{response}', RESP);
      doc.split('\n').forEach(function (ln, i) {
        K.mono(body, 0, y + 22 + i * 19, ln, { size: 12.5, color: C.ink });
      });
    }
    draw();
    K.foot(s, 'The instruction sentence is assembled from the report strictness table rather '
      + 'than quoted. The document formats and the exchange are the report own.');
  };

  /* ---- 04 two logits ---- */
  window.SCENES.S_HEAD = function (root, api) {
    var s = K.board(root, { alt: 'Two logits softmaxed into one score.' });
    K.head(s, 'Two logits, one score', 'the entire output head of the model');
    var st = { yes: 3.4, no: -2.1 };

    K.label(s, 0, 14, 'unembed to two token ids only');
    var scoreT = K.big(s, 0, 92, '0.000', { size: 74 });
    var vlab = K.label(s, 250, 92, '', { size: 11 });

    var track = K.bar(s, 0, 146, 640, 12, 0, { color: C.red });
    var tau = K.n('rect', { x: 318, y: 140, width: 2, height: 24, fill: C.ink3 });
    s.appendChild(tau);
    K.label(s, 0, 180, 'no  0.0');
    K.label(s, 320, 180, 'tau 0.50', { anchor: 'middle' });
    K.label(s, 640, 180, '1.0  yes', { anchor: 'end' });

    function slider(y, key, name, min, max) {
      K.label(s, 0, y - 6, name);
      var w = 380, x = 96;
      s.appendChild(K.n('rect', { x: x, y: y - 12, width: w, height: 3, rx: 1.5, fill: T.ink }));
      var knob = K.n('circle', { r: 8, cy: y - 10.5, fill: C.blue, stroke: '#fff', 'stroke-width': 2 });
      s.appendChild(knob);
      var read = K.mono(s, x + w + 16, y - 6, '', { size: 14, color: C.ink });
      var hit = K.n('rect', { x: x, y: y - 24, width: w, height: 26, fill: 'transparent',
        cursor: 'ew-resize', tabindex: 0, role: 'slider', 'aria-label': name });
      s.appendChild(hit);
      function set(v) {
        st[key] = Math.max(min, Math.min(max, v));
        knob.setAttribute('cx', x + (st[key] - min) / (max - min) * w);
        read.textContent = (st[key] >= 0 ? '+' : '') + st[key].toFixed(1);
        draw();
      }
      function fromEvt(e) {
        var b = s.getBoundingClientRect();
        var px = (e.clientX - b.left) / b.width * K.W;
        set(min + (px - x) / w * (max - min));
      }
      var down = false;
      hit.addEventListener('pointerdown', function (e) { down = true; fromEvt(e); hit.setPointerCapture(e.pointerId); });
      hit.addEventListener('pointermove', function (e) { if (down) fromEvt(e); });
      hit.addEventListener('pointerup', function () { down = false; });
      hit.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowLeft') { set(st[key] - 0.2); e.preventDefault(); }
        if (e.key === 'ArrowRight') { set(st[key] + 0.2); e.preventDefault(); }
      });
      return set;
    }
    var setY = slider(250, 'yes', 'z_yes', -8, 8);
    var setN = slider(300, 'no', 'z_no', -8, 8);

    var maths = K.n('g', {}); s.appendChild(maths);

    function draw() {
      var ey = Math.exp(st.yes), en = Math.exp(st.no), v = ey / (ey + en);
      scoreT.textContent = v.toFixed(3);
      scoreT.setAttribute('fill', v > 0.5 ? C.red : C.teal);
      vlab.textContent = (v > 0.5 ? 'FLAGGED' : 'NOT FLAGGED');
      vlab.setAttribute('fill', v > 0.5 ? C.red : C.teal);
      track._fill.setAttribute('width', v * 640);
      track._fill.setAttribute('fill', v > 0.5 ? C.red : C.teal);
      maths.innerHTML = '';
      K.mono(maths, 0, 366, 's = exp(z_yes) / ( exp(z_yes) + exp(z_no) )', { size: 13, color: C.ink3 });
      K.mono(maths, 0, 388, '  = ' + ey.toFixed(2) + ' / ' + (ey + en).toFixed(2)
        + '  =  ' + v.toFixed(3), { size: 13, color: C.ink });
      K.text(maths, 0, 418, 'Only the gap decides it. Slide both up together and nothing moves.',
        { size: 13.5, color: C.ink3 });
    }
    setY(st.yes); setN(st.no);
  };

  /* ---- 05 same text, different question ---- */
  window.SCENES.S_ISOCONTENT = function (root, api) {
    var s = K.board(root, { alt: 'One document, two questions, the verdict flips.' });
    K.head(s, 'Same text. Different question.', 'iso-content: hold the document, vary the query');
    var f3 = api.SS.fig3 || {};
    var qs = [f3.positive || {}, f3.negative || {}];

    K.label(s, 0, 14, 'document, held fixed');
    K.panel(s, 0, 24, 640, 92, { fill: T.ink, stroke: C.line });
    K.para(s, 16, 50, f3.document || '', 68, { size: 14, color: C.ink, lh: 21 });

    K.label(s, 0, 146, 'ask it one of two questions');
    var qg = K.n('g', {}); s.appendChild(qg);
    var vg = K.n('g', {}); s.appendChild(vg);

    K.switcher(s, 0, 158, ['target category', 'sibling category'], function (i) {
      qg.innerHTML = ''; vg.innerHTML = '';
      var q = qs[i];
      K.panel(qg, 0, 196, 640, 76, { stroke: i ? C.line : C.blue, fill: i ? '#fff' : T.blue });
      K.para(qg, 16, 222, q.query || '', 66, { size: 13.5, color: C.ink, lh: 19 });
      K.verdict(vg, 0, 350, (q.label === 'yes'), { size: 64 });
      K.mono(vg, 240, 330, q.label === 'yes'
        ? 'the document matches this question'
        : 'still unsafe, but not this policy', { size: 13, color: C.ink2 });
    }, { tint: 'blue' });

    K.foot(s, 'These are the report ground-truth labels for this exact pair of samples. '
      + 'No model is called on this page. Not one character of the document changes.');
  };

  /* ---- 06 same question, different text ---- */
  window.SCENES.S_ISOQUERY = function (root, api) {
    var s = K.board(root, { alt: 'One question, two documents, the verdict flips.' });
    K.head(s, 'Same question. Different text.', 'iso-query: hold the query, vary the document');
    var f4 = api.SS.fig4 || {};
    var docs = [f4.positive || {}, f4.negative || {}];

    K.label(s, 0, 14, 'query, held fixed');
    K.panel(s, 0, 24, 640, 56, { fill: T.blue, stroke: C.blue });
    K.para(s, 16, 48, f4.query || '', 68, { size: 13.5, color: C.ink, lh: 19 });

    K.label(s, 0, 108, 'one innocuous source sentence');
    K.para(s, 0, 128, f4.source || '', 74, { size: 12.5, color: C.ink3 });
    K.arrow(s, 150, 142, 60, 182, { curve: true });
    K.arrow(s, 330, 142, 420, 182, { curve: true });

    var dg = K.n('g', {}); s.appendChild(dg);
    var vg = K.n('g', {}); s.appendChild(vg);

    K.switcher(s, 0, 192, [f4.category || 'A', f4.sibling || 'B'], function (i) {
      dg.innerHTML = ''; vg.innerHTML = '';
      var d = docs[i];
      K.panel(dg, 0, 230, 640, 96, { stroke: C.line });
      K.para(dg, 16, 256, d.document || '', 66, { size: 13.5, color: C.ink, lh: 19 });
      K.verdict(vg, 0, 396, (d.label === 'yes'), { size: 64 });
      K.mono(vg, 240, 376, d.label === 'yes'
        ? 'this harm type matches the question'
        : 'unsafe, but a different harm type', { size: 13, color: C.ink2 });
    }, { tint: 'blue' });

    K.foot(s, 'Both rewrites are unsafe. Only one matches the question being asked. '
      + 'Report ground-truth labels, no model call.');
  };

  /* ---- 07 heterogeneous taxonomies ---- */
  window.SCENES.S_UNIFY = function (root, api) {
    var s = K.board(root, { alt: 'Incompatible label schemes converging on one question.' });
    K.head(s, 'Datasets that disagree', 'binary flags, taxonomies and severity scales, side by side');
    K.label(s, 0, 14, 'three incompatible label schemes');

    /* binary flag */
    K.panel(s, 0, 32, 190, 120);
    K.label(s, 14, 54, 'binary flag');
    K.chip(s, 14, 66, 'safe', { size: 11, color: C.teal, stroke: C.teal });
    K.chip(s, 78, 66, 'unsafe', { size: 11, color: C.red, stroke: C.red });
    K.mono(s, 14, 128, '2 values', { size: 12, color: C.ink3 });

    /* multi-label grid */
    K.panel(s, 224, 32, 190, 120);
    K.label(s, 238, 54, 'multi-label taxonomy');
    for (var i = 0; i < 15; i++) {
      s.appendChild(K.n('rect', { x: 238 + (i % 5) * 20, y: 66 + Math.floor(i / 5) * 18,
        width: 15, height: 13, rx: 2, fill: i % 3 ? T.ink : T.amber }));
    }
    K.mono(s, 238, 128, '15 categories', { size: 12, color: C.ink3 });

    /* severity ladder */
    K.panel(s, 448, 32, 192, 120);
    K.label(s, 462, 54, 'severity scale');
    ['low', 'med', 'high', 'crit'].forEach(function (lv, j) {
      K.bar(s, 462, 66 + j * 16, 100, 8, (j + 1) / 4, { color: C.amber });
      K.mono(s, 574, 74 + j * 16, lv, { size: 10.5, color: C.ink3 });
    });

    [95, 319, 543].forEach(function (x) { K.arrow(s, x, 158, 320, 220, { curve: true }); });

    K.panel(s, 130, 232, 380, 100, { stroke: C.blue, fill: T.blue });
    K.label(s, 148, 256, 'one question, in that dataset own terms', { color: C.blue });
    K.mono(s, 148, 284, '<Query>: does this content promote violence?', { size: 12.5, color: C.ink });
    K.mono(s, 148, 308, 'answer:  yes | no', { size: 12.5, color: C.blue });

    K.big(s, 0, 400, 'no shared vocabulary', { size: 32, color: C.ink });
    K.text(s, 0, 428, 'so nothing has to be flattened to fit a common category set',
      { size: 14, color: C.ink3 });
    K.foot(s, 'The three schemes are drawn to show the shape of the problem. '
      + 'The report names label formats, category taxonomies and annotation conventions as the axes of disagreement.');
  };
})();
