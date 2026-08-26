/* Act II, the data pipeline. Amber. One artifact per figure. */
window.SCENES = window.SCENES || {};

(function () {
  var K = window.KIT, C = K.C, T = K.TINT, A = C.amber;

  /* 08 the training mix */
  window.SCENES.S_MOUNTAIN = function (root, api) {
    var s = K.board(root, { alt: 'The training mix at true proportion.' });
    K.head(s, 'Fifty-four million rows', 'one mark per hundred thousand training samples');
    var h = api.SS.headline, PER = 0.1;
    var parts = [
      { n: h.openSourceText, lab: 'public datasets, text', col: A },
      { n: h.syntheticText, lab: 'synthetic contrastive text', col: C.purple },
      { n: h.multimodalSamples, lab: 'multimodal, image plus text', col: C.blue }
    ];
    K.big(s, 0, 62, h.totalSamples + 'M', { size: 72, color: C.ink });
    K.text(s, 0, 100, 'samples in the training mix', { size: 14, color: C.ink3 });

    var x = 0, y = 130, col = 0;
    parts.forEach(function (p) {
      var marks = Math.round(p.n / PER);
      for (var i = 0; i < marks; i++) {
        s.appendChild(K.n('rect', { x: x + col % 44 * 14.5, y: y + Math.floor(col / 44) * 13,
          width: 9, height: 9, rx: 1.5, fill: p.col }));
        col++;
      }
    });
    var rows = Math.ceil(col / 44), ly = y + rows * 13 + 24;
    parts.forEach(function (p, i) {
      s.appendChild(K.n('rect', { x: 0, y: ly + i * 26 - 9, width: 10, height: 10, rx: 2, fill: p.col }));
      K.text(s, 18, ly + i * 26, p.lab, { size: 13.5, color: C.ink2 });
      K.mono(s, 640, ly + i * 26, p.n + 'M', { size: 14, color: C.ink, anchor: 'end' });
    });
    var ratio = ((h.openSourceText + h.syntheticText) / h.multimodalSamples).toFixed(1);
    K.text(s, 0, ly + 3 * 26 + 14, 'Text outweighs image data ' + ratio + ' to 1.', { size: 14, color: C.ink2 });
    K.foot(s, 'Bucket sizes are the report counts. A mark stands for a fixed number of rows, not for any particular rows.');
  };

  /* 09 one processor per dataset */
  window.SCENES.S_PROCESSOR = function (root, api) {
    var s = K.board(root, { alt: 'Each dataset gets its own hand-written processor.' });
    K.head(s, 'One processor per dataset', 'hand-written, from that dataset\'s own documentation');
    var inv = (api.SS.benchInventory || []).slice(0, 5);
    K.label(s, 0, 14, 'each source, its own processor');
    inv.forEach(function (r, i) {
      var y = 34 + i * 46;
      K.panel(s, 0, y, 210, 36);
      K.mono(s, 14, y + 23, String(r[0]), { size: 12.5, color: C.ink });
      K.arrow(s, 218, y + 18, 268, y + 18, { color: C.line });
      K.panel(s, 272, y, 92, 36, { fill: T.amber, stroke: A });
      K.mono(s, 288, y + 23, 'processor', { size: 11, color: A });
      K.arrow(s, 372, y + 18, 430, 146, { curve: true, color: C.line });
    });
    K.panel(s, 430, 100, 210, 100, { stroke: A, fill: T.amber });
    K.label(s, 446, 124, 'one shared row', { color: A });
    ['<Instruct>', '<Query>', '<Document>'].forEach(function (f, i) {
      K.mono(s, 446, 148 + i * 19, f, { size: 12, color: C.ink });
    });
    K.label(s, 0, 290, 'what a processor encodes');
    ['labelling logic', 'category mappings', 'instruction templates'].forEach(function (t, i) {
      K.chip(s, i * 172, 302, t, { size: 11.5, color: C.ink2 });
    });
    K.foot(s, 'The five sources shown are examples. The report does not say how many processors there are.');
  };

  /* 10 strictness tiers */
  window.SCENES.S_STRICT = function (root, api) {
    var s = K.board(root, { alt: 'Three strictness tiers move the decision boundary.' });
    K.head(s, 'Strictness moves the line', 'the same content, three intended decision boundaries');
    var tiers = api.SS.strictness || [];
    K.label(s, 0, 14, 'strictness tier, assigned per dataset');
    var body = K.n('g', {}); s.appendChild(body);
    var CUT = [0.30, 0.52, 0.74];

    K.switcher(s, 0, 26, tiers.map(function (t) { return t.level; }), function (i) {
      body.innerHTML = '';
      var t = tiers[i];
      K.big(body, 0, 112, t.level, { size: 44, color: A });
      K.text(body, 0, 142, t.domains, { size: 14.5, color: C.ink2 });
      K.text(body, 0, 166, t.rationale + '.', { size: 14, color: C.ink3 });

      K.label(body, 0, 216, 'benign', { color: C.ink3 });
      K.label(body, 640, 216, 'clearly harmful', { color: C.ink3, anchor: 'end' });
      s.appendChild(body);
      body.appendChild(K.n('rect', { x: 0, y: 228, width: 640, height: 10, rx: 5, fill: T.ink }));
      for (var d = 0; d < 42; d++) {
        var px = 12 + d * 14.6, flag = px / 640 > CUT[i];
        body.appendChild(K.n('circle', { cx: px, cy: 262, r: 4.5,
          fill: flag ? C.red : C.teal, opacity: .8 }));
      }
      body.appendChild(K.n('rect', { x: CUT[i] * 640 + 5, y: 214, width: 2, height: 68, fill: A }));
      K.label(body, CUT[i] * 640 + 13, 296, 'flag from here', { color: A });
    }, { tint: 'amber' });
    K.foot(s, 'The boundary position and the sample cloud are drawn for this figure. The report gives the tiers and their domains, not a numeric threshold.');
  };

  /* 11 three query shapes */
  window.SCENES.S_QTYPES = function (root, api) {
    var s = K.board(root, { alt: 'Three kinds of question about the same document.' });
    K.head(s, 'Three shapes of question', 'category specific, binary, and refusal detection');
    var qs = api.SS.queryTypes || [];
    var doc = ((api.SS.fig2 || [])[0] || {}).document || '';
    K.label(s, 0, 14, 'one document');
    K.panel(s, 0, 24, 640, 74, { fill: T.ink });
    doc.split('\n').slice(0, 4).forEach(function (ln, i) {
      K.mono(s, 16, 46 + i * 17, ln, { size: 12.5, color: C.ink });
    });
    K.label(s, 0, 128, 'three kinds of question');
    qs.forEach(function (q, i) {
      var y = 146 + i * 96;
      K.panel(s, 0, y, 640, 78);
      K.label(s, 16, y + 22, q.name, { color: A });
      K.mono(s, 16, y + 44, q.examples[0], { size: 12.5, color: C.ink });
      K.text(s, 16, y + 64, q.sub, { size: 12.5, color: C.ink3 });
    });
    K.foot(s, 'Example questions are quoted from the report\'s section on query templates.');
  };

  /* 12 seven document formats */
  window.SCENES.S_FORMATS = function (root, api) {
    var s = K.board(root, { alt: 'The same exchange in seven delimiter conventions.' });
    K.head(s, 'Seven ways to write a conversation', 'every one of them a convention some real system uses');
    var f = api.SS.formats || [];
    var ex = (api.SS.fig2 || [])[0] || {};
    var m = String(ex.document || '').match(/\[User\]\s*([\s\S]*?)\s*\[Assistant\]\s*([\s\S]*)$/);
    var P = m ? m[1] : 'How can I make a weapon at home?';
    var R = m ? m[2] : "I can't help with that request.";
    K.label(s, 0, 14, 'same exchange, seven conventions');
    var body = K.n('g', {}); s.appendChild(body);
    K.switcher(s, 0, 26, f.map(function (x) { return x.family.split(' ')[0]; }), function (i) {
      body.innerHTML = '';
      K.big(body, 0, 116, f[i].family, { size: 34, color: A });
      K.panel(body, 0, 140, 640, 96, { fill: T.ink });
      String(f[i].tpl).replace('{prompt}', P).replace('{response}', R)
        .split('\n').forEach(function (ln, j) {
          K.mono(body, 18, 172 + j * 22, ln, { size: 13.5, color: C.ink });
        });
      s.appendChild(body);
    }, { tint: 'amber' });
    K.foot(s, 'Prompt and response pairs draw one of these families at random per training example.');
  };

  /* 13 the pools multiply */
  window.SCENES.S_COMBO = function (root, api) {
    var s = K.board(root, { alt: 'The pools a single row can be drawn from.' });
    K.head(s, 'One row, many views', 'strictness, format and query, each drawn per sample');
    var SSd = api.SS;
    var nS = (SSd.strictness || []).length, nF = (SSd.formats || []).length,
        nQ = (SSd.queryTypes || []).length;
    K.label(s, 0, 14, 'pools in this data file');
    [[nS, 'strictness tiers'], [nF, 'document formats'], [nQ, 'query types']].forEach(function (p, i) {
      K.big(s, i * 214, 84, String(p[0]), { size: 58, color: A });
      K.text(s, i * 214, 124, p[1], { size: 13.5, color: C.ink2 });
      if (i < 2) K.text(s, i * 214 + 150, 76, '×', { size: 26, color: C.ink3 });
    });
    K.panel(s, 0, 140, 640, 76, { fill: T.amber, stroke: A });
    K.big(s, 20, 194, String(nS * nF * nQ), { size: 44, color: A });
    K.text(s, 110, 188, 'combinations of these pools', { size: 15, color: C.ink2 });
    K.text(s, 0, 250, 'The same content shows up under a different framing on every epoch,',
      { size: 14, color: C.ink2 });
    K.text(s, 0, 272, 'so there is no fixed wording for the model to attach a label to.',
      { size: 14, color: C.ink2 });
    K.foot(s, 'This counts the pools carried in this page\'s data file. The report describes per-dataset template pools and does not publish a combination count.');
  };

  /* 14 positives at three granularities */
  window.SCENES.S_POS = function (root, api) {
    var s = K.board(root, { alt: 'One harmful document answers yes to three questions.' });
    K.head(s, 'Ask it three ways, get three yeses', 'one violation, at three levels of specificity');
    var f3 = api.SS.fig3 || {};
    K.label(s, 0, 14, 'one harmful document');
    K.panel(s, 0, 24, 640, 72, { fill: T.ink });
    K.para(s, 16, 48, f3.document || '', 68, { size: 13, color: C.ink, lh: 19 });
    K.label(s, 0, 126, 'three questions, broad to narrow');
    var qs = [
      ['coarse binary', 'Is this content unsafe?'],
      ['category specific', f3.positive ? f3.positive.query : ''],
      ['target group', 'Does this content promote violence toward a specific group?']
    ];
    qs.forEach(function (q, i) {
      var y = 146 + i * 74;
      K.panel(s, 0, y, 540, 58);
      K.label(s, 16, y + 20, q[0], { color: A });
      K.para(s, 16, y + 40, q[1], 58, { size: 12.5, color: C.ink, lh: 16 });
      K.big(s, 566, y + 42, 'yes', { size: 28, color: C.red });
    });
    K.foot(s, 'The coarse and target-group questions are phrased for this figure. The category question is the report\'s own.');
  };

  /* 15 three negative strategies */
  window.SCENES.S_NEG = function (root, api) {
    var s = K.board(root, { alt: 'Three ways to build a negative.' });
    K.head(s, 'Unsafe, and the answer is still no', 'category based, demographic based, and safe content');
    var f3 = api.SS.fig3 || {};
    var STRAT = [
      ['category based', f3.document, f3.negative ? f3.negative.query : '',
       'unsafe document, absent category, so no'],
      ['demographic based', f3.document,
       'Does this content target people on the basis of disability?',
       'unsafe document, unrelated group, so no'],
      ['safe content', 'Can people really control gravitational forces with their minds?',
       'Is this content unsafe?', 'safe document, the easy case']
    ];
    K.label(s, 0, 14, 'negative strategy');
    var body = K.n('g', {}); s.appendChild(body);
    K.switcher(s, 0, 26, ['category', 'demographic', 'safe content'], function (i) {
      body.innerHTML = '';
      var st = STRAT[i];
      K.label(body, 0, 84, 'document', { color: C.ink3 });
      K.panel(body, 0, 94, 640, 66, { fill: T.ink });
      K.para(body, 16, 118, st[1] || '', 66, { size: 13, color: C.ink, lh: 18 });
      K.label(body, 0, 190, 'query', { color: A });
      K.panel(body, 0, 200, 640, 58, { fill: T.amber, stroke: A });
      K.para(body, 16, 224, st[2] || '', 64, { size: 12.5, color: C.ink, lh: 17 });
      K.verdict(body, 0, 330, false, { size: 54 });
      K.text(body, 150, 316, st[3], { size: 14, color: C.ink2 });
      s.appendChild(body);
    }, { tint: 'amber' });
    K.foot(s, 'The document and the category query are the report\'s own. The demographic and safe-content examples are phrased for this figure.');
  };

  /* 16 rebalance and filter */
  window.SCENES.S_FILTER = function (root, api) {
    var s = K.board(root, { alt: 'Positives duplicated, disagreeing labels dropped.' });
    K.head(s, 'Rebalance, then throw away the wrong labels', 'public safety datasets contain incorrect labels');
    K.label(s, 0, 14, 'contrastive construction skews negative');
    var y = 36;
    for (var i = 0; i < 12; i++) {
      s.appendChild(K.n('rect', { x: i * 30, y: y, width: 22, height: 22, rx: 3,
        fill: i < 3 ? C.red : T.ink, stroke: C.line }));
    }
    K.text(s, 0, y + 46, 'more negatives than positives, since any absent category is a negative',
      { size: 13.5, color: C.ink3 });

    K.label(s, 0, 116, 'so each positive is duplicated, with fresh wording');
    for (i = 0; i < 9; i++) {
      s.appendChild(K.n('rect', { x: i * 30, y: 130, width: 22, height: 22, rx: 3,
        fill: C.red, opacity: i < 3 ? 1 : .45, stroke: C.line }));
    }
    K.arrow(s, 100, 168, 100, 190, { color: C.line });

    K.label(s, 0, 226, 'then an llm re-checks every label');
    K.panel(s, 0, 238, 640, 96);
    var kept = [1, 1, 0, 1, 1, 1, 0, 1];
    kept.forEach(function (k, j) {
      var x = 18 + j * 76;
      s.appendChild(K.n('rect', { x: x, y: 262, width: 56, height: 30, rx: 4,
        fill: k ? T.ink : 'none', stroke: k ? C.line : C.red,
        'stroke-dasharray': k ? null : '3 3' }));
      if (!k) {
        s.appendChild(K.n('path', { d: 'M' + x + ' 292 L' + (x + 56) + ' 262',
          stroke: C.red, 'stroke-width': 1.4 }));
      }
    });
    K.text(s, 18, 320, 'dropped where the dataset label and the model disagree, at the binary and the per-category level',
      { size: 12.5, color: C.ink3 });
    K.foot(s, 'The block counts are drawn for this figure. The report states the mechanism but publishes no ratio and no drop rate.');
  };

  /* 17 one call, two rows */
  window.SCENES.S_REWRITE = function (root, api) {
    var s = K.board(root, { alt: 'One LLM call produces two contradictory training rows.' });
    K.head(s, 'One call in, two rows out', 'the same document, labelled yes and no');
    var f3 = api.SS.fig3 || {};
    K.label(s, 0, 14, 'into one llm call');
    ['safe source text', 'target category', 'sibling category'].forEach(function (t, i) {
      K.chip(s, i * 168, 26, t, { size: 11.5, color: C.ink2, w: 156 });
    });
    [78, 246, 414].forEach(function (x) { K.arrow(s, x, 54, 300, 96, { curve: true }); });
    K.panel(s, 200, 100, 240, 44, { fill: T.amber, stroke: A });
    K.mono(s, 224, 128, 'one generation call', { size: 13, color: A });
    K.arrow(s, 320, 148, 150, 196, { curve: true });
    K.arrow(s, 320, 148, 490, 196, { curve: true });

    [[0, f3.positive, 'yes', C.red], [330, f3.negative, 'no', C.teal]].forEach(function (p) {
      var x = p[0], q = p[1] || {};
      K.panel(s, x, 202, 310, 150, { stroke: p[3] === C.red ? C.red : C.line });
      K.label(s, x + 16, 226, q.role || '', { color: p[3] });
      K.para(s, x + 16, 248, q.query || '', 33, { size: 11.5, color: C.ink, lh: 15 });
      K.label(s, x + 16, 316, 'identical document', { color: C.ink3, size: 9.5 });
      K.big(s, x + 16, 344, p[2], { size: 26, color: p[3] });
    });
    K.big(s, 0, 400, 'two rows', { size: 36, color: C.ink });
    K.text(s, 168, 396, 'from a single call, disagreeing on purpose', { size: 15, color: C.ink2 });
    K.foot(s, 'The queries and the rewritten document are the report\'s own, from its worked example.');
  };

  /* 18 free positives up the tree */
  window.SCENES.S_ANCESTOR = function (root, api) {
    var s = K.board(root, { alt: 'A leaf violation counts at every level above it.' });
    K.head(s, 'Free positives, all the way up', 'a leaf violation is a violation at every level above it');
    var h = api.SS.headline;
    var sc = (api.SS.evalTaxonomy || [])[0] || {};
    var sub = (sc.subs || [])[0] || {};
    var leaf = (sub.leaves || [])[0] || {};
    var path = [
      ['leaf', leaf.name || 'Physical Violence'],
      ['subcategory', sub.name || 'Direct Violence'],
      ['super class', sc.name || 'Physical Harm']
    ];
    K.label(s, 0, 14, 'one rewrite, one row per level');
    path.forEach(function (p, i) {
      var y = 36 + i * 96;
      K.panel(s, 0, y, 300, 68, { stroke: A, fill: T.amber });
      K.label(s, 16, y + 24, p[0], { color: A });
      K.mono(s, 16, y + 46, p[1], { size: 14, color: C.ink });
      if (i < 2) K.arrow(s, 150, y + 72, 150, y + 92, { color: C.line });
      K.arrow(s, 308, y + 34, 356, y + 34, { color: C.line });
      K.panel(s, 362, y + 12, 278, 44);
      K.mono(s, 378, y + 40, 'query about ' + p[0] + '  ', { size: 11.5, color: C.ink3 });
      K.big(s, 596, y + 44, 'yes', { size: 20, color: C.red });
    });
    K.big(s, 0, 356, '1 call', { size: 34, color: C.ink });
    K.text(s, 130, 352, 'three labelled rows, no extra generation cost', { size: 15, color: C.ink2 });
    K.foot(s, 'The names shown are from the evaluation taxonomy, used here to illustrate the shape. '
      + 'The training taxonomy is a different tree of ' + h.trainSupers + ' super classes and '
      + h.trainLeaves + ' leaves, which the report does not publish in full.');
  };

  /* 19 images cannot be rewritten */
  window.SCENES.S_IMGSCARCE = function (root, api) {
    var s = K.board(root, { alt: 'The text trick does not transfer to images.' });
    K.head(s, 'You cannot rewrite an image', 'so the negatives get borrowed instead');
    K.label(s, 0, 14, 'text');
    K.panel(s, 0, 26, 280, 58, { fill: T.ink });
    K.mono(s, 16, 60, 'safe sentence', { size: 13, color: C.ink });
    K.arrow(s, 288, 55, 344, 55, { color: A });
    K.panel(s, 352, 26, 288, 58, { fill: T.amber, stroke: A });
    K.mono(s, 368, 60, 'llm rewrites it unsafe', { size: 13, color: A });

    K.label(s, 0, 122, 'images');
    K.panel(s, 0, 134, 280, 58, { fill: T.ink });
    ['landscape', 'interior', 'objects'].forEach(function (w, i) {
      K.photo(s, 16 + i * 62, 142, 54, 42, w, { r: 4 });
    });
    K.arrow(s, 288, 163, 344, 163, { color: C.red, dash: '4 4' });
    K.panel(s, 352, 134, 288, 58, { stroke: C.red, fill: T.red });
    K.mono(s, 368, 168, 'cannot be generated this way', { size: 13, color: C.red });

    K.label(s, 0, 232, 'where the negatives come from');
    K.panel(s, 0, 244, 640, 96);
    K.text(s, 18, 274, 'General-purpose classification and object-detection datasets supply', { size: 13.5, color: C.ink2 });
    K.text(s, 18, 294, 'a large pool of naturally safe images to serve as negatives.', { size: 13.5, color: C.ink2 });
    var TW = ['landscape', 'interior', 'objects'];
    for (i = 0; i < 16; i++) {
      K.photo(s, 18 + i * 38, 306, 32, 26, TW[i % 3], { r: 3 });
    }
    K.foot(s, 'Images are drawn as neutral placeholders. Nothing unsafe is depicted anywhere in this guide.');
  };

  /* 20 an image actually going through moderation */
  window.SCENES.S_IMGQUERY = function (root, api) {
    var s = K.board(root, { alt: 'One picture moderated against a direct and an inverse query.' });
    K.head(s, 'One picture, moderated twice', 'the same picture, a direct query and an inverse one');
    var h = api.SS.headline;
    var VIEWS = [
      { tag: 'direct', q: 'Does this image contain violence?', a: false,
        note: 'the plain framing. Nothing here matches, so no.' },
      { tag: 'inverse, ~' + h.inversePct + '% of the pool',
        q: 'Is this image safe from violence?', a: true,
        note: 'the same picture, and now the correct answer is yes.' }
    ];

    /* the picture, once, large, on the left */
    K.label(s, 0, 18, 'the document');
    K.photo(s, 0, 30, 246, 184, 'landscape');
    K.label(s, 0, 232, 'a rendered stand-in, nothing real is shown', { size: 8.6 });

    var body = K.n('g', {});
    K.label(s, 282, 18, 'the question');
    K.switcher(s, 282, 30, VIEWS.map(function (v) { return v.tag; }), function (i) {
      body.innerHTML = '';
      var v = VIEWS[i];
      K.panel(body, 282, 74, 358, 78, { stroke: A, fill: T.amber });
      K.para(body, 300, 102, v.q, 36, { size: 13.5, color: C.ink, lh: 19 });
      K.arrow(body, 461, 158, 461, 186, { color: C.line });
      K.panel(body, 282, 194, 358, 92);
      K.verdict(body, 300, 246, v.a, { size: 46 });
      K.para(body, 386, 232, v.note, 30, { size: 12.5, color: C.ink2, lh: 17 });
      s.appendChild(body);
    }, { tint: 'amber' });
    s.appendChild(body);

    K.callout(s, 0, 306, 640,
      h.imageQueryPhrasings + ' query phrasings from a fixed ' + h.imageSubcats
      + '-subcategory visual taxonomy. Pictures cannot be rewritten, so all the variety goes '
      + 'into the question.', { color: A, tint: T.amber, cols: 68 });
    K.foot(s, 'The two questions are phrased for this figure. The phrasing count, the taxonomy size and the inverse share are the report\'s own.');
  };

  /* 21 asymmetric cut */
  window.SCENES.S_RERANK = function (root, api) {
    var s = K.board(root, { alt: 'Two different reranker thresholds for two pools.' });
    K.head(s, 'Two pools, two different bars', 'switch to a single shared cut and watch what it costs');
    var body = K.n('g', {}); s.appendChild(body);
    var VIOL = [], NEG = [], i;
    for (i = 0; i < 26; i++) VIOL.push(0.18 + (i * 0.37) % 0.72);
    for (i = 0; i < 60; i++) NEG.push(0.30 + (i * 0.53) % 0.68);

    K.label(s, 0, 14, 'reranker agreement score');
    K.switcher(s, 0, 26, ['asymmetric, as built', 'one symmetric cut'], function (mode) {
      body.innerHTML = '';
      var cv = mode ? 0.62 : 0.36, cn = 0.62;
      [[VIOL, 'violation pool, rare', cv, C.red, 96],
       [NEG, 'negative pool, abundant', cn, C.teal, 236]].forEach(function (p) {
        var pts = p[0], y = p[4], cut = p[2];
        K.label(body, 0, y - 14, p[1], { color: C.ink3 });
        body.appendChild(K.n('rect', { x: 0, y: y, width: 640, height: 46, rx: 6, fill: T.ink }));
        var kept = 0;
        pts.forEach(function (v, j) {
          var keep = v >= cut;
          if (keep) kept++;
          body.appendChild(K.n('circle', { cx: v * 640, cy: y + 14 + (j % 3) * 11, r: 4,
            fill: keep ? p[3] : 'none', stroke: p[3], opacity: keep ? .85 : .35 }));
        });
        body.appendChild(K.n('rect', { x: cut * 640, y: y - 6, width: 2, height: 58, fill: C.ink }));
        K.mono(body, cut * 640 + 8, y - 10, cut.toFixed(2), { size: 11.5, color: C.ink });
        K.mono(body, 640, y + 70, kept + ' kept, ' + (pts.length - kept) + ' dropped',
          { size: 12.5, color: C.ink2, anchor: 'end' });
      });
      K.text(body, 0, 344, mode
        ? 'One cut for both pools throws away rare violations that are expensive to find.'
        : 'A looser cut on the rare pool keeps samples the strict cut would discard.',
        { size: 14, color: mode ? C.red : C.ink2 });
      s.appendChild(body);
    }, { tint: 'amber' });
    K.foot(s, 'Sample positions and threshold values are drawn for this figure. The report states that the thresholds are asymmetric but does not publish them.');
  };
})();
