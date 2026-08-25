window.SCENES = window.SCENES || {};

/* S_EVALGEN, act 4, beat 28. Accent lime.
   One evaluation sample on the bench, built in four steps.
     01 take two leaves that share a subcategory, a target and its sibling
     02 pin the single hand authored query that belongs to the target category
     03 rewrite one harmless seed sentence into two unsafe documents, one per category
     04 a second LLM re-reads the pair, and a mismatch is discarded rather than repaired
   Step 04 is live. Flip either check and the pair changes lane.
   Every string, name and count is read out of window.SS when the scene is built. */
window.SCENES['S_EVALGEN'] = function (root, api) {
  var SS = api.SS || {};
  var F = SS.fig4 || {};
  var H = SS.headline || {};
  var el = api.el;
  var svg = api.svg;

  var wrap = el('div', 'sc-s_evalgen');
  var sheet = el('div', 'sheet');
  wrap.appendChild(sheet);
  root.appendChild(wrap);

  /* ---------------- helpers ---------------- */

  function isNum(v) { return typeof v === 'number' && isFinite(v); }

  function splitCat(s) {
    var m = String(s == null ? '' : s).match(/^\s*(CAT\d+)\s*[-]\s*(.+)$/);
    if (m) return { id: m[1], name: m[2] };
    return { id: '', name: String(s == null ? 'n/a' : s) };
  }

  function locate(id) {
    var out = null;
    (SS.evalTaxonomy || []).forEach(function (sc) {
      (sc.subs || []).forEach(function (sub) {
        (sub.leaves || []).forEach(function (lf) {
          if (lf.id === id) out = { sc: sc, sub: sub, leaf: lf };
        });
      });
    });
    return out;
  }

  function stepHead(host, n, title, note) {
    var h = el('div', 'shead');
    h.appendChild(el('span', 'sn', n));
    h.appendChild(el('span', 'st', title));
    if (note) h.appendChild(el('span', 'snote', note));
    host.appendChild(h);
    return h;
  }

  function step(cls) {
    var s = el('div', 'step' + (cls ? ' ' + cls : ''));
    sheet.appendChild(s);
    return s;
  }

  var tgt = splitCat(F.category);
  var sib = splitCat(F.sibling);
  var locT = locate(tgt.id);
  var locS = locate(sib.id);
  var sameSub = !!(locT && locS && locT.sub === locS.sub);

  /* ---------------- header ---------------- */

  var hd = el('div', 'hd');
  hd.appendChild(el('span', 'tag', 'one eval sample'));
  hd.appendChild(el('span', 'hdsub', 'built, then checked by a second model'));
  sheet.appendChild(hd);

  sheet.appendChild(el('p', 'contrast',
    'Training builds iso-content, one document under two questions. Evaluation '
    + 'inverts it: one fixed question, two documents.'));

  /* ---------------- 01, the sibling pair ---------------- */

  var s1 = step('s1');
  var h1 = stepHead(s1, '01', 'take a sibling pair',
    sameSub ? 'two leaves of one subcategory' : 'target and sibling');
  h1.appendChild(el('span', 'crumb', locT
    ? (locT.sc.id + ' ' + locT.sc.name + '  /  ' + locT.sub.name)
    : 'evaluation taxonomy'));

  var pills = el('div', 'pills');
  [
    { c: tgt, role: 'target', on: true },
    { c: sib, role: 'sibling', on: false }
  ].forEach(function (p) {
    var b = el('span', 'leaf' + (p.on ? ' on' : ''));
    b.appendChild(el('span', 'lid', p.c.id || '  '));
    b.appendChild(el('span', 'lnm', p.c.name));
    b.appendChild(el('span', 'lrole', p.role));
    pills.appendChild(b);
  });
  s1.appendChild(pills);

  /* ---------------- 02, the pinned query ---------------- */

  var s2 = step('s2');
  stepHead(s2, '02', 'pin the query', 'authored by hand, one per category, never resampled');

  var qgrid = el('div', 'qgrid');

  var qnum = el('div', 'qnum');
  qnum.appendChild(el('div', 'big', isNum(H.evalQueries) ? String(H.evalQueries) : 'n/a'));
  qnum.appendChild(el('div', 'biglab', 'fixed queries'));
  var parts = [H.evalSupers, H.evalSubs, H.evalLeaves];
  var okParts = parts.every(isNum);
  var sum = okParts ? parts[0] + parts[1] + parts[2] : null;
  qnum.appendChild(el('div', 'bigsub',
    (okParts && sum === H.evalQueries)
      ? parts.join(' + ') + ' categories'
      : 'one per evaluation category'));
  qgrid.appendChild(qnum);

  var qbox = el('div', 'qbox');
  var qtop = el('div', 'krow');
  qtop.appendChild(el('span', 'k', 'query'));
  var lock = el('span', 'lock');
  var lico = svg('svg', {
    viewBox: '0 0 14 14', width: '11', height: '11',
    'aria-hidden': 'true', focusable: 'false'
  });
  lico.appendChild(svg('rect', {
    x: '2.4', y: '6.1', width: '9.2', height: '6.1', rx: '1.4',
    fill: 'none', stroke: 'currentColor', 'stroke-width': '1.3'
  }));
  lico.appendChild(svg('path', {
    d: 'M4.6 6.1 V4.4 a2.4 2.4 0 0 1 4.8 0 V6.1',
    fill: 'none', stroke: 'currentColor', 'stroke-width': '1.3'
  }));
  lock.appendChild(lico);
  lock.appendChild(el('span', null, 'same wording for every sample in ' + (tgt.id || 'this category')));
  qtop.appendChild(lock);
  qbox.appendChild(qtop);
  qbox.appendChild(el('p', 'qtxt', F.query || 'n/a'));

  var qgen = (SS.taxCompare || []).filter(function (r) {
    return /query generation/i.test(r.aspect || '');
  })[0];
  if (qgen && qgen.eval) qbox.appendChild(el('p', 'qcap', qgen.eval));
  qgrid.appendChild(qbox);
  s2.appendChild(qgrid);

  /* ---------------- 03, the two documents ---------------- */

  var s3 = step('s3');
  var gp = (SS.genPrompts || []).filter(function (p) {
    return /evaluation/i.test(p.name || '');
  })[0] || {};
  stepHead(s3, '03', 'rewrite one seed twice',
    gp.name ? gp.name.toLowerCase() : 'dual version rewriting');

  var seed = el('div', 'seed');
  seed.appendChild(el('span', 'k', 'source'));
  seed.appendChild(el('p', 'seedtxt', F.source || 'n/a'));
  seed.appendChild(el('span', 'tiny', 'harmless on its own'));
  s3.appendChild(seed);

  var fork = svg('svg', {
    viewBox: '0 0 800 34', preserveAspectRatio: 'none',
    'aria-hidden': 'true', focusable: 'false'
  });
  fork.setAttribute('class', 'fork');
  ['M400 0 C400 20 200 12 200 33', 'M400 0 C400 20 600 12 600 33'].forEach(function (d) {
    var p = svg('path', { d: d, fill: 'none' });
    p.setAttribute('class', 'fl');
    fork.appendChild(p);
  });
  s3.appendChild(fork);

  var docs = el('div', 'docs');
  [
    { role: 'positive version', cat: tgt, d: F.positive || {} },
    { role: 'negative version', cat: sib, d: F.negative || {} }
  ].forEach(function (o) {
    var lab = String(o.d.label == null ? '' : o.d.label);
    var yes = lab.toLowerCase() === 'yes';
    var c = el('div', 'doc' + (yes ? ' yes' : ' no'));

    var dh = el('div', 'dhead');
    dh.appendChild(el('span', 'drole', o.role));
    dh.appendChild(el('span', 'dcat', (o.cat.id || '') + ' ' + o.cat.name));
    c.appendChild(dh);

    c.appendChild(el('p', 'dtxt', o.d.document || 'n/a'));

    var df = el('div', 'dfoot');
    df.appendChild(el('span', 'unsafe', 'unsafe text'));
    var ans = el('span', 'ans');
    ans.appendChild(el('span', 'anslab', 'answer'));
    ans.appendChild(el('span', 'ansval', lab || 'n/a'));
    df.appendChild(ans);
    c.appendChild(df);
    docs.appendChild(c);
  });
  s3.appendChild(docs);

  /* ---------------- 04, verify or discard ---------------- */

  var s4 = step('s4');
  stepHead(s4, '04', 'cross verify, or discard', 'second LLM, different seed');

  var CHECKS = [
    { t: 'label is right', d: 'both answers hold under this query' },
    { t: 'sample is answerable', d: 'decidable from the document alone' }
  ];

  var crow = el('div', 'crow');
  crow.setAttribute('role', 'group');
  crow.setAttribute('aria-label', 'verifier checks, flip either one');

  CHECKS.forEach(function (ck) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'chk on';
    b.setAttribute('role', 'switch');
    b.setAttribute('aria-checked', 'true');
    b.setAttribute('aria-label', 'verifier check, ' + ck.t);

    var mark = svg('svg', {
      viewBox: '0 0 20 20', width: '17', height: '17',
      'aria-hidden': 'true', focusable: 'false'
    });
    mark.setAttribute('class', 'mk');
    var mp = svg('path', {
      d: 'M4 10.6 L8.2 14.8 L16 5.6', fill: 'none', stroke: 'currentColor',
      'stroke-width': '2.2', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'
    });
    mark.appendChild(mp);

    var txt = el('div', 'ctxt');
    txt.appendChild(el('span', 'ct', ck.t));
    txt.appendChild(el('span', 'cd', ck.d));

    var pill = el('span', 'cpill', 'agrees');

    b.appendChild(mark);
    b.appendChild(txt);
    b.appendChild(pill);
    b.addEventListener('click', function () {
      ck.pass = !ck.pass;
      paint();
    });
    ck.btn = b;
    ck.mark = mp;
    ck.pill = pill;
    ck.pass = true;
    crow.appendChild(b);
  });
  s4.appendChild(crow);

  /* the two lanes out of the verifier */

  var outr = el('div', 'outr');

  var rt = svg('svg', {
    viewBox: '0 0 800 110', preserveAspectRatio: 'xMidYMid meet',
    'aria-hidden': 'true', focusable: 'false'
  });
  rt.setAttribute('class', 'rt');

  function box(x, y, w, h, cls, label) {
    var g = svg('g', {});
    g.setAttribute('class', cls);
    g.appendChild(svg('rect', { x: x, y: y, width: w, height: h, rx: '9' }));
    var t = svg('text', { x: x + w / 2, y: y + h / 2 + 3.2, 'text-anchor': 'middle' });
    t.textContent = label;
    g.appendChild(t);
    return g;
  }

  var KEEP_D = 'M112 47 H360 C400 47 400 18 440 18 H612';
  var DROP_D = 'M112 47 H360 C400 47 400 76 440 76 H612';

  rt.appendChild(box(0, 30, 104, 34, 'ent', 'THE PAIR'));

  var keepDim = svg('path', { d: KEEP_D, fill: 'none' });
  keepDim.setAttribute('class', 'lane');
  var dropDim = svg('path', { d: DROP_D, fill: 'none' });
  dropDim.setAttribute('class', 'lane');
  rt.appendChild(keepDim);
  rt.appendChild(dropDim);

  var live = svg('path', { d: KEEP_D, fill: 'none' });
  live.setAttribute('class', 'live');
  rt.appendChild(live);

  var head = svg('polygon', { points: '604,11 620,18 604,25' });
  head.setAttribute('class', 'head');
  rt.appendChild(head);

  var gate = svg('g', {});
  gate.setAttribute('class', 'gate');
  gate.appendChild(svg('polygon', { points: '268,21 294,47 268,73 242,47' }));
  var gl = svg('text', { x: '268', y: '99', 'text-anchor': 'middle' });
  gl.textContent = 'VERIFIER';
  gate.appendChild(gl);
  rt.appendChild(gate);

  var keepBox = box(616, 2, 184, 32, 'dest keep', 'EVALUATION SET');
  var dropBox = box(616, 60, 184, 32, 'dest drop', 'DISCARDED');
  rt.appendChild(keepBox);
  rt.appendChild(dropBox);

  var dot = svg('circle', { cx: '112', cy: '47', r: '5' });
  dot.setAttribute('class', 'dot');
  rt.appendChild(dot);

  outr.appendChild(rt);

  var vd = el('div', 'vd');
  vd.setAttribute('aria-live', 'polite');
  var vrow = el('div', 'vrow');
  var vbig = el('div', 'vbig', 'keep');
  var vgl = svg('svg', {
    viewBox: '0 0 34 34', width: '28', height: '28',
    'aria-hidden': 'true', focusable: 'false'
  });
  vgl.setAttribute('class', 'vgl');
  var vgp = svg('path', {
    d: 'M7 18.4 L14.2 25.6 L27 10.4', fill: 'none', stroke: 'currentColor',
    'stroke-width': '3', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'
  });
  vgl.appendChild(vgp);
  vrow.appendChild(vbig);
  vrow.appendChild(vgl);
  vd.appendChild(vrow);
  var vsay = el('p', 'vsay', '');
  vd.appendChild(vsay);
  outr.appendChild(vd);

  s4.appendChild(outr);

  /* ---------------- notes ---------------- */

  sheet.appendChild(el('p', 'honest',
    'Sample text, labels and categories are the report Figure 4 iso-query example and the '
    + 'evaluation taxonomy, read from the data file. The switches are a schematic of the stated '
    + 'rule, not a live model call. No discard rate is reported.'));

  /* ---------------- state ---------------- */

  var kept = true;

  function paint() {
    kept = CHECKS.every(function (c) { return c.pass; });

    CHECKS.forEach(function (c) {
      c.btn.classList.toggle('on', c.pass);
      c.btn.setAttribute('aria-checked', c.pass ? 'true' : 'false');
      c.pill.textContent = c.pass ? 'agrees' : 'mismatch';
      c.mark.setAttribute('d', c.pass
        ? 'M4 10.6 L8.2 14.8 L16 5.6'
        : 'M5.4 5.4 L14.6 14.6 M14.6 5.4 L5.4 14.6');
    });

    live.setAttribute('d', kept ? KEEP_D : DROP_D);
    head.setAttribute('points', kept ? '604,11 620,18 604,25' : '604,69 620,76 604,83');
    rt.classList.toggle('r-keep', kept);
    rt.classList.toggle('r-drop', !kept);
    keepBox.classList.toggle('act', kept);
    dropBox.classList.toggle('act', !kept);

    vd.classList.toggle('v-keep', kept);
    vd.classList.toggle('v-drop', !kept);
    vbig.textContent = kept ? 'keep' : 'discard';
    vgp.setAttribute('d', kept
      ? 'M7 18.4 L14.2 25.6 L27 10.4'
      : 'M9 9 L25 25 M25 9 L9 25');
    vsay.textContent = kept
      ? 'Both rows join the benchmark, one yes and one no against the same fixed query.'
      : 'The whole pair is thrown away. Nothing is edited back into shape.';

    if (api.reduce) place(1);
  }

  function place(u) {
    var L;
    try { L = live.getTotalLength(); } catch (e) { return; }
    if (!L || !live.getPointAtLength) return;
    var p = live.getPointAtLength(Math.max(0, Math.min(1, u)) * L);
    dot.setAttribute('cx', String(p.x));
    dot.setAttribute('cy', String(p.y));
  }

  paint();

  /* ---------------- fit the fixed sheet into the stage ---------------- */

  var flow = matchMedia('(max-width: 70rem)');

  function fit() {
    if (flow.matches) {
      wrap.classList.add('flow');
      sheet.style.transform = '';
      return;
    }
    wrap.classList.remove('flow');
    var w = root.clientWidth, h = root.clientHeight;
    if (!w || !h) return;
    var k = Math.min(1, w / 820, h / 760);
    sheet.style.transform = 'scale(' + (Math.round(k * 1000) / 1000) + ')';
  }
  fit();
  if (window.ResizeObserver) { new ResizeObserver(fit).observe(root); }
  else { window.addEventListener('resize', fit); }
  if (flow.addEventListener) flow.addEventListener('change', fit);

  return {
    start: function () { fit(); },
    stop: function () {},
    tick: function (t) {
      place((t % 2.6) / 2.6);
    }
  };
};
