/* S_IMGQUERY, Act II, section 3.4.
   Query mutation on the image side. The pool numbers come from SS.headline
   (imageQueryPhrasings, imageSubcats, inversePct). The direct question and its
   label are the paper's multimodal example, SS.fig2[1]. The inverse question is
   built from that same string by flipping the polarity, and the correct answer
   flips with it. One abstract glyph sits between the two, unchanged.
   Built on a fixed 820x760 sheet and scaled into the stage, like the other
   Act II scenes, so nothing clips. */
window.SCENES = window.SCENES || {};
window.SCENES['S_IMGQUERY'] = function (root, api) {
  var SS = api.SS || {};
  var el = api.el, svg = api.svg;

  var H = SS.headline || {};
  var MM = (SS.fig2 || [])[1] || {};

  var phrasings = H.imageQueryPhrasings;
  var subcats = H.imageSubcats;
  var invPct = H.inversePct;
  var dirPct = (invPct == null) ? null : (100 - Number(invPct));

  /* ---------- split the paper's question into a mutable head and a shared tail ----- */
  var direct = String(MM.query || '');
  var m = direct.match(/^Does this (content|image|text) contain\s+(.+?)\s*\?$/i);
  var noun = m ? m[1] : 'content';
  var tail = m ? (m[2] + '?') : direct;
  var headDirect = m ? ('Does this ' + noun + ' contain') : 'Does this content contain';
  var headInverse = 'Is this ' + noun + ' free of';

  /* the paper prints the label for the direct pairing. the inverse pairing is the
     same picture asked the other way round, so the correct answer is the opposite. */
  var labDirect = String(MM.label || 'no').toLowerCase();
  var labInverse = (labDirect === 'yes') ? 'no' : 'yes';

  var VIEWS = [
    {
      tag: 'direct',
      share: (dirPct == null) ? 'the rest of the pool' : ('about ' + dirPct + '% of the pool'),
      head: headDirect,
      tail: m ? tail : '',
      whole: m ? '' : direct,
      polarity: 'asks whether the harm is present',
      ans: labDirect,
      why: 'nothing prohibited is in the frame, so the honest answer is ' + labDirect + '.'
    },
    {
      tag: 'inverse',
      share: (invPct == null) ? 'part of the pool' : ('about ' + invPct + '% of the pool'),
      head: headInverse,
      tail: m ? tail : '',
      whole: m ? '' : ('Is this content free of the above?'),
      polarity: 'asks whether the image is clear of it',
      ans: labInverse,
      why: 'the same empty frame satisfies the inverse, so the answer is ' + labInverse + '.'
    }
  ];

  var wrap = el('div', 'sc-s_imgquery');
  var sheet = el('div', 'sheet');
  wrap.appendChild(sheet);

  /* ---------------- header ---------------- */
  var hd = el('div', 'hd');
  hd.appendChild(el('span', 'eyy', 'query mutation · the image pool'));
  hd.appendChild(el('span', 'hint', 'click or tab a question to pin it'));
  sheet.appendChild(hd);

  /* ---------------- the pool ---------------- */
  var pool = el('div', 'pool');
  var stats = el('div', 'stats');

  function stat(v, k, extra) {
    var s = el('div', 'st');
    s.appendChild(el('span', 'stv', v == null ? 'n/a' : String(v)));
    s.appendChild(el('span', 'stk', k));
    if (extra) s.appendChild(extra);
    return s;
  }

  stats.appendChild(stat(phrasings, 'query phrasings generated for images'));

  var ticks = el('span', 'ticks');
  for (var i = 0; i < (Number(subcats) || 0); i++) ticks.appendChild(el('i', i === 0 ? 'lit' : null));
  ticks.setAttribute('role', 'img');
  ticks.setAttribute('aria-label', (subcats == null ? 'The' : subcats)
    + ' visual subcategories. The lit one is the subcategory in play below.');
  stats.appendChild(stat(subcats, 'subcategories in the visual taxonomy', ticks));

  stats.appendChild(stat(invPct == null ? null : invPct + '%', 'of them are inverse formulations'));
  pool.appendChild(stats);

  var bar = el('div', 'bar');
  var segD = el('span', 'seg d');
  var segI = el('span', 'seg i');
  segD.style.width = (dirPct == null ? 70 : dirPct) + '%';
  segI.style.width = (invPct == null ? 30 : invPct) + '%';
  segD.appendChild(el('b', null, 'direct ' + (dirPct == null ? '' : dirPct + '%')));
  segI.appendChild(el('b', null, 'inverse ' + (invPct == null ? '' : invPct + '%')));
  bar.appendChild(segD);
  bar.appendChild(segI);
  bar.setAttribute('role', 'img');
  bar.setAttribute('aria-label', 'Pool split, about ' + (dirPct == null ? 'most' : dirPct + ' percent')
    + ' direct phrasings and about ' + (invPct == null ? 'the rest' : invPct + ' percent') + ' inverse.');
  pool.appendChild(bar);
  sheet.appendChild(pool);

  /* ---------------- the mutation row ---------------- */
  var mid = el('div', 'mid');

  function conn(dir) {
    var s = svg('svg', {
      'class': 'conn', viewBox: '0 0 48 44', 'aria-hidden': 'true', focusable: 'false'
    });
    var d = (dir < 0) ? 'M46 22 H13' : 'M2 22 H35';
    var p = svg('path', { 'class': 'wire', d: d });
    var tip = (dir < 0)
      ? 'M2 22 L13 16.4 L13 27.6 Z'
      : 'M46 22 L35 16.4 L35 27.6 Z';
    var h = svg('path', { 'class': 'tip', d: tip });
    s.appendChild(p);
    s.appendChild(h);
    return { svg: s, wire: p, tip: h };
  }

  function card(v, idx) {
    var b = el('button', 'qa' + (v.ans === 'yes' ? ' fires' : ''));
    b.type = 'button';
    b.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-label', v.tag + ' phrasing, ' + v.head + ' ' + (v.tail || v.whole)
      + '. Correct answer on this image, ' + v.ans + '. Click to pin.');

    var qh = el('span', 'qh');
    qh.appendChild(el('span', 'qtag', v.tag));
    qh.appendChild(el('span', 'qshare', v.share));
    b.appendChild(qh);

    var qt = el('span', 'qtext');
    if (v.whole) {
      qt.appendChild(el('span', 'plain', v.whole));
    } else {
      qt.appendChild(el('span', 'mut', v.head));
      qt.appendChild(el('span', 'shared', ' ' + v.tail));
    }
    b.appendChild(qt);

    var pol = el('span', 'pol');
    pol.appendChild(el('i', null, 'polarity'));
    pol.appendChild(el('span', null, v.polarity));
    b.appendChild(pol);

    var ab = el('span', 'ansbox');
    ab.appendChild(el('span', 'acap', 'correct answer for this pairing'));
    var top = el('span', 'atop');
    top.appendChild(el('span', 'mark'));
    top.appendChild(el('span', 'ans', v.ans));
    top.appendChild(el('span', 'pin', 'pinned'));
    ab.appendChild(top);
    ab.appendChild(el('span', 'why', v.why));
    b.appendChild(ab);

    b.addEventListener('click', function () {
      pinned = (pinned === idx) ? null : idx;
      setActive(idx);
    });
    b.addEventListener('focus', function () { setActive(idx); });
    b.addEventListener('mouseenter', function () { if (pinned === null) setActive(idx); });
    return b;
  }

  var cardA = card(VIEWS[0], 0);
  var connA = conn(-1);

  /* the abstract stand-in for the picture. it is a drawing, not a dataset image. */
  var gl = el('div', 'gl');
  var g = svg('svg', { viewBox: '0 0 200 200', 'class': 'glyph', 'aria-hidden': 'true', focusable: 'false' });
  g.appendChild(svg('rect', { 'class': 'frame', x: '15', y: '25', width: '170', height: '150', rx: '11' }));
  g.appendChild(svg('circle', { 'class': 'sun', cx: '63', cy: '70', r: '13' }));
  g.appendChild(svg('path', { 'class': 'hill', d: 'M25 158 L76 100 L110 138 L138 112 L175 158 Z' }));
  g.appendChild(svg('path', { 'class': 'hill2', d: 'M25 158 L58 124 L92 158 Z' }));
  g.appendChild(svg('path', { 'class': 'scan', d: 'M15 100 H185' }));
  gl.appendChild(g);
  gl.appendChild(el('span', 'gtok', '[image]'));
  gl.appendChild(el('span', 'gcap', 'one image, one subcategory. The pixels never change.'));
  gl.appendChild(el('span', 'gnote', 'abstract placeholder, not a dataset image'));

  var connB = conn(1);
  var cardB = card(VIEWS[1], 1);

  mid.appendChild(cardA);
  mid.appendChild(connA.svg);
  mid.appendChild(gl);
  mid.appendChild(connB.svg);
  mid.appendChild(cardB);
  sheet.appendChild(mid);

  /* ---------------- the two rows that come out ---------------- */
  var rows = el('div', 'rows');
  rows.appendChild(el('div', 'rh', 'both are valid training views of the same category'));
  VIEWS.forEach(function (v, k) {
    var r = el('div', 'row');
    r.appendChild(el('span', 'rix', '0' + (k + 1)));
    r.appendChild(el('span', 'rtxt', v.tag + ' phrasing  ·  same [image] document  ·  '
      + v.polarity));
    var c = el('span', 'rlab' + (v.ans === 'yes' ? ' y' : ''));
    c.textContent = v.ans;
    r.appendChild(c);
    rows.appendChild(r);
  });
  sheet.appendChild(rows);

  sheet.appendChild(el('div', 'foot',
    'Counts and the direct question are from the paper, and so is the label printed against it. '
    + 'The inverse wording here is written to show the flip, since the paper reports the share of '
    + 'inverse phrasings rather than the phrasings themselves. Its answer follows from the direct '
    + 'label by construction, not from a model call.'));

  root.appendChild(wrap);

  /* ---------------- active side ---------------- */
  var nodes = [cardA, cardB];
  var conns = [connA, connB];
  var active = 0;
  var pinned = null;

  function setActive(i) {
    active = i;
    nodes.forEach(function (n, k) {
      n.classList.toggle('on', k === i);
      n.classList.toggle('ispin', pinned === k);
      n.setAttribute('aria-pressed', pinned === k ? 'true' : 'false');
    });
    conns.forEach(function (c, k) {
      c.wire.classList.toggle('on', k === i);
      c.tip.classList.toggle('on', k === i);
      if (k !== i) c.wire.removeAttribute('stroke-dashoffset');
    });
  }
  setActive(0);

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

  var scan = g.querySelector('.scan');

  return {
    start: function () { fit(); },
    stop: function () {},
    tick: function (t) {
      if (pinned === null) {
        var i = (Math.floor(t / 3.1) % 2);
        if (i !== active) setActive(i);
      }
      var c = conns[active];
      if (c) c.wire.setAttribute('stroke-dashoffset', String(-((t * 20) % 18)));
      if (scan) {
        var y = 34 + (Math.sin(t * 0.7) * 0.5 + 0.5) * 130;
        scan.setAttribute('d', 'M15 ' + y.toFixed(1) + ' H185');
      }
    }
  };
};
