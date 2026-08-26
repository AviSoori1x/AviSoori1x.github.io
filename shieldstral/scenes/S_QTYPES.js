/* S_QTYPES, Act 2.
   The three query task types in SS.queryTypes, all pointed at the one text-only
   document from SS.fig2[0]. Same document, three questions, three answers.
   Built at a fixed 820x760 sheet and scaled to whatever the stage gives us,
   the way a viewBox would, so nothing ever clips. */
window.SCENES = window.SCENES || {};
window.SCENES['S_QTYPES'] = function (root, api) {
  var SS = api.SS || {};
  var el = api.el, svg = api.svg;

  var types = (SS.queryTypes || []).slice(0, 3);
  var fig = (SS.fig2 || [])[0] || {};
  var fmt = (SS.formats || [])[1] || {};

  /* A reading of the one document below. Not a model call, not a paper label.
     Keyed by name so a reorder of the data cannot mismatch them. */
  var READ = {
    'Category-specific': { ans: 'no', why: 'the assistant refuses, nothing here promotes violence' },
    'Binary': { ans: 'no', why: 'the harmful request is never actually answered' },
    'Refusal detection': { ans: 'yes', why: 'the assistant turn is a refusal' }
  };

  var wrap = el('div', 'sc-s_qtypes');
  var sheet = el('div', 'sheet');
  wrap.appendChild(sheet);

  /* ---------------- header ---------------- */
  var hd = el('div', 'hd');
  hd.appendChild(el('span', 'eyy', 'query pools · one document, three questions'));
  hd.appendChild(el('span', 'hint', 'click or tab a column to pin it'));
  sheet.appendChild(hd);

  /* ---------------- the one shared document ---------------- */
  var doc = el('div', 'doc');
  var dh = el('div', 'dh');
  dh.appendChild(el('span', 'dht', 'the document'));
  dh.appendChild(el('span', 'dhf', (fmt.family || 'Bracketed') + ' format'));
  doc.appendChild(dh);

  var body = el('div', 'db');
  String(fig.document || '').split('\n').forEach(function (ln) {
    var isRole = /^\s*\[.*\]\s*$/.test(ln);
    body.appendChild(el('span', 'dl' + (isRole ? ' role' : ''), ln));
  });
  doc.appendChild(body);

  var prov = el('span', 'prov');
  prov.appendChild(el('i', null, 'paper label'));
  prov.appendChild(el('b', null, String(fig.label || '')));
  prov.appendChild(el('u', null, 'for ' + (fig.query || '')));
  doc.appendChild(prov);
  sheet.appendChild(doc);

  /* ---------------- the bus, one document down into three questions ------- */
  var XS = [161.3, 500, 838.7];
  var bus = svg('svg', {
    'class': 'bus', viewBox: '0 0 1000 64', 'aria-hidden': 'true', focusable: 'false'
  });
  bus.appendChild(svg('path', { 'class': 'stem', d: 'M500 0 V16' }));
  bus.appendChild(svg('path', { 'class': 'stem', d: 'M' + XS[0] + ' 16 H' + XS[2] }));
  var drops = [], heads = [];
  XS.forEach(function (x, i) {
    bus.appendChild(svg('circle', { 'class': 'jn', cx: x, cy: 16, r: 3 }));
    var p = svg('path', { 'class': 'drop', d: 'M' + x + ' 16 V52' });
    var h = svg('path', {
      'class': 'head', d: 'M' + (x - 5.5) + ' 52 L' + (x + 5.5) + ' 52 L' + x + ' 63 Z'
    });
    bus.appendChild(p); bus.appendChild(h);
    drops.push(p); heads.push(h);
  });
  sheet.appendChild(bus);

  /* ---------------- the three query types ---------------- */
  var cols = el('div', 'cols');
  var nodes = [];
  types.forEach(function (t, i) {
    var r = READ[t.name] || { ans: '', why: '' };
    var b = el('button', 'col' + (r.ans === 'yes' ? ' fires' : ''));
    b.type = 'button';
    b.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-label', t.name + ' query, ' + (t.sub || '')
      + '. Answer on this document: ' + (r.ans || 'not shown') + '. Click to pin.');

    b.appendChild(el('span', 'ix', '0' + (i + 1)));
    b.appendChild(el('span', 'nm', t.name));

    var te = el('span', 'tests');
    te.appendChild(el('i', null, 'tests'));
    te.appendChild(el('span', null, t.sub || ''));
    b.appendChild(te);

    var qs = el('span', 'qs');
    qs.appendChild(el('span', 'qlbl', 'from the pool'));
    (t.examples || []).forEach(function (q) { qs.appendChild(el('span', 'q', q)); });
    b.appendChild(qs);

    b.appendChild(el('span', 'desc', t.desc || ''));

    var ab = el('span', 'ansbox');
    var top = el('span', 'atop');
    top.appendChild(el('span', 'mark'));
    top.appendChild(el('span', 'ans', r.ans));
    top.appendChild(el('span', 'pin', 'pinned'));
    ab.appendChild(top);
    ab.appendChild(el('span', 'why', r.why));
    b.appendChild(ab);

    b.addEventListener('click', function () {
      pinned = (pinned === i) ? null : i;
      setActive(i);
    });
    b.addEventListener('focus', function () { setActive(i); });
    b.addEventListener('mouseenter', function () { if (pinned === null) setActive(i); });

    cols.appendChild(b);
    nodes.push(b);
  });
  sheet.appendChild(cols);

  sheet.appendChild(el('div', 'foot',
    'Names, questions and explanations are verbatim from the paper. The three answers are a '
    + 'reading of that one document, not a live model call. Only the pairing printed inside the '
    + 'document panel is a ground-truth label.'));

  root.appendChild(wrap);

  /* ---------------- active column ---------------- */
  var active = types.length - 1;
  var pinned = null;

  function setActive(i) {
    active = i;
    nodes.forEach(function (n, k) {
      n.classList.toggle('on', k === i);
      n.classList.toggle('ispin', pinned === k);
      n.setAttribute('aria-pressed', pinned === k ? 'true' : 'false');
    });
    drops.forEach(function (p, k) {
      p.classList.toggle('on', k === i);
      if (k !== i) p.removeAttribute('stroke-dashoffset');
    });
    heads.forEach(function (p, k) { p.classList.toggle('on', k === i); });
  }
  setActive(active);

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
      if (pinned === null && types.length) {
        var i = Math.floor(t / 2.9) % types.length;
        if (i !== active) setActive(i);
      }
      var d = drops[active];
      if (d) d.setAttribute('stroke-dashoffset', String(-((t * 22) % 20)));
    }
  };
};
