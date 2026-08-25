/* S_ISOQUERY, Act I.
   Mirror of the iso-content scene. One question pinned at the top, one innocuous
   source sentence forking into two unsafe rewrites, one in the target category and
   one in a sibling category. Toggling the document flips the ground truth label.
   Every string is read from window.SS.fig4 at runtime. */
window.SCENES = window.SCENES || {};
window.SCENES['S_ISOQUERY'] = function (root, api) {
  var SS = api.SS || {};
  var F = SS.fig4 || {};
  var el = api.el;
  var svg = api.svg;

  var wrap = el('div', 'sc-s_isoquery');
  root.appendChild(wrap);

  /* ---------- header ---------- */
  var hd = el('div', 'hd');
  hd.appendChild(el('span', 'tag', 'iso-query'));
  hd.appendChild(el('span', 'hdsub', 'one question held still, two rewrites of one sentence'));
  wrap.appendChild(hd);

  /* ---------- the pinned question ---------- */
  var qw = el('div', 'qwrap');

  var fi = el('div', 'fld');
  fi.appendChild(el('span', 'k', 'instruct'));
  fi.appendChild(el('p', 'ins', F.instruct || 'n/a'));
  qw.appendChild(fi);

  var fq = el('div', 'fld');
  var krow = el('div', 'krow');
  krow.appendChild(el('span', 'k', 'query'));
  var pin = el('span', 'pin');
  var pico = svg('svg', {
    viewBox: '0 0 14 14', width: '12', height: '12',
    'aria-hidden': 'true', focusable: 'false'
  });
  pico.appendChild(svg('rect', {
    x: '2.4', y: '6.1', width: '9.2', height: '6.1', rx: '1.4',
    fill: 'none', stroke: 'currentColor', 'stroke-width': '1.3'
  }));
  pico.appendChild(svg('path', {
    d: 'M4.6 6.1 V4.4 a2.4 2.4 0 0 1 4.8 0 V6.1',
    fill: 'none', stroke: 'currentColor', 'stroke-width': '1.3'
  }));
  pin.appendChild(pico);
  pin.appendChild(el('span', null, 'held fixed for both documents'));
  krow.appendChild(pin);
  fq.appendChild(krow);
  fq.appendChild(el('p', 'qtxt', F.query || 'n/a'));
  qw.appendChild(fq);
  wrap.appendChild(qw);

  /* ---------- the innocuous seed ---------- */
  var sr = el('div', 'srcrow');
  var srk = el('div', 'krow');
  srk.appendChild(el('span', 'k', 'source sentence'));
  srk.appendChild(el('span', 'tiny', 'harmless on its own, rewritten twice'));
  sr.appendChild(srk);
  sr.appendChild(el('p', 'src', F.source || 'n/a'));
  wrap.appendChild(sr);

  /* ---------- the fork ---------- */
  var branch = svg('svg', {
    viewBox: '0 0 800 52', preserveAspectRatio: 'none',
    'aria-hidden': 'true', focusable: 'false'
  });
  branch.setAttribute('class', 'branch');
  var pA = svg('path', { d: 'M400 1 C400 32 176 20 176 51', fill: 'none' });
  var pB = svg('path', { d: 'M400 1 C400 32 624 20 624 51', fill: 'none' });
  pA.setAttribute('class', 'ln');
  pB.setAttribute('class', 'ln');
  branch.appendChild(pA);
  branch.appendChild(pB);
  wrap.appendChild(branch);
  wrap.appendChild(el('div', 'stem'));

  /* ---------- the two rewrites ---------- */
  var sides = [
    { letter: 'A', cat: F.category, role: 'target category', d: F.positive || {} },
    { letter: 'B', cat: F.sibling, role: 'sibling category', d: F.negative || {} }
  ];

  var docs = el('div', 'docs');
  docs.setAttribute('role', 'group');
  docs.setAttribute('aria-label', 'choose which rewrite the classifier reads');

  sides.forEach(function (s, i) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'doc';
    b.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-label', 'read rewrite ' + s.letter + ', written toward ' + (s.cat || 'n/a'));

    var h = el('div', 'dhead');
    h.appendChild(el('span', 'lt', s.letter));
    h.appendChild(el('span', 'cat', s.cat || 'n/a'));
    h.appendChild(el('span', 'role', s.role));
    b.appendChild(h);

    b.appendChild(el('p', 'dtxt', s.d.document || 'n/a'));

    var f = el('div', 'dfoot');
    f.appendChild(el('span', 'unsafe', 'unsafe rewrite'));
    s.state = el('span', 'state', 'click to read');
    f.appendChild(s.state);
    b.appendChild(f);

    b.addEventListener('click', function () { select(i); });
    s.btn = b;
    docs.appendChild(b);
  });

  docs.addEventListener('keydown', function (e) {
    var k = e.key;
    if (k !== 'ArrowLeft' && k !== 'ArrowRight' && k !== 'ArrowUp' && k !== 'ArrowDown') return;
    var next = (k === 'ArrowLeft' || k === 'ArrowUp') ? 0 : 1;
    e.preventDefault();
    sides[next].btn.focus();
    select(next);
  });
  wrap.appendChild(docs);

  /* ---------- the verdict ---------- */
  var v = el('div', 'verdict');
  v.setAttribute('aria-live', 'polite');

  var vl = el('div', 'vleft');
  vl.appendChild(el('span', 'k', 'ground truth label'));
  var bigrow = el('div', 'bigrow');
  var big = el('div', 'big', 'yes');
  var glyph = svg('svg', { viewBox: '0 0 34 34', width: '32', height: '32',
    'aria-hidden': 'true', focusable: 'false' });
  glyph.setAttribute('class', 'gl');
  var gp = svg('path', { d: 'M7 18.4 L14.2 25.6 L27 10.4', fill: 'none',
    stroke: 'currentColor', 'stroke-width': '3',
    'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
  glyph.appendChild(gp);
  bigrow.appendChild(big);
  bigrow.appendChild(glyph);
  vl.appendChild(bigrow);
  var bar = el('div', 'bar');
  var barIn = el('i');
  bar.appendChild(barIn);
  vl.appendChild(bar);
  v.appendChild(vl);

  function row(parent, lab) {
    var r = el('div', 'row');
    r.appendChild(el('span', 'lab', lab));
    var val = el('span', 'val');
    r.appendChild(val);
    parent.appendChild(r);
    return val;
  }
  var vr = el('div', 'vright');
  var vQ = row(vr, 'question targets');
  var vD = row(vr, 'rewrite sits in');
  var vM = row(vr, 'match');
  vQ.textContent = F.category || 'n/a';
  v.appendChild(vr);
  wrap.appendChild(v);

  wrap.appendChild(el('p', 'caveat',
    'Both rewrites are unsafe by any ordinary reading. B is a no only because it is the '
    + 'wrong harm for this question.'));
  wrap.appendChild(el('p', 'honest',
    'Text and labels are the report Figure 4 iso-query pair, the annotated ground truth. '
    + 'Nothing here is a live model call.'));

  /* ---------- state ---------- */
  var live = pA;

  function select(i) {
    var s = sides[i];
    sides.forEach(function (o, j) {
      o.btn.classList.toggle('on', j === i);
      o.btn.setAttribute('aria-pressed', j === i ? 'true' : 'false');
      o.state.textContent = j === i ? 'reading this one' : 'click to read';
    });

    pA.classList.toggle('live', i === 0);
    pB.classList.toggle('live', i === 1);
    live = i === 0 ? pA : pB;
    if (api.reduce) live.removeAttribute('stroke-dashoffset');

    var label = String(s.d.label == null ? '' : s.d.label);
    var yes = label.toLowerCase() === 'yes';
    var match = !!s.cat && s.cat === F.category;

    v.classList.toggle('v-yes', yes);
    v.classList.toggle('v-no', !yes);
    big.textContent = label || 'n/a';
    gp.setAttribute('d', yes ? 'M7 18.4 L14.2 25.6 L27 10.4' : 'M8.5 17 H25.5');
    vD.textContent = (s.cat || 'n/a') + (match ? '' : ', a sibling');
    vM.textContent = match
      ? 'same category, the question fires'
      : 'different category, the question stays quiet';
    barIn.style.width = yes ? '100%' : '13%';
  }

  select(0);

  return {
    start: function () {},
    stop: function () {},
    tick: function (t) {
      if (!live) return;
      live.setAttribute('stroke-dashoffset', String(-((t * 26) % 18)));
    }
  };
};
