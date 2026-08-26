/* S_EVALTAX, Act 4.
   Two taxonomies built on purpose not to match. The top half is the structural
   asymmetry: the training side is ragged, the evaluation side is a lattice of
   pairs drawn straight from SS.evalTaxonomy. The bottom half lets the reader
   pick a harm domain from SS.divergence and see how each side carves it, with
   the paper's own "why" line as the payoff. Jailbreak is in the list because
   the evaluation side of it is deliberately empty.
   Built at a fixed 820x760 sheet and scaled to whatever the stage gives us. */
window.SCENES = window.SCENES || {};
window.SCENES['S_EVALTAX'] = function (root, api) {
  var SS = api.SS || {};
  var el = api.el, svg = api.svg;

  var TAX = SS.taxCompare || [];
  var DIV = SS.divergence || [];
  var ET = SS.evalTaxonomy || [];

  /* ---------------- read everything out of SS ---------------- */
  function aspect(prefix) {
    for (var i = 0; i < TAX.length; i++) {
      if (String(TAX[i].aspect || '').toLowerCase().indexOf(prefix) === 0) return TAX[i];
    }
    return {};
  }
  function firstInt(s) {
    var m = String(s == null ? '' : s).match(/\d+/);
    return m ? +m[0] : null;
  }
  /* "4 tiers: Critical (19), High (28), ..." into name and count pairs */
  function tiers(s) {
    var out = [], re = /([A-Za-z][A-Za-z ]*?)\s*\((\d+)\)/g, m;
    while ((m = re.exec(String(s == null ? '' : s)))) out.push({ name: m[1], n: +m[2] });
    return out;
  }
  /* leaf count carried inside a divergence string, or null when there is none */
  function leavesOf(s) {
    s = String(s == null ? '' : s);
    var m = s.match(/(\d+)\s*leaves/i);
    if (m) return +m[1];
    var p = s.match(/\((\d+)\)/g);
    if (p) {
      var t = 0;
      for (var i = 0; i < p.length; i++) t += +p[i].replace(/\D/g, '');
      return t;
    }
    return null;
  }

  var rSC = aspect('super classes');
  var rLeaf = aspect('leaf categories');
  var rSub = aspect('subcategories');
  var rSev = aspect('severity');
  var rSrc = aspect('source');

  var trainSC = firstInt(rSC.train), evalSC = firstInt(rSC.eval);
  var trainLeaf = firstInt(rLeaf.train), evalLeaf = firstInt(rLeaf.eval);
  var trainSev = tiers(rSev.train), evalSev = tiers(rSev.eval);

  var evalSub = 0;
  ET.forEach(function (sc) { evalSub += (sc.subs || []).length; });
  if (!evalSub && evalLeaf) evalSub = evalLeaf / 2;

  var W = 374, ROW = 18, TH = 10;

  var wrap = el('div', 'sc-s_evaltax');
  var sheet = el('div', 'sheet');
  wrap.appendChild(sheet);

  /* ---------------- header ---------------- */
  var hd = el('div', 'hd');
  hd.appendChild(el('span', 'eyy', 'two taxonomies · built on purpose not to match'));
  hd.appendChild(el('span', 'hint', 'click or tab a domain to pin it'));
  sheet.appendChild(hd);

  /* ---------------- the numerals ---------------- */
  function nums(items) {
    var n = el('div', 'nums');
    items.forEach(function (it) {
      var b = el('div', 'n');
      b.appendChild(el('span', 'v', it.v == null ? 'n/a' : String(it.v)));
      b.appendChild(el('span', 'kk', it.k));
      n.appendChild(b);
    });
    return n;
  }

  /* ---------------- training side, the four published severity tiers -------- */
  function sevStrip(list) {
    var H = 90, X0 = 98, top = 16;
    var maxN = 1;
    list.forEach(function (t) { if (t.n > maxN) maxN = t.n; });
    var pitch = Math.min(10, (W - X0 - 2) / maxN);
    var tw = Math.max(3.4, pitch - 2.8);
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, 'class': 'dia amber', role: 'img' });
    s.setAttribute('aria-label', 'Training leaf categories grouped by severity tier: '
      + list.map(function (t) { return t.name + ' ' + t.n; }).join(', ') + '.');
    var cap = svg('text', { 'class': 'cap', x: 0, y: 7 });
    cap.textContent = 'ALL ' + (trainLeaf || list.length) + ' LEAVES, BY SEVERITY TIER';
    s.appendChild(cap);
    list.forEach(function (t, i) {
      var y = top + i * ROW;
      var nm = svg('text', { 'class': 'rl', x: 62, y: y + TH - 1, 'text-anchor': 'end' });
      nm.textContent = t.name.toUpperCase();
      s.appendChild(nm);
      var ct = svg('text', { 'class': 'rn', x: 88, y: y + TH - 1, 'text-anchor': 'end' });
      ct.textContent = String(t.n);
      s.appendChild(ct);
      for (var k = 0; k < t.n; k++) {
        s.appendChild(svg('rect', {
          'class': 'tk', x: X0 + k * pitch, y: y, width: tw, height: TH, rx: 2
        }));
      }
    });
    return s;
  }

  /* ---------------- evaluation side, the real lattice of pairs -------------- */
  function lattice(tax) {
    var H = 90, top = 16, tw = 9;
    var cols = tax.length || evalSC || 12;
    var maxSubs = 1;
    tax.forEach(function (sc) { maxSubs = Math.max(maxSubs, (sc.subs || []).length); });
    var pitch = (W - 8) / cols;
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, 'class': 'dia lime', role: 'img' });
    s.setAttribute('aria-label', cols + ' evaluation super classes, ' + evalSub
      + ' subcategories, exactly two leaf categories in every subcategory, '
      + evalLeaf + ' leaves in total. Severity tiers: '
      + evalSev.map(function (t) { return t.name + ' ' + t.n; }).join(', ') + '.');
    var cap = svg('text', { 'class': 'cap', x: 0, y: 7 });
    cap.textContent = 'EVERY SUBCATEGORY IS A PAIR';
    s.appendChild(cap);
    var cap2 = svg('text', { 'class': 'cap', x: W, y: 7, 'text-anchor': 'end' });
    cap2.textContent = evalSev.length + ' TIERS  '
      + evalSev.map(function (t) { return t.n; }).join(' / ');
    s.appendChild(cap2);
    tax.forEach(function (sc, i) {
      var cx = 4 + pitch * (i + 0.5);
      var subs = sc.subs || [];
      s.appendChild(svg('line', {
        'class': 'stem', x1: cx, y1: top - 4, x2: cx, y2: top + maxSubs * ROW - 8
      }));
      subs.forEach(function (sub, j) {
        var y = top + j * ROW;
        (sub.leaves || []).forEach(function (lf, k) {
          var x = cx + (k === 0 ? -(tw + 1.6) : 1.6);
          var r = svg('rect', { 'class': 'tk', x: x, y: y, width: tw, height: TH, rx: 2 });
          var ttl = svg('title');
          ttl.textContent = sc.name + ' / ' + sub.name + ' / ' + (lf.name || '');
          r.appendChild(ttl);
          s.appendChild(r);
        });
      });
      var lb = svg('text', { 'class': 'cl', x: cx, y: H - 3, 'text-anchor': 'middle' });
      lb.textContent = String(i + 1);
      s.appendChild(lb);
    });
    return s;
  }

  /* the design constraint, lead sentence bright, the rest quiet */
  function cons(text) {
    var d = el('div', 'cons');
    var s = String(text == null ? '' : text), i = s.indexOf('. ');
    if (i > 0) {
      d.appendChild(el('b', null, s.slice(0, i + 1)));
      d.appendChild(el('span', null, ' ' + s.slice(i + 2)));
    } else {
      d.textContent = s;
    }
    return d;
  }

  var rCon = aspect('design constraint');
  var srow = el('div', 'srow');

  var pT = el('div', 'pan train');
  pT.appendChild(el('div', 'ptag', 'training taxonomy'));
  pT.appendChild(nums([
    { v: trainSC, k: 'super classes' },
    { v: trainLeaf, k: 'leaf categories' }
  ]));
  pT.appendChild(el('div', 'shape', rSub.train || ''));
  pT.appendChild(cons(rCon.train));
  pT.appendChild(sevStrip(trainSev));
  pT.appendChild(el('div', 'sev', rSrc.train || ''));
  srow.appendChild(pT);

  var pE = el('div', 'pan ev');
  pE.appendChild(el('div', 'ptag', 'evaluation taxonomy'));
  pE.appendChild(nums([
    { v: evalSC, k: 'super classes' },
    { v: evalSub, k: 'subcategories' },
    { v: evalLeaf, k: 'leaf categories' }
  ]));
  pE.appendChild(el('div', 'shape', rSub.eval || ''));
  pE.appendChild(cons(rCon.eval));
  pE.appendChild(lattice(ET));
  pE.appendChild(el('div', 'sev', rSrc.eval || ''));
  srow.appendChild(pE);

  sheet.appendChild(srow);

  /* ---------------- the domain picker ---------------- */
  var counts = DIV.map(function (d) {
    return { t: leavesOf(d.train), e: leavesOf(d.eval) };
  });

  var tabs = el('div', 'tabs');
  var tabNodes = [];
  DIV.forEach(function (d, i) {
    var c = counts[i];
    var b = el('button', 'tab' + (c.e == null ? ' gap' : ''));
    b.type = 'button';
    b.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-label', d.domain + '. Training ' + (c.t == null ? 'unstated' : c.t)
      + ' leaves, evaluation ' + (c.e == null ? 'none, not in the evaluation taxonomy' : c.e + ' leaves')
      + '. Click to pin.');
    b.appendChild(el('span', 'dot'));
    b.appendChild(el('span', 'tn', d.domain));
    var cc = el('span', 'tc');
    cc.appendChild(el('i', 'ct', c.t == null ? '?' : String(c.t)));
    cc.appendChild(el('i', 'cs', 'vs'));
    cc.appendChild(el('i', 'ce', c.e == null ? 'none' : String(c.e)));
    b.appendChild(cc);
    b.addEventListener('click', function () {
      pinned = (pinned === i) ? null : i;
      setActive(i);
    });
    b.addEventListener('focus', function () { setActive(i); });
    b.addEventListener('mouseenter', function () { if (pinned === null) setActive(i); });
    tabs.appendChild(b);
    tabNodes.push(b);
  });
  sheet.appendChild(tabs);

  /* ---------------- the two carve cards ---------------- */
  function card(cls, tag) {
    var c = el('div', 'card2 ' + cls);
    c.appendChild(el('div', 'ctag', tag));
    var top = el('div', 'ctop');
    top.appendChild(el('span', 'cnum', ''));
    top.appendChild(el('span', 'clab', ''));
    c.appendChild(top);
    c.appendChild(el('div', 'rail'));
    c.appendChild(el('div', 'ctxt', ''));
    return c;
  }
  var drow = el('div', 'drow');
  var cT = card('train', 'training carves it');
  var cE = card('ev', 'evaluation carves it');
  drow.appendChild(cT);
  drow.appendChild(cE);
  sheet.appendChild(drow);

  var whyBar = el('div', 'why');
  whyBar.appendChild(el('span', 'wtag', 'why they differ'));
  var whyTxt = el('span', 'wtxt', '');
  whyBar.appendChild(whyTxt);
  sheet.appendChild(whyBar);

  sheet.appendChild(el('div', 'foot',
    'Every count is read from the paper. The right hand lattice is the real evaluation taxonomy. '
    + 'The left hand rows are the four published severity tiers, the only breakdown of the '
    + (trainLeaf || 'training') + ' training leaves given, so position inside a tier means '
    + 'nothing. No model call here.'));

  root.appendChild(wrap);

  /* ---------------- fill a card for one side of one domain ---------------- */
  function fill(c, txt, n, isEval) {
    var none = (n == null);
    c.classList.toggle('empty', none && isEval);
    c.querySelector('.cnum').textContent = none ? '0' : String(n);
    c.querySelector('.clab').textContent = none
      ? 'leaves, no counterpart' : (n === 1 ? 'leaf category' : 'leaf categories');
    var rail = c.querySelector('.rail');
    rail.innerHTML = '';
    if (none) {
      rail.appendChild(el('span', 'norail', 'nothing on this side'));
    } else {
      for (var i = 0; i < n && i < 20; i++) {
        var t = el('span', 'rt');
        t.style.animationDelay = (i * 26) + 'ms';
        rail.appendChild(t);
      }
    }
    c.querySelector('.ctxt').textContent = txt || '';
  }

  var active = 0, pinned = null;
  for (var q = 0; q < DIV.length; q++) {
    if (String(DIV[q].domain || '').toLowerCase() === 'hate') { active = q; break; }
  }
  var start = active;

  function setActive(i) {
    if (!DIV.length) return;
    active = i;
    var d = DIV[i], c = counts[i];
    tabNodes.forEach(function (n, k) {
      n.classList.toggle('on', k === i);
      n.classList.toggle('ispin', pinned === k);
      n.setAttribute('aria-pressed', pinned === k ? 'true' : 'false');
    });
    fill(cT, d.train, c.t, false);
    fill(cE, d.eval, c.e, true);
    whyTxt.textContent = d.why || '';
    whyBar.classList.toggle('gap', c.e == null);
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
      if (pinned !== null || !DIV.length) return;
      var i = (start + Math.floor(t / 3.4)) % DIV.length;
      if (i !== active) setActive(i);
    }
  };
};
