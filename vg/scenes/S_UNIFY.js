/* S_UNIFY, Act I, beat 07.
   Four source label shapes, a binary flag, a fixed category list, a graded
   severity ladder and a turn level refusal mark, all folding into the same
   yes/no record. The question is phrased in whichever scheme is folding, so
   the four vocabularies never have to be reconciled with each other.
   Category names, tier counts, query pools, transcript format and taxonomy
   sizes are all read from window.SS at runtime. */
window.SCENES = window.SCENES || {};
window.SCENES['S_UNIFY'] = function (root, api) {
  var SS = api.SS || {};
  var el = api.el;
  var svg = api.svg;

  var wrap = el('div', 'sc-s_unify');
  root.appendChild(wrap);

  /* ---------------- data ---------------- */
  var qt = SS.queryTypes || [];
  var head = SS.headline || {};
  var strict = SS.strictness || [];
  var formats = SS.formats || [];
  var domains = (SS.divergence || []).map(function (d) { return d.domain; })
    .filter(function (d) { return !!d; });

  if (!qt.length) {
    wrap.appendChild(el('div', 'nodata', 'SS.queryTypes is empty'));
    return;
  }

  function pool(name, fallbackIdx) {
    for (var i = 0; i < qt.length; i++) {
      if (String(qt[i].name || '').toLowerCase() === name) return qt[i];
    }
    return qt[fallbackIdx] || qt[0] || {};
  }
  function ex(q, i) {
    var e = (q && q.examples) || [];
    return e[i] || e[0] || 'n/a';
  }
  var qCat = pool('category-specific', 0);
  var qBin = pool('binary', 1);
  var qRef = pool('refusal detection', 2);

  /* severity tiers, parsed out of the taxonomy comparison table */
  var sevRow = null;
  (SS.taxCompare || []).forEach(function (r) {
    if (/severity/i.test(String(r.aspect || ''))) sevRow = r;
  });
  var rungs = [];
  if (sevRow && sevRow.train) {
    var m, re = /([A-Za-z][A-Za-z ]*?)\s*\((\d+)\)/g;
    while ((m = re.exec(String(sevRow.train)))) {
      rungs.push({ name: m[1].trim(), n: +m[2] });
    }
  }

  var fmtRow = null;
  formats.forEach(function (f) { if (/bracket/i.test(String(f.family || ''))) fmtRow = f; });
  if (!fmtRow) fmtRow = formats[1] || formats[0] || {};
  var tplLines = String(fmtRow.tpl || '[User] {prompt}\n[Assistant] {response}').split('\n');

  var litCat = domains.indexOf('Hate') >= 0 ? 'Hate' : (domains[0] || 'n/a');
  var sample = rungs.length ? rungs[rungs.length - 1] : null;
  var strictNames = strict.map(function (s) { return s.level; }).filter(Boolean);

  /* ---------------- header ---------------- */
  var hd = el('div', 'hd');
  hd.appendChild(el('span', 'tag', 'one objective'));
  hd.appendChild(el('span', 'hsub', 'four label shapes, no vocabulary in common'));
  var hint = el('span', 'hint', 'cycling, click a shape to pin it');
  hd.appendChild(hint);
  wrap.appendChild(hd);

  /* ---------------- the four schemes ---------------- */
  var schemes = [
    {
      nm: 'binary flag',
      sub: 'two states, no categories at all',
      word: 'unsafe',
      shape: 'flag',
      q: ex(qBin, 0),
      pool: qBin,
      src: 'the single flag is set to unsafe',
      yes: true
    },
    {
      nm: 'category list',
      sub: 'a fixed set, more than one can fire',
      word: litCat,
      shape: 'grid',
      q: ex(qCat, 1),
      pool: qCat,
      src: 'the ' + litCat + ' category is set on this sample',
      yes: true
    },
    {
      nm: 'severity scale',
      sub: (rungs.length ? rungs.length + ' graded tiers' : 'graded tiers')
        + ', the cut is a policy choice',
      word: sample ? sample.name : 'graded',
      shape: 'ladder',
      q: ex(qCat, 0),
      pool: qCat,
      src: sample
        ? 'graded ' + sample.name + ', which sits under the cut'
        : 'a graded tier under the cut',
      yes: false
    },
    {
      nm: 'refusal mark',
      sub: (fmtRow.family || 'Bracketed') + ' transcript, one turn labelled',
      word: 'refusal',
      shape: 'turns',
      q: ex(qRef, 0),
      pool: qRef,
      src: 'the assistant turn is marked as a refusal',
      yes: true
    }
  ];

  var COLS = 4;
  var GAPPC = 1.43;
  var CW = (100 - GAPPC * (COLS - 1)) / COLS;

  var row = el('div', 'shapes');
  row.setAttribute('role', 'group');
  row.setAttribute('aria-label', 'four source label schemes, choose one to fold');

  schemes.forEach(function (s, i) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'shp s-' + s.shape;
    b.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-label', 'fold the ' + s.nm + ' scheme, its own word is ' + s.word);

    var top = el('span', 'stop');
    top.appendChild(el('i', 'ix', '0' + (i + 1)));
    top.appendChild(el('b', 'snm', s.nm));
    b.appendChild(top);
    b.appendChild(el('span', 'ssub', s.sub));

    var g = el('span', 'geom');
    buildShape(g, s);
    b.appendChild(g);

    var f = el('span', 'sfoot');
    f.appendChild(el('i', 'ow', 'its word'));
    f.appendChild(el('b', 'wd', s.word));
    b.appendChild(f);

    b.appendChild(el('i', 'prog'));

    b.addEventListener('click', function () { pin(i); });
    s.btn = b;
    s.progEl = b.querySelector('.prog');
    row.appendChild(b);
  });

  row.addEventListener('keydown', function (e) {
    var k = e.key, d = 0;
    if (k === 'ArrowLeft' || k === 'ArrowUp') d = -1;
    else if (k === 'ArrowRight' || k === 'ArrowDown') d = 1;
    else return;
    e.preventDefault();
    var n = (cur + d + schemes.length) % schemes.length;
    schemes[n].btn.focus();
    pin(n);
  });
  wrap.appendChild(row);

  function buildShape(g, s) {
    if (s.shape === 'flag') {
      var f = el('span', 'flag');
      ['safe', 'unsafe'].forEach(function (t, j) {
        var c = el('span', 'fc' + (j === 1 ? ' lit' : ''));
        c.appendChild(el('i', null, t));
        if (j === 1) c.appendChild(el('b', 'mk', 'set'));
        f.appendChild(c);
      });
      g.appendChild(f);
      return;
    }
    if (s.shape === 'grid') {
      var gr = el('span', 'grid');
      (domains.length ? domains : ['n/a']).forEach(function (d) {
        var c = el('span', 'gc' + (d === litCat ? ' lit' : ''));
        if (d === litCat) {
          var tick = svg('svg', {
            viewBox: '0 0 20 20', width: '9', height: '9',
            'aria-hidden': 'true', focusable: 'false'
          });
          tick.appendChild(svg('path', {
            d: 'M4 10.6 L8.2 15 L16 5', fill: 'none', stroke: 'currentColor',
            'stroke-width': '3', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'
          }));
          c.appendChild(tick);
        }
        c.appendChild(el('i', null, d));
        gr.appendChild(c);
      });
      g.appendChild(gr);
      return;
    }
    if (s.shape === 'ladder') {
      var ld = el('span', 'ladder');
      if (!rungs.length) { ld.appendChild(el('span', 'gc', 'n/a')); g.appendChild(ld); return; }
      var max = rungs.reduce(function (a, r) { return Math.max(a, r.n); }, 1);
      rungs.forEach(function (r, j) {
        var last = j === rungs.length - 1;
        if (last) {
          var cut = el('span', 'cut');
          cut.appendChild(el('i', null, 'yes starts here'));
          ld.appendChild(cut);
        }
        var rw = el('span', 'rung' + (last ? ' under' : ' over'));
        var bar = el('span', 'rb');
        bar.style.width = (26 + 74 * (r.n / max)).toFixed(1) + '%';
        rw.appendChild(bar);
        rw.appendChild(el('i', 'rn', r.name));
        rw.appendChild(el('u', 'rc', String(r.n)));
        if (last) rw.appendChild(el('b', 'mk', 'this sample'));
        ld.appendChild(rw);
      });
      g.appendChild(ld);
      return;
    }
    var tw = el('span', 'turns');
    tplLines.slice(0, 2).forEach(function (ln, j) {
      var t = el('span', 'tl' + (j === 1 ? ' lit' : ''));
      var mrole = ln.match(/^\s*(\[[^\]]*\]|[^{]*?:)\s*/);
      if (mrole) {
        t.appendChild(el('i', 'ro', mrole[1]));
        t.appendChild(el('u', 'tx', ln.slice(mrole[0].length)));
      } else {
        t.appendChild(el('u', 'tx', ln));
      }
      if (j === 1) t.appendChild(el('b', 'mk', 'refusal'));
      tw.appendChild(t);
    });
    g.appendChild(tw);
  }

  /* ---------------- the funnel ---------------- */
  var fn = svg('svg', {
    'class': 'funnel', viewBox: '0 0 840 80', preserveAspectRatio: 'none',
    'aria-hidden': 'true', focusable: 'false'
  });
  var paths = schemes.map(function (s, i) {
    var cx = (i * (CW + GAPPC) + CW / 2) * 8.4;
    var p = svg('path', {
      'class': 'ln',
      d: 'M' + cx.toFixed(1) + ' 1 C' + cx.toFixed(1) + ' 44 420 30 420 74'
    });
    fn.appendChild(p);
    return p;
  });
  fn.appendChild(svg('path', { 'class': 'ln stem', d: 'M420 70 V80' }));
  wrap.appendChild(fn);
  wrap.appendChild(el('div', 'vstem'));

  /* ---------------- the one record ---------------- */
  var rec = el('div', 'rec');
  rec.setAttribute('aria-live', 'polite');

  var rhd = el('div', 'rhd');
  rhd.appendChild(el('span', 'k', 'the one objective'));
  rhd.appendChild(el('span', 'nochip', 'no shared category vocabulary'));
  rec.appendChild(rhd);

  rec.appendChild(el('span', 'qk', "query, in that scheme's own words"));
  var qEl = el('p', 'q', schemes[0].q);
  rec.appendChild(qEl);

  var body = el('div', 'rbody');
  var answ = el('div', 'answ');
  var bigrow = el('div', 'bigrow');
  var big = el('div', 'big', 'yes');
  var glyph = svg('svg', {
    viewBox: '0 0 34 34', width: '30', height: '30', 'aria-hidden': 'true', focusable: 'false'
  });
  glyph.setAttribute('class', 'gl');
  var gp = svg('path', {
    d: 'M7 18.4 L14.2 25.6 L27 10.4', fill: 'none', stroke: 'currentColor',
    'stroke-width': '3', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'
  });
  glyph.appendChild(gp);
  bigrow.appendChild(big);
  bigrow.appendChild(glyph);
  answ.appendChild(bigrow);
  answ.appendChild(el('span', 'acap', 'one token, the whole answer'));
  body.appendChild(answ);

  function metaRow(parent, lab) {
    var r = el('div', 'mrow');
    r.appendChild(el('span', 'lab', lab));
    var v = el('span', 'val');
    r.appendChild(v);
    parent.appendChild(r);
    return v;
  }
  var rows = el('div', 'rows');
  var vPool = metaRow(rows, 'query pool');
  var vSrc = metaRow(rows, 'source label');
  var vMap = metaRow(rows, 'mapping table');
  vMap.textContent = 'none, the words never have to line up';
  vMap.className = 'val quiet';
  body.appendChild(rows);
  rec.appendChild(body);
  wrap.appendChild(rec);

  /* ---------------- the honest footer ---------------- */
  var foot = el('div', 'foot');
  var l1 = el('p', 'alt');
  l1.appendChild(el('i', 'dot', ''));
  l1.appendChild(document.createTextNode(
    'the other way, fold every scheme into one shared set, '
    + (head.trainSupers == null ? 'n/a' : head.trainSupers) + ' super classes and '
    + (head.trainLeaves == null ? 'n/a' : head.trainLeaves)
    + ' leaves, then write a lossy mapping for each dataset'));
  foot.appendChild(l1);
  var l2 = el('p', 'this');
  l2.appendChild(el('i', 'dot', ''));
  l2.appendChild(document.createTextNode(
    'this way, ' + qt.length + ' query pools, each dataset keeps its own words'
    + (strictNames.length ? ' and picks one of ' + strictNames.length
      + ' strictness tiers, ' + strictNames.join(', ') : '')
    + '. Nothing has to agree with anything else.'));
  foot.appendChild(l2);
  wrap.appendChild(foot);

  wrap.appendChild(el('p', 'honest',
    'Shape drawings, the lit category and the position of the severity cut are illustrative. '
    + 'Category names, tier counts, query pools, transcript format and taxonomy sizes are read '
    + 'from the paper tables at runtime. Nothing here is a live model call.'));

  /* ---------------- state ---------------- */
  var cur = -1;
  var auto = true;
  var running = false;
  var t0 = null;
  var PERIOD = 3.9;
  var INTRO = 1.15;

  function select(i) {
    cur = i;
    var s = schemes[i];
    schemes.forEach(function (o, j) {
      o.btn.classList.toggle('on', j === i);
      o.btn.setAttribute('aria-pressed', j === i ? 'true' : 'false');
      paths[j].classList.toggle('live', j === i);
      if (j !== i) o.progEl.style.width = '0%';
    });
    rec.classList.remove('swap');
    void rec.offsetWidth;
    if (!api.reduce) rec.classList.add('swap');

    qEl.textContent = s.q;
    big.textContent = s.yes ? 'yes' : 'no';
    gp.setAttribute('d', s.yes ? 'M7 18.4 L14.2 25.6 L27 10.4' : 'M8.5 17 H25.5');
    rec.classList.toggle('v-yes', s.yes);
    rec.classList.toggle('v-no', !s.yes);
    vPool.textContent = (s.pool.name || 'n/a')
      + (s.pool.sub ? ', ' + s.pool.sub : '');
    vSrc.textContent = s.src;
  }

  function pin(i) {
    auto = false;
    hint.textContent = 'pinned, arrow keys move';
    schemes.forEach(function (o) { o.progEl.style.width = '0%'; });
    select(i);
  }

  select(0);
  if (api.reduce) {
    wrap.classList.add('go');
    auto = false;
    hint.textContent = 'click a shape to fold it';
  }

  return {
    start: function () {
      running = true;
      t0 = null;
      if (api.reduce) return;
      wrap.classList.remove('go');
      requestAnimationFrame(function () {
        if (running) wrap.classList.add('go');
      });
    },
    stop: function () {
      running = false;
      t0 = null;
    },
    tick: function (t) {
      if (!running) return;
      if (t0 == null) t0 = t;
      var dt = t - t0;

      if (paths[cur]) {
        paths[cur].setAttribute('stroke-dashoffset', String(-((t * 22) % 18)));
      }
      if (!auto) return;

      var p = (dt - INTRO) / PERIOD;
      if (p < 0) return;
      if (p >= 1) {
        t0 = t - INTRO;
        p = 0;
        select((cur + 1) % schemes.length);
      }
      if (schemes[cur]) schemes[cur].progEl.style.width = (p * 100).toFixed(1) + '%';
    }
  };
};
