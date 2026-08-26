/* S_QA, Act I.
   Convergence. Four benchmarks whose native label formats have nothing in common,
   drawn with four different label glyphs, funnelling into one row with the same
   four slots every time: instruct, query, document, one token out of {yes, no}.
   Names, target and counts come from window.SS.benchInventory, the instruct from
   window.SS.systemPrompt, the queries from window.SS.queryTypes, the document
   template from window.SS.formats. Nothing here is typed in by hand. */
window.SCENES = window.SCENES || {};
window.SCENES['S_QA'] = function (root, api) {
  var SS = api.SS || {};
  var el = api.el;
  var svg = api.svg;

  var inv = SS.benchInventory || [];
  var qts = SS.queryTypes || [];
  var fmts = SS.formats || [];
  var head = SS.headline || {};

  function invRow(name) {
    for (var i = 0; i < inv.length; i++) if (inv[i][0] === name) return inv[i];
    return null;
  }
  function qtype(name) {
    for (var i = 0; i < qts.length; i++) if (qts[i].name === name) return qts[i];
    return null;
  }
  function template(family) {
    for (var i = 0; i < fmts.length; i++) if (fmts[i].family === family) return fmts[i].tpl;
    return '';
  }
  function comma(n) {
    return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  /* the bracketed family is the one Figure 2 uses for its document */
  var tplLines = String(template('Bracketed') || '{prompt}\n{response}').split('\n');
  var lineUser = (tplLines[0] || '{prompt}').trim();
  var lineAsst = (tplLines[1] || '{response}').trim();

  /* ---------- the four sources ---------- */
  var SPEC = [
    {
      name: 'ToxicChat', shape: 'binary flag', glyph: 'flag',
      native: 'one flag, set or not set',
      qt: 'Binary', ex: 0
    },
    {
      name: 'Aegis v2', shape: 'multi-label taxonomy', glyph: 'tree',
      native: 'a set of category tags',
      qt: 'Category-specific', ex: 1
    },
    {
      name: 'Qwen3GuardTest', shape: 'severity scale', glyph: 'scale',
      native: 'an ordered level, mild to severe',
      qt: 'Binary', ex: 0
    },
    {
      name: 'XSTest', shape: 'refusal corpus', glyph: 'refusal',
      native: 'refused, or complied',
      qt: 'Refusal detection', ex: 0
    }
  ];

  var sources = [];
  SPEC.forEach(function (s) {
    var r = invRow(s.name);
    if (!r) return;
    var q = qtype(s.qt);
    var exs = (q && q.examples) || [];
    sources.push({
      name: r[0], target: r[1], lang: r[2], count: r[3],
      shape: s.shape, glyph: s.glyph, native: s.native,
      qname: q ? q.name : 'n/a',
      qsub: q ? q.sub : '',
      query: exs[s.ex] || exs[0] || 'n/a'
    });
  });

  var wrap = el('div', 'sc-s_qa');
  root.appendChild(wrap);

  /* ---------- header ---------- */
  var hd = el('div', 'hd');
  hd.appendChild(el('span', 'tag', 'convergence'));
  hd.appendChild(el('span', 'hdsub',
    'four label formats that cannot be merged, one row that they all fit'));
  wrap.appendChild(hd);

  var conv = el('div', 'conv');
  wrap.appendChild(conv);

  /* ---------- left, the four sources ---------- */
  var left = el('div', 'srcs');
  left.setAttribute('role', 'group');
  left.setAttribute('aria-label', 'four source benchmarks, pick one to fill the row');
  conv.appendChild(left);

  function glyphFor(kind) {
    var g = svg('svg', {
      viewBox: '0 0 62 40', 'aria-hidden': 'true', focusable: 'false'
    });
    g.setAttribute('class', 'gl gl-' + kind);
    function add(tag, attrs, cls) {
      var n = svg(tag, attrs);
      if (cls) n.setAttribute('class', cls);
      g.appendChild(n);
      return n;
    }
    if (kind === 'flag') {
      add('rect', { x: 7, y: 12, width: 17, height: 17, rx: 4 }, 'o');
      add('rect', { x: 31, y: 12, width: 17, height: 17, rx: 4 }, 'f');
      add('path', { d: 'M35.2 20.6 L38.4 23.8 L44.4 17' }, 'tick');
    } else if (kind === 'tree') {
      add('rect', { x: 26, y: 2, width: 10, height: 9, rx: 2 }, 'f');
      add('path', { d: 'M31 11 V16 M11 16 H51 M11 16 V22 M31 16 V22 M51 16 V22' }, 'w');
      add('rect', { x: 5, y: 22, width: 12, height: 11, rx: 2 }, 'f');
      add('rect', { x: 25, y: 22, width: 12, height: 11, rx: 2 }, 'o');
      add('rect', { x: 45, y: 22, width: 12, height: 11, rx: 2 }, 'f');
    } else if (kind === 'scale') {
      add('path', { d: 'M5 35 H57' }, 'w');
      add('rect', { x: 8, y: 26, width: 9, height: 7, rx: 1.5 }, 'o');
      add('rect', { x: 21, y: 20, width: 9, height: 13, rx: 1.5 }, 'o');
      add('rect', { x: 34, y: 13, width: 9, height: 20, rx: 1.5 }, 'f');
      add('rect', { x: 47, y: 5, width: 9, height: 28, rx: 1.5 }, 'f');
    } else {
      add('rect', { x: 3, y: 4, width: 30, height: 14, rx: 5 }, 'o');
      add('path', { d: 'M9 18 L9 23 L15 18' }, 'w');
      add('rect', { x: 27, y: 22, width: 32, height: 14, rx: 5 }, 'o');
      add('path', { d: 'M31 34.5 L55 23.5' }, 'slash');
    }
    return g;
  }

  sources.forEach(function (s, i) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'src';
    b.id = 'S_QA-src' + i;
    b.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-label',
      s.name + ', ' + comma(s.count) + ' rows, native label format is a ' + s.shape
      + '. Show it as one row.');

    b.appendChild(glyphFor(s.glyph));

    var body = el('div', 'sbody');

    var l1 = el('div', 'l1');
    l1.appendChild(el('span', 'sname', s.name));
    var st = el('span', 'sstate', 'select');
    l1.appendChild(st);
    body.appendChild(l1);

    body.appendChild(el('div', 'l2', s.shape));
    body.appendChild(el('div', 'l3', s.native));

    var l4 = el('div', 'l4');
    var cnt = el('span', 'cnt');
    cnt.appendChild(el('b', null, comma(s.count)));
    cnt.appendChild(el('span', 'u', 'rows'));
    l4.appendChild(cnt);
    l4.appendChild(el('span', 'dot', '·'));
    l4.appendChild(el('span', 'tg', s.target));
    body.appendChild(l4);

    b.appendChild(body);
    s.stateEl = st;
    s.btn = b;

    b.addEventListener('click', function () { manual(i); });
    left.appendChild(b);
  });

  left.addEventListener('keydown', function (e) {
    var k = e.key;
    var d = (k === 'ArrowUp' || k === 'ArrowLeft') ? -1
      : (k === 'ArrowDown' || k === 'ArrowRight') ? 1 : 0;
    if (!d || !sources.length) return;
    e.preventDefault();
    var n = ((cur < 0 ? 0 : cur) + d + sources.length) % sources.length;
    sources[n].btn.focus();
    manual(n);
  });

  /* ---------- middle, the funnel ---------- */
  var FH = 400;
  var ys = [];
  for (var i = 0; i < sources.length; i++) {
    ys.push(FH * (2 * i + 1) / (2 * sources.length));
  }
  var fun = svg('svg', {
    viewBox: '0 0 100 ' + FH, preserveAspectRatio: 'none',
    'aria-hidden': 'true', focusable: 'false'
  });
  fun.setAttribute('class', 'fun');
  var paths = ys.map(function (y) {
    var p = svg('path', {
      d: 'M0 ' + y + ' C 54 ' + y + ' 36 200 86 200', fill: 'none'
    });
    p.setAttribute('class', 'fl');
    fun.appendChild(p);
    return p;
  });
  var junc = svg('path', { d: 'M86 188 V212 M86 200 H100', fill: 'none' });
  junc.setAttribute('class', 'junc');
  fun.appendChild(junc);
  var funwrap = el('div', 'funwrap');
  funwrap.appendChild(fun);
  conv.appendChild(funwrap);

  /* the same thing for narrow screens, where the fan does not fit */
  var stem = el('div', 'stem');
  conv.appendChild(stem);

  /* ---------- right, the one row ---------- */
  var card = el('div', 'rowcard');
  card.setAttribute('role', 'region');
  card.setAttribute('aria-live', 'polite');
  card.setAttribute('aria-label', 'the row every source becomes');

  var ch = el('div', 'chead');
  ch.appendChild(el('span', 'ctitle', 'one row, always these four slots'));
  var from = el('span', 'from');
  from.appendChild(el('span', 'fl1', 'from'));
  var fromName = el('b', null, '');
  from.appendChild(fromName);
  ch.appendChild(from);
  card.appendChild(ch);

  function slot(host, key, pillTxt, pillCls) {
    var s = el('div', 'slot');
    var kr = el('div', 'krow');
    kr.appendChild(el('span', 'k', key));
    if (pillTxt) kr.appendChild(el('span', 'pill' + (pillCls ? ' ' + pillCls : ''), pillTxt));
    s.appendChild(kr);
    var v = el('div', 'v');
    s.appendChild(v);
    host.appendChild(s);
    return v;
  }

  var vIns = slot(card, 'instruct', 'identical for all four', 'same');
  vIns.className = 'v mono dim';
  vIns.textContent = SS.systemPrompt || 'n/a';

  var vQry = slot(card, 'query', 'the caller writes this one', 'vary');
  vQry.className = 'v qtxt';
  var qLine = el('p', 'qline', '');
  var qTag = el('div', 'qtag');
  var qTagName = el('span', 'qtn', '');
  var qTagSub = el('span', 'qts', '');
  qTag.appendChild(qTagName);
  qTag.appendChild(qTagSub);
  vQry.appendChild(qLine);
  vQry.appendChild(qTag);

  var vDoc = slot(card, 'document', 'the source text, reformatted', 'vary');
  vDoc.className = 'v doc';
  var dl1 = el('p', 'dline', lineUser);
  var dl2 = el('p', 'dline', lineAsst);
  var dhint = el('div', 'dhint', '');
  vDoc.appendChild(dl1);
  vDoc.appendChild(dl2);
  vDoc.appendChild(dhint);

  conv.appendChild(card);

  /* the answer runs the full width, because it is the same for all four */
  var band = el('div', 'ansband');
  var vAns = slot(band, 'answer', 'identical for all four', 'same');
  vAns.className = 'v ans';
  var big = el('div', 'big');
  var wYes = el('span', 'w yes');
  var ky = svg('svg', { viewBox: '0 0 34 34', 'aria-hidden': 'true', focusable: 'false' });
  ky.setAttribute('class', 'kg');
  ky.appendChild(svg('path', {
    d: 'M7 18.4 L14.2 25.6 L27 10.4', fill: 'none', stroke: 'currentColor',
    'stroke-width': '3.4', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'
  }));
  wYes.appendChild(ky);
  wYes.appendChild(el('span', 'wt', 'yes'));

  var wNo = el('span', 'w no');
  var kn = svg('svg', { viewBox: '0 0 34 34', 'aria-hidden': 'true', focusable: 'false' });
  kn.setAttribute('class', 'kg');
  kn.appendChild(svg('circle', {
    cx: '17', cy: '17', r: '11.6', fill: 'none', stroke: 'currentColor', 'stroke-width': '3.4'
  }));
  wNo.appendChild(kn);
  wNo.appendChild(el('span', 'wt', 'no'));

  big.appendChild(wYes);
  big.appendChild(el('span', 'sep', '/'));
  big.appendChild(wNo);
  var ansrow = el('div', 'ansrow');
  ansrow.appendChild(big);
  ansrow.appendChild(el('div', 'anote',
    'the entire output space, one token, read off two logits'));
  vAns.appendChild(ansrow);
  conv.appendChild(band);

  /* ---------- notes ---------- */
  wrap.appendChild(el('p', 'insight',
    'A flag, a tag set, a level and a refusal judgement cannot be concatenated as labels. '
    + 'Asked as questions they are the same row, so the binary sources and the severity '
    + 'scale even land on the same query text.'));
  wrap.appendChild(el('p', 'honest',
    'Names, target and row counts are read from the report inventory of '
    + (head.benchmarks == null ? 'the' : head.benchmarks) + ' evaluation benchmarks, '
    + (head.splits == null ? '' : head.splits + ' splits, ')
    + 'so the counts are evaluation split sizes. The four label shapes on the left are '
    + 'illustrative drawings of each source native annotation format, not a field in the '
    + 'paper. Queries are examples from the report query pools and the document is the '
    + 'bracketed template, not a real row.'));

  /* ---------- selection ---------- */
  var cur = -1;

  function select(i) {
    if (!sources.length) return;
    cur = i;
    var s = sources[i];
    sources.forEach(function (o, j) {
      o.btn.classList.toggle('on', j === i);
      o.btn.setAttribute('aria-pressed', j === i ? 'true' : 'false');
      o.stateEl.textContent = j === i ? 'in the row' : 'select';
    });
    paths.forEach(function (p, j) { p.classList.toggle('live', j === i); });

    fromName.textContent = s.name;
    qLine.textContent = s.query;
    qTagName.textContent = s.qname + ' pool';
    qTagSub.textContent = s.qsub ? ', ' + s.qsub : '';

    var t = String(s.target || '');
    dl2.classList.toggle('off', t === 'prompt');
    dl1.classList.toggle('tgt', t === 'prompt' || t === 'prompt+response');
    dl2.classList.toggle('tgt', t === 'response' || t === 'prompt+response');
    dhint.textContent = 'the label sits on the ' + t;

    card.classList.remove('pending');
  }

  function manual(i) {
    autoAt = sources.length;
    for (var j = 0; j < paths.length; j++) drawn[j] = 1;
    select(i);
  }

  if (sources.length) select(0);

  /* ---------- the opening convergence ---------- */
  var lens = null;
  var drawn = sources.map(function () { return api.reduce ? 1 : 0; });
  var autoAt = api.reduce ? sources.length : 0;
  var base = null;
  var T0 = 0.32, STEP = 0.5, DUR = 0.62;

  function setFlow(p, t) {
    p.setAttribute('stroke-dasharray', '4 13');
    p.setAttribute('stroke-dashoffset', String(-((t * 22) % 17)));
  }
  if (api.reduce) {
    paths.forEach(function (p) {
      p.setAttribute('stroke-dasharray', '4 13');
      p.setAttribute('stroke-dashoffset', '0');
    });
  }

  return {
    start: function () {
      base = null;
      if (!sources.length) return;
      autoAt = 0;
      drawn = sources.map(function () { return 0; });
      card.classList.add('pending');
      wrap.classList.remove('run');
      void wrap.offsetWidth;
      wrap.classList.add('run');
    },
    stop: function () {
      base = null;
      wrap.classList.remove('run');
      card.classList.remove('pending');
    },
    tick: function (t) {
      if (!paths.length) return;
      if (lens === null) {
        lens = paths.map(function (p) {
          var L = 0;
          try { L = p.getTotalLength(); } catch (e) { L = 0; }
          return L > 1 ? L : 320;
        });
      }
      if (base === null) base = t;
      var p = t - base;
      for (var j = 0; j < paths.length; j++) {
        var g = (p - (T0 + j * STEP)) / DUR;
        if (g < 0) g = 0;
        if (g > 1) g = 1;
        if (drawn[j] === 1 || g >= 1) {
          if (drawn[j] !== 1) {
            drawn[j] = 1;
            if (autoAt === j) { select(j); autoAt = j + 1; }
          }
          setFlow(paths[j], t);
        } else {
          /* a line is already lit while it is travelling, not only once it lands */
          if (g > 0) paths[j].classList.add('live');
          paths[j].setAttribute('stroke-dasharray', String(lens[j]));
          paths[j].setAttribute('stroke-dashoffset', String(lens[j] * (1 - g)));
        }
      }
    }
  };
};
