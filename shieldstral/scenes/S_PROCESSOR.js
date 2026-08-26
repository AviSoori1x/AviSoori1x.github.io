/* S_PROCESSOR, Act II, beat 09.
   One hand-written processor per source dataset. Each one emits labelling logic,
   category mappings and instruction templates, and every one of them lands in the
   same four-field record. All names, sizes, tiers, pools, formats and scenario
   counts are read from window.SS at runtime. */
window.SCENES = window.SCENES || {};
window.SCENES['S_PROCESSOR'] = function (root, api) {
  var SS = api.SS || {};
  var el = api.el;
  var frag = api.frag;

  var inv = (SS.benchInventory || []).slice();
  var strict = SS.strictness || [];
  var qtypes = (SS.queryTypes || []).map(function (q) { return q.name; });
  var formats = SS.formats || [];
  var scen = (SS.evalTaxonomy || []).map(function (t) { return t.name; });

  var wrap = el('div', 'sc-s_processor');
  root.appendChild(wrap);

  if (!inv.length) {
    wrap.appendChild(el('div', 'nodata', 'SS.benchInventory is empty'));
    return;
  }

  function fmt(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function slug(s) {
    return String(s).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
  }
  function langLabel(v) { return /^\d+$/.test(String(v)) ? String(v) + ' languages' : String(v); }

  /* ---------------- header ---------------- */
  var total = inv.reduce(function (a, r) { return a + (r[3] || 0); }, 0);
  var hd = frag(
    '<div class="hd">' +
      '<div class="hdl">' +
        '<div class="big"></div>' +
        '<div class="lede"><b>one processor per training dataset</b>' +
        '<span>hand written from that dataset&#39;s documentation. The paper does not ' +
        'report how many there are. The sources below are the evaluation inventory, ' +
        'shown to illustrate the variety.</span></div>' +
      '</div>' +
      '<div class="hdr">' +
        '<div class="st"><i>datasets</i><b class="vds"></b></div>' +
        '<div class="st"><i>examples</i><b class="vex"></b></div>' +
        '<div class="st fin"><i>output formats</i><b>1</b></div>' +
      '</div>' +
    '</div>').firstChild;
  // The paper says every training dataset has its own processor but never says
  // how many there are. inv.length is the count of EVALUATION benchmarks, so it
  // must not be presented as a processor census.
  hd.querySelector('.big').textContent = '1:1';
  hd.querySelector('.vds').textContent = inv.length;
  hd.querySelector('.vex').textContent = fmt(total);
  wrap.appendChild(hd);

  /* ---------------- focus row ---------------- */
  var focus = frag(
    '<div class="focus">' +
      '<div class="src">' +
        '<div class="cap">source dataset</div>' +
        '<div class="snm"></div>' +
        '<div class="chips"><span class="ch cmod"></span><span class="ch clang"></span></div>' +
        '<div class="cnt"></div>' +
        '<div class="cap2">labelled examples</div>' +
      '</div>' +
      '<div class="arw a1" aria-hidden="true">' +
        '<svg viewBox="0 0 26 12" width="26" height="12">' +
        '<path class="ln" d="M0 6 H19.5" vector-effect="non-scaling-stroke"/>' +
        '<path class="fl" d="M19.5 2.7 L25.4 6 L19.5 9.3 Z"/></svg></div>' +
      '<div class="proc">' +
        '<div class="cap">processor <em>hand written</em></div>' +
        '<div class="file"><span class="dir">processors/</span><span class="fn"></span></div>' +
        '<div class="code"></div>' +
        '<div class="fnote">read the dataset docs, write down what its labels mean</div>' +
      '</div>' +
      '<div class="arw a3" aria-hidden="true">' +
        '<svg viewBox="0 0 26 240" preserveAspectRatio="none">' +
        '<path class="ln" d="M0 120 H9 M9 36 V204 M9 36 H26 M9 120 H26 M9 204 H26" ' +
        'vector-effect="non-scaling-stroke"/></svg></div>' +
      '<div class="emits">' +
        '<div class="em e1"><div class="ek"><span class="ix">1</span>' +
          '<span class="et">labelling logic</span></div><div class="eb"></div></div>' +
        '<div class="em e2"><div class="ek"><span class="ix">2</span>' +
          '<span class="et">category mappings</span></div><div class="eb"></div></div>' +
        '<div class="em e3"><div class="ek"><span class="ix">3</span>' +
          '<span class="et">instruction templates</span></div><div class="eb"></div></div>' +
      '</div>' +
    '</div>').firstChild;
  wrap.appendChild(focus);

  var uiName = focus.querySelector('.snm');
  var uiMod = focus.querySelector('.cmod');
  var uiLang = focus.querySelector('.clang');
  var uiCnt = focus.querySelector('.cnt');
  var uiFile = focus.querySelector('.file .fn');
  var uiCode = focus.querySelector('.code');
  var body1 = focus.querySelector('.e1 .eb');
  var body2 = focus.querySelector('.e2 .eb');
  var body3 = focus.querySelector('.e3 .eb');

  /* ---------------- the fan, every dataset ---------------- */
  var fan = frag(
    '<div class="fan">' +
      '<div class="fcap"><b>every source dataset, its own processor</b>' +
      '<span>click a column, or use the arrow keys</span></div>' +
      '<div class="cols" role="radiogroup" aria-label="source dataset"></div>' +
      '<div class="bus"></div>' +
      '<div class="drop"></div>' +
      '<div class="out">' +
        '<div class="ocap">one shared record, whatever the source</div>' +
        '<div class="fields"><span>Instruction</span><span>Query</span>' +
        '<span>Document</span><span class="lab">Label</span></div>' +
        '<div class="sys"></div>' +
      '</div>' +
    '</div>').firstChild;
  wrap.appendChild(fan);

  var sys = fan.querySelector('.sys');
  if (SS.systemPrompt) sys.textContent = SS.systemPrompt;
  else sys.textContent = 'system prompt not in SS';

  var cols = fan.querySelector('.cols');
  var btns = inv.map(function (row, i) {
    var b = el('button', 'col');
    b.type = 'button';
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', 'false');
    b.setAttribute('tabindex', '-1');
    b.setAttribute('aria-label', row[0] + ', ' + row[1] + ', ' + fmt(row[3]) + ' examples');
    b.appendChild(el('span', 'cn', row[0]));
    b.appendChild(el('span', 'cg'));
    b.appendChild(el('span', 'cs'));
    b.addEventListener('click', function () { stopAuto(); select(i); });
    cols.appendChild(b);
    return b;
  });

  var note = el('div', 'note');
  note.appendChild(el('b', null, 'from the paper'));
  note.appendChild(document.createTextNode('names, sizes, modalities, the ' + strict.length
    + ' tiers, the query pools, the ' + formats.length + ' formats, the ' + scen.length
    + ' scenarios, the shared record. '));
  note.appendChild(el('b', 'ill', 'illustrative'));
  note.appendChild(document.createTextNode('the processor code is not published, so the file name, '
    + 'the stanza, and which tier, pool and format each dataset draws are shape only.'));
  wrap.appendChild(note);

  /* ---------------- per-dataset derivations ---------------- */
  function tierFor(mod) {
    var want = mod === 'image' ? 'strict' : (mod === 'response' ? 'lenient' : 'moderate');
    for (var i = 0; i < strict.length; i++) {
      if (String(strict[i].level).toLowerCase() === want) return strict[i];
    }
    return strict[0] || null;
  }
  function poolFor(mod) {
    var want = mod === 'response' ? 'Refusal detection'
      : (mod === 'image' ? 'Binary' : 'Category-specific');
    return qtypes.indexOf(want) >= 0 ? want : (qtypes[0] || 'query pool not in SS');
  }
  function docLines(tpl, mod) {
    var out = [];
    if (mod.indexOf('image') >= 0) out.push('{image}');
    String(tpl || '').split('\n').forEach(function (L) {
      if (L.indexOf('{prompt}') >= 0 && mod.indexOf('prompt') < 0) return;
      if (L.indexOf('{response}') >= 0 && mod.indexOf('response') < 0) return;
      out.push(L);
    });
    if (!out.length) out.push('{' + mod + '}');
    return out;
  }
  function monoRow(parent, a, b, cls) {
    var r = el('div', 'mr' + (cls ? ' ' + cls : ''));
    r.appendChild(el('span', 'ma', a));
    if (b != null) r.appendChild(el('span', 'mb', b));
    parent.appendChild(r);
    return r;
  }

  /* ---------------- selection ---------------- */
  var cur = -1;
  function select(i) {
    if (i === cur) return;
    cur = i;
    var row = inv[i];
    var name = row[0], mod = String(row[1]), lang = row[2], n = row[3];

    btns.forEach(function (b, j) {
      var on = j === i;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', on ? 'true' : 'false');
      b.setAttribute('tabindex', on ? '0' : '-1');
    });

    uiName.textContent = name;
    var L = name.length;
    uiName.style.fontSize = (L <= 8 ? 23 : L <= 13 ? 19 : L <= 16 ? 16.5 : 14.5) + 'px';
    uiMod.textContent = mod;
    uiLang.textContent = langLabel(lang);
    uiCnt.textContent = fmt(n);
    uiFile.textContent = slug(name) + '.py';

    uiCode.textContent = '';
    ['def build(row):',
      '  doc   = FMT.render(row)',
      '  label = MAP[row.cat]',
      '  query = TPL.pick(row)',
      '  return Record(',
      '    instruction=INSTR,',
      '    query=query,',
      '    document=doc,',
      '    label=label)'
    ].forEach(function (line) { uiCode.appendChild(el('div', 'cl', line)); });

    /* 1, labelling logic */
    body1.textContent = '';
    monoRow(body1, 'source marks it unsafe', 'yes', 'yes');
    monoRow(body1, 'everything else', 'no', 'no');
    var j1 = el('div', 'sub');
    j1.appendChild(document.createTextNode('judged on the '));
    j1.appendChild(el('b', null, mod.split('+').join(' and ')));
    body1.appendChild(j1);

    /* 2, category mappings */
    body2.textContent = '';
    var pick = el('div', 'pills');
    var k = scen.length;
    for (var s = 0; s < 3 && s < k; s++) {
      pick.appendChild(el('span', 'pill', scen[(i * 3 + s) % k]));
    }
    if (k > 3) pick.appendChild(el('span', 'pill more', '+' + (k - 3) + ' more'));
    body2.appendChild(pick);
    body2.appendChild(el('div', 'sub', k
      ? ('its own category names, folded into ' + k + ' scenarios')
      : 'taxonomy not in SS'));

    /* 3, instruction templates */
    body3.textContent = '';
    var t = tierFor(mod);
    var tl = el('div', 'tier');
    tl.appendChild(el('b', null, t ? t.level : 'tier not in SS'));
    tl.appendChild(el('span', null, t ? t.domains : ''));
    body3.appendChild(tl);
    monoRow(body3, 'query pool', poolFor(mod));
    if (formats.length) {
      var f = formats[i % formats.length];
      monoRow(body3, 'document format', f.family);
      var tp = el('div', 'tpl');
      docLines(f.tpl, mod).forEach(function (L) { tp.appendChild(el('div', 'cl', L)); });
      body3.appendChild(tp);
    } else {
      monoRow(body3, 'document format', 'not in SS');
    }
  }

  /* the three-way arrow is drawn against the measured card centres */
  var fanSvg = focus.querySelector('.a3 svg');
  var fanPath = fanSvg.querySelector('path');
  var emitsBox = focus.querySelector('.emits');
  function layoutFan() {
    var box = emitsBox.getBoundingClientRect();
    if (!box.height) return;
    var y = ['.e1', '.e2', '.e3'].map(function (s) {
      var r = focus.querySelector(s).getBoundingClientRect();
      return +(r.top - box.top + r.height / 2).toFixed(1);
    });
    var h = Math.round(box.height);
    fanSvg.setAttribute('viewBox', '0 0 26 ' + h);
    fanPath.setAttribute('d', 'M0 ' + (h / 2).toFixed(1) + ' H9 M9 ' + y[0] + ' V' + y[2] +
      ' M9 ' + y[0] + ' H26 M9 ' + y[1] + ' H26 M9 ' + y[2] + ' H26');
  }
  if (window.ResizeObserver) new ResizeObserver(layoutFan).observe(emitsBox);
  else window.addEventListener('resize', layoutFan);
  requestAnimationFrame(layoutFan);

  cols.addEventListener('keydown', function (e) {
    var n = inv.length, d = 0, j = cur;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') d = 1;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') d = -1;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = n - 1;
    else return;
    e.preventDefault();
    stopAuto();
    if (d) j = (cur + d + n) % n;
    select(j);
    btns[j].focus();
  });

  /* ---------------- slow tour, stops on any input ---------------- */
  var auto = !api.reduce;
  var base = null, from = 0;
  function stopAuto() {
    if (!auto) return;
    auto = false;
    var h = fan.querySelector('.fcap span');
    if (h) h.textContent = 'tour paused, arrow keys move';
  }
  cols.addEventListener('focusin', stopAuto);

  select(0);

  return {
    start: function () { base = null; from = cur < 0 ? 0 : cur; layoutFan(); },
    stop: function () { base = null; },
    tick: function (t) {
      if (!auto) return;
      if (base == null) { base = t; return; }
      select((from + Math.floor((t - base) / 2.6)) % inv.length);
    }
  };
};
