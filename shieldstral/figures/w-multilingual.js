(function () {
  var ID = 'w-multilingual';
  var root = document.getElementById(ID);
  if (!root) return;

  var SS = (typeof window !== 'undefined' && window.SS) ? window.SS : null;
  if (!SS) return;

  var ML = SS.multilingual || {};
  var MODELS = Array.isArray(ML.models) ? ML.models : [];
  var LANGS = ML.langs || {};
  var SCORES = ML.scores || {};
  var SUPPORTED = Array.isArray(SS.languages) ? SS.languages : [];
  var INVENTORY = Array.isArray(SS.benchInventory) ? SS.benchInventory : [];
  var LIMITS = Array.isArray(SS.limitations) ? SS.limitations : [];

  var CODES = Object.keys(LANGS);
  if (!MODELS.length || !CODES.length) return;

  var TASKS = ['prompt', 'response'];
  var TASK_LABEL = { prompt: 'Prompts', response: 'Responses' };
  var TASK_PHRASE = { prompt: 'prompt classification', response: 'response classification' };

  /* Shieldstral's column, found by name so the index is never assumed. */
  var SS_COL = -1;
  for (var mi = 0; mi < MODELS.length; mi++) {
    if (/shieldstral/i.test(MODELS[mi])) { SS_COL = mi; break; }
  }

  var MIX_MAX = 75;      /* deepest cell is 75% accent, keeps the digits legible */
  var LABEL_W = '7.4rem';

  /* ---------- helpers ---------- */

  function pick(suffix) { return document.getElementById(ID + '-' + suffix); }

  function el(tag, cls, txt) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt !== undefined && txt !== null) n.textContent = String(txt);
    return n;
  }

  function clear(n) { while (n && n.firstChild) n.removeChild(n.firstChild); }

  function num(v) { return (v === null || v === undefined || isNaN(v)) ? null : Number(v); }

  function f1(v) { var n = num(v); return n === null ? 'n/a' : n.toFixed(1); }

  function signed(v) { return (v > 0 ? '+' : '') + v.toFixed(1); }

  function thousands(n) {
    var s = String(Math.round(n)), out = '', c = 0;
    for (var i = s.length - 1; i >= 0; i--) {
      out = s.charAt(i) + out;
      c++;
      if (c % 3 === 0 && i > 0) out = ',' + out;
    }
    return out;
  }

  function row(task, code) {
    var t = SCORES[task] || {};
    return Array.isArray(t[code]) ? t[code] : [];
  }

  /* A row keyed by something other than a two letter code is an aggregate bucket. */
  function isBucket(code) { return String(code).length !== 2; }

  function isSupported(code) {
    var name = LANGS[code];
    for (var i = 0; i < SUPPORTED.length; i++) if (SUPPORTED[i] === name) return true;
    return false;
  }

  /* ---------- shared colour domain across both tasks ---------- */

  var LO = Infinity, HI = -Infinity;
  TASKS.forEach(function (t) {
    CODES.forEach(function (c) {
      row(t, c).forEach(function (v) {
        var n = num(v);
        if (n === null) return;
        if (n < LO) LO = n;
        if (n > HI) HI = n;
      });
    });
  });
  /* no usable numbers means no honest scale, so render nothing rather than invent one */
  if (!isFinite(LO) || !isFinite(HI) || HI <= LO) return;

  function ramp(v) {
    var n = num(v);
    if (n === null) return 'var(--paper-2)';
    var t = (n - LO) / (HI - LO);
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    var p = Math.round(t * MIX_MAX * 10) / 10;
    return 'color-mix(in srgb, var(--accent) ' + p + '%, var(--paper-2))';
  }

  /* ---------- derived facts ---------- */

  function rankOf(arr, v) {
    var ahead = 0;
    for (var i = 0; i < arr.length; i++) { var n = num(arr[i]); if (n !== null && n > v) ahead++; }
    return { ahead: ahead, rank: ahead + 1 };
  }

  function bestOf(arr) {
    var b = -Infinity, bi = -1;
    for (var i = 0; i < arr.length; i++) {
      var n = num(arr[i]);
      if (n !== null && n > b) { b = n; bi = i; }
    }
    return { value: bi === -1 ? null : b, index: bi };
  }

  /* Weakest named language for Shieldstral on a given task, computed not asserted. */
  function weakest(task) {
    if (SS_COL === -1) return null;
    var list = [];
    CODES.forEach(function (c) {
      if (isBucket(c)) return;
      var v = num(row(task, c)[SS_COL]);
      if (v === null) return;
      list.push({ code: c, value: v });
    });
    list.sort(function (a, b) { return a.value - b.value; });
    return list.length ? list : null;
  }

  var ENGLISH_CODE = (function () {
    for (var i = 0; i < CODES.length; i++) if (LANGS[CODES[i]] === 'English') return CODES[i];
    return null;
  })();

  /* multi language benchmarks, language count read from the inventory */
  var MULTI = INVENTORY.filter(function (b) { return /^[0-9]+$/.test(String(b[2])); })
    .map(function (b) { return { name: b[0], split: b[1], langs: parseInt(b[2], 10), items: Number(b[3]) }; })
    .sort(function (a, b) { return b.langs - a.langs; });

  var COVERAGE_QUOTE = (function () {
    for (var i = 0; i < LIMITS.length; i++) {
      if (/languag/i.test(String(LIMITS[i].d))) return LIMITS[i];
    }
    return LIMITS.length ? LIMITS[0] : null;
  })();

  /* ---------- state ---------- */

  var state = {
    task: TASKS[0],
    sort: 'table',
    rowCode: null,
    col: SS_COL === -1 ? 0 : SS_COL,
    userPicked: false
  };

  function orderedCodes() {
    if (state.sort !== 'weak' || SS_COL === -1) return CODES.slice();
    return CODES.slice().sort(function (a, b) {
      var av = num(row(state.task, a)[SS_COL]);
      var bv = num(row(state.task, b)[SS_COL]);
      if (av === null) return 1;
      if (bv === null) return -1;
      return av - bv;
    });
  }

  function defaultRow() {
    var w = weakest(state.task);
    return w && w.length ? w[0].code : CODES[0];
  }

  /* ---------- segmented controls ---------- */

  function segment(host, options, get, set) {
    clear(host);
    options.forEach(function (o) {
      var b = el('button', null, o.label);
      b.type = 'button';
      b.setAttribute('aria-pressed', String(get() === o.value));
      b.addEventListener('click', function () { set(o.value); });
      host.appendChild(b);
    });
  }

  function syncSegment(host, options, get) {
    var kids = host.children;
    for (var i = 0; i < kids.length && i < options.length; i++) {
      kids[i].setAttribute('aria-pressed', String(get() === options[i].value));
    }
  }

  var TASK_OPTS = TASKS.map(function (t) { return { value: t, label: TASK_LABEL[t] || t }; });
  var SORT_OPTS = [
    { value: 'table', label: 'Table order' },
    { value: 'weak', label: 'Weakest first' }
  ];

  /* ---------- colour key ---------- */

  function buildKey() {
    var host = pick('legend');
    if (!host) return;
    clear(host);
    var bar = el('div', 'ml-ramp');
    var STEPS = 26;
    for (var i = 0; i < STEPS; i++) {
      var s = el('span');
      s.style.background = ramp(LO + (HI - LO) * (i / (STEPS - 1)));
      bar.appendChild(s);
    }
    host.appendChild(bar);
    var ticks = el('div', 'ml-ticks');
    ticks.appendChild(el('span', null, f1(LO)));
    ticks.appendChild(el('span', null, f1((LO + HI) / 2)));
    ticks.appendChild(el('span', null, f1(HI)));
    host.appendChild(ticks);
    host.appendChild(el('div', 'ml-keynote', 'one scale, both tasks'));
  }

  /* ---------- heatmap ---------- */

  var cellIndex = {};   /* code + "|" + col -> node */

  function buildGrid() {
    var grid = pick('grid');
    if (!grid) return;
    clear(grid);
    cellIndex = {};
    grid.style.gridTemplateColumns = LABEL_W + ' repeat(' + MODELS.length + ', minmax(2.9rem, 1fr))';

    var head = el('div', 'ml-row');
    head.setAttribute('role', 'row');
    var corner = el('div', 'ml-corner', 'Language');
    corner.setAttribute('role', 'columnheader');
    head.appendChild(corner);
    MODELS.forEach(function (m, i) {
      var h = el('div', 'ml-mh' + (i === SS_COL ? ' ml-focusmodel' : ''));
      h.setAttribute('role', 'columnheader');
      h.setAttribute('data-col', String(i));
      h.appendChild(el('span', null, m));
      head.appendChild(h);
    });
    grid.appendChild(head);

    orderedCodes().forEach(function (code) {
      var vals = row(state.task, code);
      var bucket = isBucket(code);
      var r = el('div', 'ml-row');
      r.setAttribute('role', 'row');

      var rh = el('div', 'ml-rh' + (bucket ? ' ml-agg' : ''));
      rh.setAttribute('role', 'rowheader');
      rh.setAttribute('data-code', code);
      rh.appendChild(el('span', 'ml-name', LANGS[code]));
      if (!bucket && !isSupported(code)) {
        var mk = el('span', 'ml-mark', '†');
        mk.setAttribute('title', 'not one of the officially supported languages');
        rh.appendChild(mk);
      }
      rh.appendChild(el('span', 'ml-code', bucket ? 'pooled' : code));
      r.appendChild(rh);

      MODELS.forEach(function (m, i) {
        var v = num(vals[i]);
        var c = el('div', 'ml-cell' + (v === null ? ' ml-na' : ''), f1(v));
        c.setAttribute('role', 'gridcell');
        c.setAttribute('tabindex', '-1');
        c.setAttribute('data-code', code);
        c.setAttribute('data-col', String(i));
        c.setAttribute('aria-label', LANGS[code] + ', ' + m + ', ' + f1(v) + ' F1');
        if (v !== null) c.style.background = ramp(v);
        cellIndex[code + '|' + i] = c;
        r.appendChild(c);
      });

      grid.appendChild(r);
    });

    markSelection();
    syncHint();
  }

  function markSelection() {
    var grid = pick('grid');
    if (!grid) return;
    var codes = orderedCodes();
    if (codes.indexOf(state.rowCode) === -1) state.rowCode = codes[0];

    var i, k;
    for (k in cellIndex) {
      if (!Object.prototype.hasOwnProperty.call(cellIndex, k)) continue;
      var node = cellIndex[k];
      var on = (k === state.rowCode + '|' + state.col);
      node.className = 'ml-cell' + (node.textContent === 'n/a' ? ' ml-na' : '') + (on ? ' ml-sel' : '');
      node.setAttribute('tabindex', on ? '0' : '-1');
      node.setAttribute('aria-selected', on ? 'true' : 'false');
    }

    var rows = grid.children;
    for (i = 1; i < rows.length; i++) {
      var rh = rows[i].children[0];
      if (!rh) continue;
      var bucket = isBucket(rh.getAttribute('data-code'));
      rh.className = 'ml-rh' + (bucket ? ' ml-agg' : '') +
        (rh.getAttribute('data-code') === state.rowCode ? ' ml-rowsel' : '');
    }

    var heads = rows[0] ? rows[0].children : [];
    for (i = 1; i < heads.length; i++) {
      var ci = parseInt(heads[i].getAttribute('data-col'), 10);
      heads[i].className = 'ml-mh' + (ci === SS_COL ? ' ml-focusmodel' : '') +
        (ci === state.col && ci !== SS_COL ? ' ml-colsel' : '');
    }
  }

  function select(code, col, focus) {
    state.rowCode = code;
    state.col = col;
    state.userPicked = true;
    markSelection();
    renderDetail();
    if (focus) {
      var n = cellIndex[code + '|' + col];
      if (n && typeof n.focus === 'function') n.focus();
    }
  }

  function wireGrid() {
    var grid = pick('grid');
    if (!grid) return;

    grid.addEventListener('click', function (e) {
      var t = e.target;
      while (t && t !== grid && !(t.className && String(t.className).indexOf('ml-cell') === 0)) t = t.parentNode;
      if (!t || t === grid) return;
      select(t.getAttribute('data-code'), parseInt(t.getAttribute('data-col'), 10), true);
    });

    grid.addEventListener('keydown', function (e) {
      var key = e.key;
      var codes = orderedCodes();
      var ri = codes.indexOf(state.rowCode);
      var ci = state.col;
      var moved = false;
      if (key === 'ArrowRight') { ci = Math.min(MODELS.length - 1, ci + 1); moved = true; }
      else if (key === 'ArrowLeft') { ci = Math.max(0, ci - 1); moved = true; }
      else if (key === 'ArrowDown') { ri = Math.min(codes.length - 1, ri + 1); moved = true; }
      else if (key === 'ArrowUp') { ri = Math.max(0, ri - 1); moved = true; }
      else if (key === 'Home') { ci = 0; moved = true; }
      else if (key === 'End') { ci = MODELS.length - 1; moved = true; }
      else if (key === 'Enter' || key === ' ' || key === 'Spacebar') {
        select(state.rowCode, state.col, true);
        if (e.preventDefault) e.preventDefault();
        return;
      }
      if (!moved) return;
      if (e.preventDefault) e.preventDefault();
      select(codes[ri], ci, true);
    });
  }

  function syncHint() {
    var scroll = pick('scroll'), hint = pick('hint');
    if (!scroll || !hint) return;
    var over = (scroll.scrollWidth || 0) > (scroll.clientWidth || 0) + 2;
    hint.textContent = over ? 'Table scrolls sideways' : '';
  }

  /* ---------- weakness callout ---------- */

  function renderWeak() {
    var host = pick('weak');
    if (!host) return;
    clear(host);
    var list = weakest(state.task);
    if (!list || !list.length) return;

    var w = list[0];
    var vals = row(state.task, w.code);
    var b = bestOf(vals);
    var rk = rankOf(vals, w.value);
    var enVal = ENGLISH_CODE ? num(row(state.task, ENGLISH_CODE)[SS_COL]) : null;

    var top = el('div', 'ml-weak-top');
    top.appendChild(el('span', 'ml-weak-kick', 'Softest row'));
    var nameBtn = el('button', 'ml-weak-name', LANGS[w.code] + ', ' + TASK_PHRASE[state.task]);
    nameBtn.type = 'button';
    nameBtn.setAttribute('aria-label', 'Show the full ranking for ' + LANGS[w.code]);
    nameBtn.addEventListener('click', function () { select(w.code, SS_COL === -1 ? 0 : SS_COL, true); });
    top.appendChild(nameBtn);
    top.appendChild(el('span', 'ml-weak-val', f1(w.value) + ' F1'));
    host.appendChild(top);

    var stats = el('div', 'ml-stats');

    function stat(k, v, sub, neg) {
      var s = el('div', 'ml-stat');
      s.appendChild(el('span', 'ml-stat-k', k));
      var right = el('span', 'ml-stat-v' + (neg ? ' ml-neg' : ''));
      right.appendChild(document.createTextNode(v));
      if (sub) {
        var sn = el('span', 'ml-stat-sub', '  ' + sub);
        right.appendChild(sn);
      }
      s.appendChild(right);
      stats.appendChild(s);
    }

    if (enVal !== null && w.code !== ENGLISH_CODE) {
      stat('against its own English row', signed(w.value - enVal) + ' F1',
        '(' + f1(enVal) + ')', w.value < enVal);
    }
    if (b.index !== -1 && b.index !== SS_COL) {
      stat('against the leader on this row', signed(w.value - b.value) + ' F1',
        '(' + MODELS[b.index] + ', ' + f1(b.value) + ')', true);
    }
    stat('models ahead of it here', rk.ahead + ' of ' + MODELS.length, null, rk.ahead > 0);
    stat('officially supported language',
      isSupported(w.code) ? 'yes' : 'no', null, !isSupported(w.code));

    if (list.length > 2) {
      stat('next softest', LANGS[list[1].code] + ' ' + f1(list[1].value) +
        ', ' + LANGS[list[2].code] + ' ' + f1(list[2].value), null, false);
    }

    /* the same language on the other task, so the reader sees this is task specific */
    var other = TASKS[0] === state.task ? TASKS[1] : TASKS[0];
    var ov = num(row(other, w.code)[SS_COL]);
    if (ov !== null) {
      stat('same language, ' + String(TASK_LABEL[other] || other).toLowerCase(), f1(ov) + ' F1', null, false);
    }

    host.appendChild(stats);

    if (COVERAGE_QUOTE) {
      var q = el('p', 'ml-weak-quote');
      q.appendChild(el('b', null, 'Stated limitation: ' + COVERAGE_QUOTE.t));
      q.appendChild(document.createTextNode(COVERAGE_QUOTE.d));
      host.appendChild(q);
    }
  }

  /* ---------- detail ranking ---------- */

  function renderDetail() {
    var host = pick('detail');
    if (!host) return;
    clear(host);
    var code = state.rowCode;
    var vals = row(state.task, code);
    if (!vals.length) return;

    var head = el('div', 'ml-dhead');
    head.appendChild(el('span', 'ml-dkick', 'Full ranking'));
    head.appendChild(el('span', 'ml-dtitle', LANGS[code] + ', ' + TASK_PHRASE[state.task]));
    host.appendChild(head);

    var order = MODELS.map(function (m, i) { return { m: m, i: i, v: num(vals[i]) }; })
      .filter(function (o) { return o.v !== null; })
      .sort(function (a, b) { return b.v - a.v; });

    order.forEach(function (o) {
      var r = el('div', 'ml-rank' +
        (o.i === SS_COL ? ' ml-is-ss' : '') +
        (o.i === state.col ? ' ml-is-pick' : ''));
      r.appendChild(el('span', 'ml-rank-n', String(rankOf(vals, o.v).rank)));
      var mname = el('span', 'ml-rank-m');
      mname.appendChild(document.createTextNode(o.m));
      if (o.i === state.col && o.i !== SS_COL) mname.appendChild(el('span', 'ml-tag', 'picked'));
      r.appendChild(mname);
      var bar = el('div', 'ml-bar');
      var fill = el('i');
      fill.style.width = Math.max(0, Math.min(100, (o.v / HI) * 100)).toFixed(1) + '%';
      fill.style.background = ramp(o.v);
      bar.appendChild(fill);
      r.appendChild(bar);
      r.appendChild(el('span', 'ml-rank-v', f1(o.v)));
      host.appendChild(r);
    });

    if (SS_COL !== -1) {
      var sv = num(vals[SS_COL]);
      if (sv !== null) {
        var b = bestOf(vals);
        var rk = rankOf(vals, sv);
        var txt = 'Shieldstral places ' + rk.rank + ' of ' + order.length + ' on this row';
        if (b.index !== SS_COL) txt += ', ' + Math.abs(b.value - sv).toFixed(1) + ' F1 behind ' + MODELS[b.index];
        txt += '. Bars are drawn against the shared ceiling of ' + f1(HI) + '.';
        host.appendChild(el('p', 'ml-dnote', txt));
      }
    }
  }

  /* ---------- footer ---------- */

  function renderFoot() {
    var host = pick('foot');
    if (!host) return;
    clear(host);

    var rowNames = CODES.map(function (c) { return LANGS[c]; });

    var b1 = el('div', 'ml-fblock');
    b1.appendChild(el('span', 'ml-flab', 'Officially supported (' + SUPPORTED.length + ')'));
    var chips = el('div', 'ml-chips');
    var pooled = 0;
    SUPPORTED.forEach(function (name) {
      var own = rowNames.indexOf(name) !== -1;
      if (!own) pooled++;
      var c = el('span', 'ml-chip' + (own ? '' : ' ml-pooled'));
      c.appendChild(document.createTextNode(name));
      if (!own) c.appendChild(el('i', 'ml-chip-tag', 'pooled'));
      c.setAttribute('title', own ? 'has its own row above' : 'pooled into the aggregate row above');
      chips.appendChild(c);
    });
    b1.appendChild(chips);
    b1.appendChild(el('p', 'ml-fnote',
      'Solid border, the language has its own row above. Dashed border, it is folded into the pooled row (' +
      pooled + ' of ' + SUPPORTED.length + ').'));
    host.appendChild(b1);

    if (MULTI.length) {
      var b2 = el('div', 'ml-fblock');
      b2.appendChild(el('span', 'ml-flab', 'Multilingual evaluation sets'));
      var spans = el('div', 'ml-spans');
      MULTI.forEach(function (m) {
        var s = el('div', 'ml-span');
        s.appendChild(el('span', 'ml-span-n', m.name + ' (' + m.split + ')'));
        s.appendChild(el('span', 'ml-span-v', m.langs + ' languages, ' + thousands(m.items) + ' items'));
        spans.appendChild(s);
      });
      b2.appendChild(spans);
      var widest = MULTI[0].langs;
      var note = el('p', 'ml-fnote');
      note.appendChild(document.createTextNode(
        'The evaluation reaches ' + widest + ' languages, wider than the ' + SUPPORTED.length +
        ' the release supports. Ten languages get their own row here, the rest are folded into the '
        + 'pooled row, and the two lists do not line up in either direction. '));
      var mark = el('span', 'ml-key-mark', '†');
      note.appendChild(mark);
      note.appendChild(document.createTextNode(' above marks a row outside the supported set.'));
      b2.appendChild(note);
      host.appendChild(b2);
    }
  }

  /* ---------- boot ---------- */

  function refresh(rebuild) {
    if (!state.userPicked) state.rowCode = defaultRow();
    if (rebuild) buildGrid(); else markSelection();
    renderWeak();
    renderDetail();
  }

  var taskHost = pick('task'), sortHost = pick('sort');

  if (taskHost) {
    segment(taskHost, TASK_OPTS, function () { return state.task; }, function (v) {
      state.task = v;
      syncSegment(taskHost, TASK_OPTS, function () { return state.task; });
      refresh(true);
    });
  }
  if (sortHost) {
    segment(sortHost, SORT_OPTS, function () { return state.sort; }, function (v) {
      state.sort = v;
      syncSegment(sortHost, SORT_OPTS, function () { return state.sort; });
      refresh(true);
    });
  }

  buildKey();
  wireGrid();
  state.rowCode = defaultRow();
  buildGrid();
  renderWeak();
  renderDetail();
  renderFoot();

  if (typeof window.addEventListener === 'function') {
    var t = null;
    window.addEventListener('resize', function () {
      if (t) window.clearTimeout(t);
      t = window.setTimeout(syncHint, 120);
    });
  }
})();
