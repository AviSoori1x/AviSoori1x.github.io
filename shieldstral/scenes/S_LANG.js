window.SCENES = window.SCENES || {};

/* S_LANG, act 4, beat 32. The language holes.

   Appendix A of the report gives F1 for eleven language buckets against ten
   guardrails, once for prompt classification and once for response
   classification. This draws that table as a heatmap, languages down and models
   across, with the number printed in every cell as well as coloured, because a
   heatmap you cannot read the values off is a decoration.

   Two things here are computed rather than asserted.

   First, weakness. A cell is called weak when it sits ten F1 or more below that
   same model's own median across the eleven buckets. That is a relative rule,
   so a model that is poor everywhere does not light up as a language hole, and
   a strong model that drops in one place does. The count is recomputed for
   whichever view is showing.

   Second, official support. window.SS.languages carries the twelve languages
   the model card claims. Cross that against the eleven buckets in the table and
   Indonesian and the Others bucket are the two rows that are not on the list.
   They are also exactly where Shieldstral's prompt score falls apart. Nothing
   asserts that connection, the scene derives it from the two arrays at runtime.

   Every number on screen is read from window.SS. */
window.SCENES['S_LANG'] = function (root, api) {
  var SS = api.SS || {};
  var M = SS.multilingual || {};
  var models = (M.models || []).slice();
  var langMap = M.langs || {};
  var codes = Object.keys(langMap);
  var SC = M.scores || {};

  root.classList.add('sc-s_lang');

  if (!models.length || !codes.length || !(SC.prompt || SC.response)) {
    root.appendChild(api.frag('<div class="lg-empty">window.SS.multilingual is not in the '
      + 'data file, so there is no table to draw.</div>').firstChild);
    return;
  }

  /* ---------------- views ---------------- */
  var views = [];
  if (SC.prompt) views.push({ k: 'prompt', t: 'prompt classification', s: 'prompt' });
  if (SC.response) views.push({ k: 'response', t: 'response classification', s: 'response' });

  /* ---------------- short column labels ----------------
     The heatmap header is vertical and the ranking list underneath carries the
     full name, so the column only needs enough to identify the model. Anything
     not in this map falls through to the full string from the data file. */
  var SHORT = {
    'ShieldGemma-9B': ['ShieldGemma', '9B'],
    'WildGuard-7B': ['WildGuard', '7B'],
    'LlamaGuard-4-12B': ['LlamaGuard-4', '12B'],
    'PolyGuard-Qwen-7B': ['PolyGuard', '7B'],
    'Qwen3Guard-8B': ['Qwen3Guard', '8B'],
    'Nemotron-Safety-8B': ['Nemotron', '8B'],
    'Nemotron-3.5-Safety-4B': ['Nemotron-3.5', '4B'],
    'OmniGuard-7B': ['OmniGuard', '7B'],
    'GPT-OSS-Safeguard-20B': ['GPT-OSS', '20B'],
    'Shieldstral-3B': ['Shieldstral', '3B']
  };
  function shortOf(m) {
    if (SHORT[m]) return SHORT[m];
    var bits = String(m).split('-');
    var tail = bits[bits.length - 1];
    return /^\d/.test(tail) ? [bits.slice(0, -1).join('-'), tail] : [m, ''];
  }

  var OURS = -1, mi;
  for (mi = 0; mi < models.length; mi++) {
    if (/^shieldstral/i.test(models[mi])) { OURS = mi; }
  }
  if (OURS < 0) OURS = models.length - 1;
  var OURNAME = models[OURS];

  /* ---------------- which buckets are on the supported list ---------------- */
  var official = (SS.languages || []).map(function (s) { return String(s).toLowerCase(); });
  var supported = {}, offList = [];
  codes.forEach(function (c) {
    var full = String(langMap[c] || c);
    var ok = official.indexOf(full.toLowerCase()) >= 0;
    supported[c] = ok;
    if (!ok) offList.push(c);
  });
  var hasOfficial = official.length > 0;

  /* ---------------- statistics, all derived from the table ---------------- */
  var HOLE = 10;   /* F1 below a model's own median before a cell counts as weak */

  function median(a) {
    var b = a.slice().sort(function (x, y) { return x - y; });
    var n = b.length;
    if (!n) return 0;
    return (n % 2) ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2;
  }

  var lo = Infinity, hi = -Infinity;
  var stats = {};
  views.forEach(function (v) {
    var tab = SC[v.k];
    var med = models.map(function (m, i) {
      return median(codes.map(function (c) { return tab[c][i]; }));
    });
    var cells = {}, holes = 0, holeLangs = {};
    codes.forEach(function (c) {
      var row = tab[c];
      var byScore = row.map(function (x, i) { return i; }).sort(function (a, b) {
        return row[b] - row[a];
      });
      var rank = [];
      byScore.forEach(function (idx, j) { rank[idx] = j + 1; });
      cells[c] = row.map(function (x, i) {
        var d = med[i] - x;
        var hole = d >= HOLE;
        if (hole) { holes++; holeLangs[c] = (holeLangs[c] || 0) + 1; }
        if (x < lo) lo = x;
        if (x > hi) hi = x;
        return { v: x, def: d, hole: hole, rank: rank[i] };
      });
    });
    var oursRow = codes.map(function (c) { return tab[c][OURS]; });
    var worst = codes[0], k;
    for (k = 0; k < codes.length; k++) {
      if (tab[codes[k]][OURS] < tab[worst][OURS]) worst = codes[k];
    }
    stats[v.k] = {
      med: med, cells: cells, holes: holes, holeLangs: holeLangs,
      ours: oursRow, worst: worst
    };
  });
  if (!(hi > lo)) { lo = 0; hi = 100; }

  /* ---------------- colour, one scale shared by both views ---------------- */
  var BASE = [26, 32, 48];        /* --panel2 */
  var LIME = [182, 242, 77];
  var ROSE = [255, 122, 151];

  function mix(base, c, a) {
    return [
      Math.round(base[0] + (c[0] - base[0]) * a),
      Math.round(base[1] + (c[1] - base[1]) * a),
      Math.round(base[2] + (c[2] - base[2]) * a)
    ];
  }
  function lum(c) { return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; }
  function rgb(c) { return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')'; }

  function paint(cell) {
    var t = (cell.v - lo) / (hi - lo);
    if (t < 0) t = 0; if (t > 1) t = 1;
    var a, col, ink;
    if (cell.hole) {
      a = 0.14 + 0.30 * Math.min(1, cell.def / 34);
      col = mix(BASE, ROSE, a);
      ink = '#ff9fb3';
    } else {
      a = 0.045 + 0.66 * Math.pow(t, 1.75);
      col = mix(BASE, LIME, a);
      ink = (lum(col) > 122) ? '#080a10' : (t > 0.42 ? '#e6f0d3' : '#7d8798');
    }
    return { bg: rgb(col), fg: ink, t: t };
  }

  /* ---------------- state ---------------- */
  var view = views[0].k;
  var sortMode = 'paper';
  var selLang = stats[view].worst;
  var selModel = OURS;
  var order = codes.slice();

  /* ---------------- markup ---------------- */
  function segHtml(id, label, opts) {
    var s = '<div class="lg-seg" role="group" aria-label="' + label + '">';
    opts.forEach(function (o) {
      s += '<button type="button" class="lg-sgb" id="S_LANG-' + id + '-' + o.k + '"'
        + ' data-k="' + o.k + '" aria-pressed="false">' + o.t + '</button>';
    });
    return s + '</div>';
  }

  var headHtml = '<div class="lg-row lg-hrow" role="row">'
    + '<span class="lg-lab lg-hlab" role="columnheader"><b>lang</b></span>';
  models.forEach(function (m, i) {
    var sh = shortOf(m);
    headHtml += '<span class="lg-h' + (i === OURS ? ' lg-hours' : '') + '" role="columnheader"'
      + ' title="' + m + '"><b>' + sh[0] + '</b><em>' + sh[1] + '</em></span>';
  });
  headHtml += '</div>';

  var rowsHtml = '';
  codes.forEach(function (c) {
    rowsHtml += '<div class="lg-row lg-drow" role="row" data-c="' + c + '">'
      + '<span class="lg-lab" role="rowheader">'
      + '<b>' + c + '</b>'
      + (String(langMap[c]) === c ? '' : '<i>' + langMap[c] + '</i>')
      + (hasOfficial && !supported[c]
        ? '<u class="lg-off" title="not one of the ' + official.length
          + ' supported languages in the data file">off list</u>' : '')
      + '</span>';
    models.forEach(function (m, i) {
      rowsHtml += '<button type="button" class="lg-c" role="gridcell" tabindex="-1"'
        + ' data-c="' + c + '" data-m="' + i + '"><span></span></button>';
    });
    rowsHtml += '</div>';
  });

  var rankHtml = '';
  models.forEach(function (m, i) {
    rankHtml += '<div class="lg-rk" id="S_LANG-rk-' + i + '">'
      + '<i></i><b>' + m + '</b>'
      + '<span class="lg-rb"><u></u></span><em></em></div>';
  });

  root.appendChild(api.frag(
    '<div class="lg-wrap">'

    + '<div class="lg-hd">'
    +   '<div class="lg-hl">'
    +     '<span class="lg-eyebrow">appendix a &nbsp;&middot;&nbsp; f1 by language</span>'
    +     '<span class="lg-hnote">' + codes.length + ' language buckets, ' + models.length
    +     ' guardrails, every score printed</span>'
    +   '</div>'
    +   '<div class="lg-ctl">'
    +     segHtml('v', 'Which classification task to show',
      views.map(function (v) { return { k: v.k, t: v.s }; }))
    +     segHtml('s', 'Row order', [
      { k: 'paper', t: 'table order' },
      { k: 'weak', t: 'weakest first' }
    ])
    +   '</div>'
    + '</div>'

    + '<div class="lg-grid" id="S_LANG-grid" role="grid"'
    +   ' aria-label="F1 by language and model. Arrow keys move, the reading below updates.">'
    +   headHtml
    +   '<div class="lg-body" id="S_LANG-body" role="rowgroup">' + rowsHtml + '</div>'
    + '</div>'

    + '<div class="lg-leg">'
    +   '<span class="lg-scale" aria-hidden="true"><u></u>'
    +     '<em id="S_LANG-slo"></em><em id="S_LANG-shi"></em></span>'
    +   '<span class="lg-lk lg-lkw"><i aria-hidden="true"></i>'
    +     '<b id="S_LANG-holes"></b></span>'
    +   (hasOfficial
      ? '<span class="lg-lk"><u>off list</u><b id="S_LANG-offc"></b></span>' : '')
    + '</div>'

    + '<div class="lg-detail">'
    +   '<div class="lg-focus" id="S_LANG-focus">'
    +     '<span class="lg-flab" id="S_LANG-flab"></span>'
    +     '<b class="lg-big" id="S_LANG-big">0.0</b>'
    +     '<span class="lg-fsub" id="S_LANG-fsub"></span>'
    +     '<span class="lg-mini" id="S_LANG-mini"></span>'
    +   '</div>'
    +   '<div class="lg-rank">'
    +     '<div class="lg-rhd"><b id="S_LANG-rhd"></b>'
    +       '<span>full ranking, best first</span></div>'
    +     '<div class="lg-rlist">' + rankHtml + '</div>'
    +   '</div>'
    + '</div>'

    + '<p class="lg-sr" id="S_LANG-say" role="status"></p>'

    + '<div class="lg-foot">'
    +   '<span class="lg-gt" id="S_LANG-gt"></span>'
    +   '<span class="lg-hint">click any cell, or arrow keys once one is focused</span>'
    + '</div>'

    + '</div>'
  ).firstChild);

  /* ---------------- handles ---------------- */
  var $ = function (id) { return root.querySelector('#S_LANG-' + id); };
  var body = $('body');
  var say = $('say');
  var big = $('big'), flab = $('flab'), fsub = $('fsub'), mini = $('mini');
  var rhd = $('rhd');
  var holesEl = $('holes'), offcEl = $('offc'), gt = $('gt');

  $('grid').style.setProperty('--nm', models.length);

  var rowEl = {}, cellEl = {};
  [].slice.call(body.children).forEach(function (r) {
    var c = r.getAttribute('data-c');
    rowEl[c] = r;
    cellEl[c] = [].slice.call(r.querySelectorAll('.lg-c'));
  });
  var rankEl = models.map(function (m, i) { return $('rk-' + i); });

  $('slo').textContent = api.num(lo, 1);
  $('shi').textContent = api.num(hi, 1);
  if (offcEl) {
    // "Others" is an aggregate whose members the table never names, so it cannot be
    // called unsupported wholesale. Only name the buckets we can actually check.
    var named = offList.filter(function (c) { return String(langMap[c]) !== 'Others'; });
    offcEl.textContent = named.map(function (c) { return langMap[c]; }).join(' and ')
      + (named.length === 1 ? ' is not in the ' : ' are not in the ')
      + official.length + '-language support list. Others is an aggregate bucket and '
      + 'this table does not name its constituent languages.';
  }

  /* ---------------- cell fill ---------------- */
  function paintAll() {
    var st = stats[view];
    codes.forEach(function (c) {
      cellEl[c].forEach(function (b, i) {
        var cell = st.cells[c][i];
        var p = paint(cell);
        b.style.background = p.bg;
        b.style.color = p.fg;
        b.className = 'lg-c'
          + (cell.hole ? ' lg-hole' : '')
          + (i === OURS ? ' lg-ours' : '')
          + (p.t > 0.86 && !cell.hole ? ' lg-top' : '');
        b.firstChild.textContent = api.num(cell.v, 1);
        b.setAttribute('aria-label',
          langMap[c] + ', ' + models[i] + ', ' + view + ' F1 ' + api.num(cell.v, 1)
          + ', rank ' + cell.rank + ' of ' + models.length + ' in this language'
          + (cell.hole
            ? ', weak, ' + api.num(cell.def, 1) + ' below this model\'s own median'
            : ''));
      });
    });
    holesEl.textContent = 'weak, ' + st.holes + ' of ' + (codes.length * models.length)
      + ' cells, ' + HOLE + ' F1 or more under their own model\'s median';
    gt.textContent = 'Scores read from the data file. Weak is computed here, not a label the '
      + 'report gives: ' + HOLE + ' F1 under that model\'s own median. Others is an aggregate.';
  }

  /* ---------------- selection ---------------- */
  function selectedCell() { return stats[view].cells[selLang][selModel]; }

  function paintSel() {
    codes.forEach(function (c) {
      cellEl[c].forEach(function (b, i) {
        var on = (c === selLang && i === selModel);
        b.classList.toggle('lg-sel', on);
        b.tabIndex = on ? 0 : -1;
        b.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      rowEl[c].classList.toggle('lg-rsel', c === selLang);
    });
  }

  function paintDetail() {
    var st = stats[view];
    var cell = st.cells[selLang][selModel];
    var vname = views.filter(function (v) { return v.k === view; })[0].t;

    flab.textContent = langMap[selLang] + '  ·  ' + vname + '  ·  F1';
    big.textContent = api.num(cell.v, 1);
    big.className = 'lg-big' + (cell.hole ? ' lg-rose' : '');
    fsub.innerHTML = '<b>' + models[selModel] + '</b>, rank ' + cell.rank + ' of '
      + models.length;

    function mrow(k, v, cls) {
      return '<span class="lg-mrow' + (cls ? ' ' + cls : '') + '"><i>' + k + '</i>'
        + '<b>' + v + '</b></span>';
    }
    var rows = mrow('its own median, ' + codes.length + ' buckets', api.num(st.med[selModel], 1));
    if (cell.hole) {
      rows += mrow('under that median', '&minus;' + api.num(cell.def, 1), 'lg-mbad');
    }
    var other = views.filter(function (v) { return v.k !== view; })[0];
    if (other) {
      var ov = stats[other.k].cells[selLang][selModel].v;
      var d = ov - cell.v;
      rows += mrow('same cell, ' + other.s, api.num(ov, 1)
        + ' <u>' + (d >= 0 ? '+' : '&minus;') + api.num(Math.abs(d), 1) + '</u>');
    }
    if (hasOfficial) {
      rows += mrow('on the ' + official.length + ' supported languages',
        supported[selLang] ? 'yes' : 'no', supported[selLang] ? '' : 'lg-mwarn');
    }
    mini.innerHTML = rows;

    rhd.textContent = langMap[selLang] + ', ' + vname;
    var row = st.cells[selLang];
    var byRank = models.map(function (m, i) { return i; }).sort(function (a, b) {
      return row[a].rank - row[b].rank;
    });
    var list = rankEl[0].parentNode;
    byRank.forEach(function (idx) {
      var e = rankEl[idx];
      var c = row[idx];
      e.className = 'lg-rk'
        + (idx === selModel ? ' lg-rsel2' : '')
        + (idx === OURS ? ' lg-rours' : '')
        + (c.hole ? ' lg-rhole' : '');
      e.children[0].textContent = c.rank;
      e.children[2].firstChild.style.width = Math.max(1, c.v) + '%';
      e.children[3].textContent = api.num(c.v, 1);
      list.appendChild(e);
    });
  }

  function announce() {
    var cell = selectedCell();
    say.textContent = langMap[selLang] + ', ' + models[selModel] + ', ' + view + ' F1 '
      + api.num(cell.v, 1) + ', rank ' + cell.rank + ' of ' + models.length + '.';
  }

  function select(c, m, focus) {
    selLang = c;
    selModel = m;
    paintSel();
    paintDetail();
    if (focus) cellEl[c][m].focus();
    announce();
  }

  /* ---------------- ordering, with a flip so the weak rows visibly rise ------ */
  function targetOrder() {
    if (sortMode !== 'weak') return codes.slice();
    var o = stats[view].ours;
    return codes.slice().sort(function (a, b) {
      return o[codes.indexOf(a)] - o[codes.indexOf(b)];
    });
  }

  function applyOrder(animate) {
    var next = targetOrder();
    var same = next.length === order.length && next.every(function (c, i) {
      return c === order[i];
    });
    if (same && body.children.length === codes.length) return;

    var first = {}, canAnim = animate && !api.reduce;
    if (canAnim) {
      codes.forEach(function (c) {
        var r = rowEl[c].getBoundingClientRect();
        first[c] = r.top;
        if (!r.height) canAnim = false;
      });
    }
    next.forEach(function (c) { body.appendChild(rowEl[c]); });
    order = next;
    if (!canAnim) return;

    var moved = [];
    codes.forEach(function (c) {
      var d = first[c] - rowEl[c].getBoundingClientRect().top;
      if (!d) return;
      rowEl[c].style.transition = 'none';
      rowEl[c].style.transform = 'translateY(' + d + 'px)';
      moved.push(c);
    });
    if (!moved.length) return;
    void body.offsetWidth;
    moved.forEach(function (c, i) {
      rowEl[c].style.transition = 'transform .52s cubic-bezier(.22,.61,.36,1) '
        + (i * 12) + 'ms';
      rowEl[c].style.transform = '';
    });
  }

  /* ---------------- controls ---------------- */
  function paintSeg() {
    views.forEach(function (v) {
      var b = $('v-' + v.k);
      b.setAttribute('aria-pressed', v.k === view ? 'true' : 'false');
    });
    ['paper', 'weak'].forEach(function (k) {
      var b = $('s-' + k);
      b.setAttribute('aria-pressed', k === sortMode ? 'true' : 'false');
    });
  }

  views.forEach(function (v) {
    $('v-' + v.k).addEventListener('click', function () {
      if (view === v.k) return;
      view = v.k;
      paintSeg();
      paintAll();
      paintSel();
      paintDetail();
      applyOrder(true);
      announce();
    });
  });
  ['paper', 'weak'].forEach(function (k) {
    $('s-' + k).addEventListener('click', function () {
      if (sortMode === k) return;
      sortMode = k;
      paintSeg();
      applyOrder(true);
      say.textContent = (k === 'weak')
        ? 'Rows ordered by ' + OURNAME + ' on ' + view + ', weakest first. '
          + langMap[stats[view].worst] + ' is now at the top at '
          + api.num(stats[view].cells[stats[view].worst][OURS].v, 1) + '.'
        : 'Rows back in the order the table gives.';
    });
  });

  body.addEventListener('click', function (e) {
    var b = e.target.closest ? e.target.closest('.lg-c') : null;
    if (!b || !body.contains(b)) return;
    select(b.getAttribute('data-c'), +b.getAttribute('data-m'), true);
  });

  body.addEventListener('keydown', function (e) {
    var b = e.target.closest ? e.target.closest('.lg-c') : null;
    if (!b) return;
    var c = b.getAttribute('data-c'), m = +b.getAttribute('data-m');
    var vi = order.indexOf(c), nm = m, nc = c, k = e.key;
    if (k === 'ArrowRight') nm = Math.min(models.length - 1, m + 1);
    else if (k === 'ArrowLeft') nm = Math.max(0, m - 1);
    else if (k === 'ArrowDown') nc = order[Math.min(order.length - 1, vi + 1)];
    else if (k === 'ArrowUp') nc = order[Math.max(0, vi - 1)];
    else if (k === 'Home') nm = 0;
    else if (k === 'End') nm = models.length - 1;
    else if (k === 'PageUp') nc = order[0];
    else if (k === 'PageDown') nc = order[order.length - 1];
    else return;
    e.preventDefault();
    select(nc, nm, true);
  });

  /* ---------------- first paint ---------------- */
  paintSeg();
  paintAll();
  paintSel();
  paintDetail();
  announce();
};
