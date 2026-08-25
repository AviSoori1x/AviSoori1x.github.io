window.SCENES = window.SCENES || {};

/* Act III, scene 26. Where the capability actually comes from.
   SS.stageAblation holds three cumulative training stages measured on the
   fine-grained taxonomy validation set. A fourth bar is added from the
   0.6PG+0.3P+0.1I row of SS.merge, taxonomy side, which is the model that
   shipped. The picture is a stepped waterfall on a fixed 0 to 100 axis, so
   the base model's zero is visibly zero, with the delta between every pair
   of steps computed at runtime.
   The reader picks which of the four metrics in SS.stageAblation.cols is
   plotted, because precision falls while recall climbs and the waterfall
   changes shape completely between them.
   Every score, delta and name is read from window.SS at runtime. Nothing
   here is a live model call. */
window.SCENES['S_STAGES'] = function (root, api) {
  var SS = api.SS || {};
  var el = api.el;
  var num = api.num;

  var wrap = el('div', 'sc-s_stages');
  root.appendChild(wrap);

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
  function key(s) {
    return String(s == null ? '' : s).toLowerCase().replace(/[^a-z0-9]/g, '');
  }

  var A = SS.stageAblation || {};
  var COLS = (A.cols || []).slice();
  var AR = (A.rows || []).filter(function (r) { return r && r.vals && r.vals.length; });

  if (!COLS.length || AR.length < 2) {
    wrap.appendChild(el('p', 'miss',
      'SS.stageAblation does not carry the measured stages, so this waterfall cannot be drawn.'));
    return null;
  }

  /* ---------------- the merged model, taken from the merge table ---------------- */
  var MG = SS.merge || {};
  var MCOLS = (MG.cols || []).slice();
  var MROWS = (MG.rows || []).filter(function (r) { return r && r.taxonomy; });
  var WANT = '0.6PG+0.3P+0.1I';

  var mrow = null;
  MROWS.forEach(function (r) { if (key(r.name) === key(WANT)) mrow = r; });
  if (!mrow) {
    /* fall back to the recipe that blends the most ingredients */
    var most = -1;
    MROWS.forEach(function (r) {
      var k = String(r.name).split('+').length;
      if (k > most) { most = k; mrow = r; }
    });
  }
  /* the merge table is located by column name, never by assumed position */
  function mIdx(i) {
    var k = key(COLS[i]), j;
    for (j = 0; j < MCOLS.length; j++) if (key(MCOLS[j]) === k) return j;
    return i;
  }
  function taxOf(r) {
    return COLS.map(function (c, i) { return Number((r.taxonomy || [])[mIdx(i)]); });
  }

  /* ---------------- the four steps ---------------- */
  function split(name) {
    var m = /^(.*?)\s*\(([^)]*)\)\s*$/.exec(String(name));
    return m ? { main: m[1], sub: m[2] } : { main: String(name), sub: '' };
  }
  function shortOf(main) {
    var t = String(main).trim();
    if (t.charAt(0) === '+') return '+ ' + t.replace(/^\+\s*/, '').split(/\s+/)[0].toLowerCase();
    return t.split(/\s+/)[0].toLowerCase();
  }

  var STEPS = AR.map(function (r) {
    var s = split(r.name);
    return {
      main: s.main,
      sub: s.sub || 'cumulative training stage',
      short: shortOf(s.main),
      vals: (r.vals || []).map(Number),
      src: 'SS.stageAblation',
      fin: false
    };
  });
  if (mrow) {
    STEPS.push({
      main: '+ SLERP merge',
      sub: String(mrow.name),
      short: '+ merge',
      vals: taxOf(mrow),
      src: 'SS.merge, taxonomy side',
      fin: true
    });
  }
  var NS = STEPS.length;
  var NM = COLS.length;

  /* do the two trained stages appear verbatim in the merge table? if they do,
     the fourth bar is measured on the same set and belongs on the same axis */
  function same(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (!(Math.abs(Number(a[i]) - Number(b[i])) < 0.051)) return false;
    }
    return true;
  }
  var anchored = false;
  if (mrow && STEPS.length >= 3) {
    var hitP = false, hitPG = false;
    MROWS.forEach(function (r) {
      var v = taxOf(r);
      if (same(v, STEPS[1].vals)) hitP = true;
      if (same(v, STEPS[2].vals)) hitPG = true;
    });
    anchored = hitP && hitPG;
  }

  /* ---------------- metric columns, located by name ---------------- */
  var iAcc = -1, iPre = -1, iRec = -1, iF1 = -1;
  COLS.forEach(function (c, i) {
    var k = key(c);
    if (/^acc/.test(k)) iAcc = i;
    if (/^prec/.test(k)) iPre = i;
    if (/^rec/.test(k)) iRec = i;
    if (/^f1/.test(k)) iF1 = i;
  });
  var cur = iF1 >= 0 ? iF1 : NM - 1;

  function v(i, m) {
    var x = Number((STEPS[i].vals || [])[m]);
    return isFinite(x) ? x : null;
  }
  function dOf(i, m) {
    var a = v(i - 1, m), b = v(i, m);
    return (a == null || b == null) ? null : b - a;
  }
  function sgn(d) {
    if (d == null) return 'n/a';
    if (Math.abs(d) < 0.05) return '0.0';
    return (d > 0 ? '+' : '-') + num(Math.abs(d), 1);
  }

  /* ---------------- header ---------------- */
  var chips = '';
  for (var m0 = 0; m0 < NM; m0++) {
    chips += '<button type="button" class="chp" role="radio" aria-checked="false"'
      + ' tabindex="-1" id="S_STAGES-chp-' + m0 + '">' + esc(COLS[m0]) + '</button>';
  }

  var provenance = anchored
    ? 'The P and PG rows of SS.merge repeat these two trained stages score for score, '
      + 'which is what puts the merged model on the same axis.'
    : 'The fourth bar is read from SS.merge, taxonomy side, the same validation set.';

  var baseNote = '';
  if (iPre >= 0 && iRec >= 0 && v(0, iRec) === 0 && v(0, iPre) === 0) {
    baseNote = '<b>Zero here is not a small number.</b> Precision '
      + num(v(0, iPre), 1) + ' and recall ' + num(v(0, iRec), 1)
      + ' together mean the base model never answers yes on this set, so it catches nothing at '
      + 'all. Its accuracy of <b>' + num(v(0, iAcc >= 0 ? iAcc : 0), 1) + '</b> is then only the '
      + 'share of documents whose true label is no. A constant no is not partial competence. '
      + '<i>That last step is arithmetic on the published row, not a separate measurement.</i>';
  } else {
    baseNote = '<b>Read the base row carefully.</b> Accuracy on its own says nothing about '
      + 'whether a classifier ever fires.';
  }

  wrap.appendChild(api.frag(
    '<div class="hd">'
    +   '<div class="hl">'
    +     '<span class="eyebrow">stage ablation</span>'
    +     '<span class="hset">fine-grained taxonomy validation set</span>'
    +   '</div>'
    +   '<div class="mrow">'
    +     '<span class="mlab" id="S_STAGES-mlab">metric</span>'
    +     '<div class="chips" role="radiogroup" aria-labelledby="S_STAGES-mlab"'
    +       ' id="S_STAGES-chips">' + chips + '</div>'
    +   '</div>'
    + '</div>'

    + '<div class="chart" id="S_STAGES-chart"></div>'

    + '<div class="read">'
    +   '<div class="big">'
    +     '<span class="bk" id="S_STAGES-bk">after the merge</span>'
    +     '<b class="bn" id="S_STAGES-bn">0.0</b>'
    +     '<div class="brow">'
    +       '<span class="bs"><em>over base</em><b id="S_STAGES-d0">+0.0</b></span>'
    +       '<span class="bs"><em>over last stage</em><b id="S_STAGES-d1">+0.0</b></span>'
    +     '</div>'
    +     '<span class="bsrc" id="S_STAGES-bsrc"></span>'
    +   '</div>'
    +   '<div class="say">'
    +     '<span class="sk">the trade</span>'
    +     '<p class="st" id="S_STAGES-say"></p>'
    +     '<p class="sp">' + provenance + '</p>'
    +   '</div>'
    + '</div>'

    + '<p class="warn">' + baseNote + '</p>'

    + '<div class="foot">'
    +   '<span class="gt">Ablation validation set, not the Act IV benchmarks. Every score and '
    +   'every delta is read from SS.stageAblation and SS.merge at runtime, no live model '
    +   'call.</span>'
    +   '<span class="hint" id="S_STAGES-hint">cycling on its own, click a metric or use the '
    +   'arrow keys to take over</span>'
    + '</div>'
  ));

  /* ---------------- chart ---------------- */
  var chart = wrap.querySelector('#S_STAGES-chart');

  var G_WIDE = {
    W: 862, H: 404, ml: 54, mr: 22, mt: 48, mb: 92,
    fV: 25, fL: 12.5, fS: 9.5, fT: 10.5, fD: 13, fN: 10,
    bmax: 122, rw: 0.54, cw: 78, chH: 25, tri: 4.6
  };
  var G_NAR = {
    W: 380, H: 306, ml: 28, mr: 8, mt: 30, mb: 60,
    fV: 13.5, fL: 9, fS: 0, fT: 8, fD: 9.5, fN: 8,
    bmax: 44, rw: 0.44, cw: 40, chH: 17, tri: 3.4
  };
  var narrow = matchMedia('(max-width: 46rem)');

  function drawChart() {
    var G = narrow.matches ? G_NAR : G_WIDE;
    var pw = G.W - G.ml - G.mr, ph = G.H - G.mt - G.mb;
    var colw = pw / NS;
    var barw = Math.min(colw * G.rw, G.bmax);
    var yy = function (t) {
      var c = Math.max(0, Math.min(100, t));
      return G.mt + ph * (1 - c / 100);
    };
    var cxi = function (i) { return G.ml + colw * (i + 0.5); };
    var y0 = yy(0);
    var s = '', i, t;

    /* grid, fixed 0 to 100 so the zero row is visibly zero */
    for (t = 0; t <= 100; t += 20) {
      var gy = yy(t).toFixed(1);
      s += '<line class="gl' + (t === 0 ? ' zero' : '') + '" x1="' + G.ml + '" y1="' + gy
        + '" x2="' + (G.ml + pw).toFixed(1) + '" y2="' + gy + '"></line>';
      s += '<text class="tk" x="' + (G.ml - 8) + '" y="' + gy + '" dy=".33em" text-anchor="end"'
        + ' font-size="' + G.fT + '">' + t + '</text>';
    }

    /* step connectors, the ledger line of a waterfall */
    for (i = 0; i + 1 < NS; i++) {
      var a = v(i, cur);
      if (a == null || v(i + 1, cur) == null) continue;
      s += '<line class="cn" x1="' + (cxi(i) + barw / 2).toFixed(1) + '" y1="' + yy(a).toFixed(1)
        + '" x2="' + (cxi(i + 1) + barw / 2).toFixed(1) + '" y2="' + yy(a).toFixed(1) + '"></line>';
    }

    /* bars */
    for (i = 0; i < NS; i++) {
      var val = v(i, cur);
      var x = (cxi(i) - barw / 2).toFixed(1);
      var fin = STEPS[i].fin ? ' fin' : '';
      if (val == null) {
        s += '<text class="val na" x="' + cxi(i).toFixed(1) + '" y="' + (y0 - 10).toFixed(1)
          + '" text-anchor="middle" font-size="' + G.fL + '">not published</text>';
        continue;
      }
      var base = i === 0 ? 0 : v(i - 1, cur);
      if (base == null) base = 0;

      if (i === 0) {
        if (val > 0) {
          s += '<rect class="bar b0" x="' + x + '" y="' + yy(val).toFixed(1) + '" width="'
            + barw.toFixed(1) + '" height="' + (y0 - yy(val)).toFixed(1) + '" rx="2"></rect>';
        }
      } else if (val >= base) {
        if (base > 0) {
          s += '<rect class="bar bcar' + fin + '" x="' + x + '" y="' + yy(base).toFixed(1)
            + '" width="' + barw.toFixed(1) + '" height="' + (y0 - yy(base)).toFixed(1)
            + '" rx="2"></rect>';
        }
        s += '<rect class="bar bgain' + fin + '" x="' + x + '" y="' + yy(val).toFixed(1)
          + '" width="' + barw.toFixed(1) + '" height="' + Math.max(1.5, yy(base) - yy(val)).toFixed(1)
          + '" rx="2"></rect>';
      } else {
        s += '<rect class="bar bcar' + fin + '" x="' + x + '" y="' + yy(val).toFixed(1)
          + '" width="' + barw.toFixed(1) + '" height="' + (y0 - yy(val)).toFixed(1)
          + '" rx="2"></rect>';
        s += '<rect class="bar bloss" x="' + x + '" y="' + yy(base).toFixed(1) + '" width="'
          + barw.toFixed(1) + '" height="' + Math.max(1.5, yy(val) - yy(base)).toFixed(1)
          + '" rx="2"></rect>';
      }

      /* a bar of no height still needs to be seen */
      if (val <= 0.0001) {
        s += '<line class="zbar" x1="' + x + '" y1="' + y0.toFixed(1) + '" x2="'
          + (cxi(i) + barw / 2).toFixed(1) + '" y2="' + y0.toFixed(1) + '"></line>';
      }

      /* a step that loses ground carries a dashed block above the bar, so its
         value label moves inside the solid part rather than through the dashes */
      var lossAbove = val < base - 0.05;
      var room = (y0 - yy(val)) > G.fV * 1.8;
      var vy = (lossAbove && room) ? yy(val) + G.fV + 3 : yy(val) - 11;
      if (lossAbove && !room) vy = yy(base) - 11;
      s += '<text class="val' + (val <= 0.0001 ? ' z' : '') + '" x="' + cxi(i).toFixed(1)
        + '" y="' + vy.toFixed(1) + '" text-anchor="middle" font-size="' + G.fV
        + '">' + num(val, 1) + '</text>';

      /* the base row gets a word, because a number alone misleads here */
      if (i === 0 && G.fS && iRec >= 0 && v(0, iRec) === 0) {
        s += '<text class="flag" x="' + cxi(i).toFixed(1) + '" y="'
          + (yy(val) - 11 - G.fV * 0.95).toFixed(1) + '" text-anchor="middle" font-size="'
          + G.fS + '">' + (val <= 0.0001 ? 'never answers yes' : 'always answers no')
          + '</text>';
      }

      /* column names under the axis */
      s += '<text class="nm" x="' + cxi(i).toFixed(1) + '" y="' + (y0 + 22).toFixed(1)
        + '" text-anchor="middle" font-size="' + G.fL + '">'
        + esc(G.fS ? STEPS[i].main : STEPS[i].short) + '</text>';
      if (G.fS) {
        s += '<text class="nms" x="' + cxi(i).toFixed(1) + '" y="' + (y0 + 38).toFixed(1)
          + '" text-anchor="middle" font-size="' + G.fS + '">' + esc(STEPS[i].sub) + '</text>';
        if (STEPS[i].fin) {
          s += '<text class="tag" x="' + cxi(i).toFixed(1) + '" y="' + (y0 + 56).toFixed(1)
            + '" text-anchor="middle" font-size="' + G.fS + '">the model that shipped</text>';
        }
      }
    }

    /* delta chips between the steps */
    for (i = 1; i < NS; i++) {
      var d = dOf(i, cur);
      if (d == null) continue;
      var up = d > 0.05, dn = d < -0.05;
      var cxg = G.ml + colw * i;
      var ymid = (yy(v(i - 1, cur)) + yy(v(i, cur))) / 2;
      ymid = Math.max(G.mt + G.chH / 2, Math.min(y0 - G.chH / 2, ymid));
      var cls = up ? 'up' : dn ? 'dn' : 'lv';
      var tx = cxg - G.cw / 2 + 8 + G.tri * 2 + 4;
      s += '<g class="chip ' + cls + '">'
        + '<rect x="' + (cxg - G.cw / 2).toFixed(1) + '" y="' + (ymid - G.chH / 2).toFixed(1)
        + '" width="' + G.cw + '" height="' + G.chH + '" rx="7"></rect>';
      if (up) {
        s += '<path class="tri" d="M' + (cxg - G.cw / 2 + 8 + G.tri) + ',' + (ymid - G.tri)
          + ' l' + G.tri + ',' + (G.tri * 1.7) + ' l' + (-G.tri * 2) + ',0 Z"></path>';
      } else if (dn) {
        s += '<path class="tri" d="M' + (cxg - G.cw / 2 + 8 + G.tri) + ',' + (ymid + G.tri)
          + ' l' + G.tri + ',' + (-G.tri * 1.7) + ' l' + (-G.tri * 2) + ',0 Z"></path>';
      } else {
        s += '<rect class="tri" x="' + (cxg - G.cw / 2 + 8) + '" y="' + (ymid - 1) + '" width="'
          + (G.tri * 2) + '" height="2"></rect>';
      }
      s += '<text x="' + tx.toFixed(1) + '" y="' + ymid.toFixed(1) + '" dy=".34em" font-size="'
        + G.fD + '">' + sgn(d) + '</text></g>';
    }

    s += '<text class="axn" x="' + G.ml + '" y="' + (G.mt - 16) + '" font-size="' + G.fN + '">'
      + esc(COLS[cur]) + (G.fS ? ' on the taxonomy validation set' : ', axis 0 to 100')
      + '</text>';
    if (G.fS) {
      s += '<text class="axn" x="' + (G.ml + pw).toFixed(1) + '" y="' + (G.mt - 16)
        + '" text-anchor="end" font-size="' + G.fN
        + '">axis fixed 0 to 100 for every metric</text>';
    }

    var desc = STEPS.map(function (st, k) {
      var val = v(k, cur);
      return st.main + ' ' + (val == null ? 'not published' : num(val, 1))
        + (k ? ', a change of ' + sgn(dOf(k, cur)) : '');
    }).join('. ');

    chart.innerHTML = '<svg class="plot" viewBox="0 0 ' + G.W + ' ' + G.H + '" role="img"'
      + ' aria-label="Stepped waterfall of ' + esc(COLS[cur])
      + ' on the fine-grained taxonomy validation set. ' + esc(desc) + '.">' + s + '</svg>';
  }

  /* ---------------- readout ---------------- */
  var bk = wrap.querySelector('#S_STAGES-bk');
  var bn = wrap.querySelector('#S_STAGES-bn');
  var d0 = wrap.querySelector('#S_STAGES-d0');
  var d1 = wrap.querySelector('#S_STAGES-d1');
  var bsrc = wrap.querySelector('#S_STAGES-bsrc');
  var sayEl = wrap.querySelector('#S_STAGES-say');
  var hint = wrap.querySelector('#S_STAGES-hint');
  var chipBox = wrap.querySelector('#S_STAGES-chips');
  var chipEls = [];
  for (var c0 = 0; c0 < NM; c0++) chipEls.push(wrap.querySelector('#S_STAGES-chp-' + c0));

  var LAST = NS - 1;

  function moveWord(a, b) {
    if (a == null || b == null) return 'moves';
    if (b < a - 0.05) return 'falls';
    if (b > a + 0.05) return 'climbs';
    return 'holds';
  }

  function sayFor(m) {
    var gen = NS >= 3 ? 2 : NS - 1;
    var pre = iPre >= 0 ? [v(gen - 1, iPre), v(gen, iPre)] : null;
    var rec = iRec >= 0 ? [v(gen - 1, iRec), v(gen, iRec)] : null;
    var out = '';
    if (pre && rec && pre[0] != null && rec[0] != null) {
      out += 'Adding <b>' + esc(STEPS[gen].main.replace(/^\+\s*/, '')) + '</b>, precision '
        + moveWord(pre[0], pre[1]) + ' from ' + num(pre[0], 1) + ' to <b>' + num(pre[1], 1)
        + '</b> while recall ' + moveWord(rec[0], rec[1]) + ' from ' + num(rec[0], 1)
        + ' to <b>' + num(rec[1], 1) + '</b>. The model starts firing, and it fires more '
        + 'often than it should. ';
      if (STEPS[LAST].fin && v(LAST, iPre) != null) {
        out += 'The merge buys the precision back to ' + num(v(LAST, iPre), 1)
          + ' and settles recall at ' + num(v(LAST, iRec), 1) + '. ';
      }
    }
    var dl = dOf(LAST, m);
    if (dl != null) {
      out += 'On <b>' + esc(COLS[m]) + '</b> the last step is <b>' + sgn(dl) + '</b>.';
    }
    return out;
  }

  function setMetric(idx, byUser) {
    if (idx < 0 || idx >= NM || idx === cur) {
      if (byUser) takeOver();
      return;
    }
    cur = idx;
    for (var j = 0; j < NM; j++) {
      chipEls[j].setAttribute('aria-checked', j === idx ? 'true' : 'false');
      chipEls[j].tabIndex = j === idx ? 0 : -1;
    }
    drawChart();

    var last = v(LAST, idx), first = v(0, idx), prev = v(LAST - 1, idx);
    bk.textContent = String(COLS[idx]) + (STEPS[LAST].fin ? ' after the merge' : ' at the last step');
    bn.textContent = last == null ? 'n/a' : num(last, 1);
    d0.textContent = (last == null || first == null) ? 'n/a' : sgn(last - first);
    d1.textContent = (last == null || prev == null) ? 'n/a' : sgn(last - prev);
    d0.className = (last != null && first != null && last < first) ? 'dn' : 'up';
    d1.className = (last != null && prev != null && last < prev) ? 'dn' : 'up';
    bsrc.textContent = STEPS[LAST].sub + ' · ' + STEPS[LAST].src;
    sayEl.innerHTML = sayFor(idx);
    if (byUser) takeOver();
  }

  var auto = !api.reduce;
  function takeOver() {
    if (!auto) return;
    auto = false;
    hint.textContent = 'manual, arrow keys move between the four metrics';
  }

  chipEls.forEach(function (b, j) {
    b.addEventListener('click', function () { setMetric(j, true); });
  });
  chipBox.addEventListener('keydown', function (e) {
    var k = e.key, nx = -1;
    if (k === 'ArrowLeft' || k === 'ArrowUp') nx = (cur - 1 + NM) % NM;
    else if (k === 'ArrowRight' || k === 'ArrowDown') nx = (cur + 1) % NM;
    else if (k === 'Home') nx = 0;
    else if (k === 'End') nx = NM - 1;
    if (nx < 0) return;
    e.preventDefault();
    setMetric(nx, true);
    chipEls[nx].focus();
  });
  if (narrow.addEventListener) narrow.addEventListener('change', drawChart);
  else if (narrow.addListener) narrow.addListener(drawChart);

  if (api.reduce) hint.textContent = 'pick a metric, motion is off';

  var open = cur;
  cur = -1;
  setMetric(open, false);

  var nextAt = null, running = false;
  return {
    start: function () { running = true; nextAt = null; drawChart(); },
    stop: function () { running = false; },
    tick: function (t) {
      if (!running || !auto || api.reduce) return;
      if (nextAt === null) { nextAt = t + 3.6; return; }
      if (t >= nextAt) {
        nextAt = t + 3.6;
        setMetric((cur + 1) % NM, false);
      }
    }
  };
};
