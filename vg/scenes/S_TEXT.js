window.SCENES = window.SCENES || {};

/* Act IV, scene 29. Text safety benchmarks.
   SS.benchmarks.promptClassification and SS.benchmarks.responseClassification,
   drawn as one grouped horizontal bar block per benchmark row. Shieldstral is
   always the first bar and the only lime one, the baselines are muted and are
   keyed by parameter count, which is matched back against SS.baselines so the
   3B against 20B comparison is visible on every single row. The best value in
   each row carries a diamond marker and the word best, so the win is never
   signalled by colour alone, and the losses stay in the reading order of the
   paper. Nothing here is hardcoded, every number is read from window.SS. */
window.SCENES['S_TEXT'] = function (root, api) {
  var SS = api.SS || {};
  var B = SS.benchmarks || {};
  var H = SS.headline || {};
  var BASE = SS.baselines || [];

  root.classList.add('sc-s_text');

  var TASKS = [
    { key: 'prompt', tab: 'prompt classification', noun: 'prompt-classification', src: B.promptClassification },
    { key: 'response', tab: 'response classification', noun: 'response-classification', src: B.responseClassification }
  ];

  function usable(t) {
    return t && t.src && t.src.models && t.src.models.length && t.src.rows && t.src.rows.length;
  }
  if (!TASKS.every(usable)) {
    root.appendChild(api.frag('<div class="wrap"><p class="miss">'
      + 'SS.benchmarks.promptClassification or SS.benchmarks.responseClassification is not present '
      + 'in the data file, so this comparison cannot be drawn.</p></div>'));
    return null;
  }

  /* ---------------- small helpers ---------------- */

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function norm(s) {
    return String(s == null ? '' : s).toLowerCase()
      .replace(/\(ours\)/g, '').replace(/[^a-z0-9.]/g, '');
  }
  function splitName(full) {
    var m = /^(.*?)[-\s](\d+(?:\.\d+)?B)$/i.exec(String(full == null ? '' : full));
    return m ? { base: m[1], size: m[2] } : { base: String(full == null ? '' : full), size: null };
  }
  function parseB(sz) {
    var m = /(\d+(?:\.\d+)?)\s*B/i.exec(String(sz == null ? '' : sz));
    return m ? parseFloat(m[1]) : null;
  }

  /* model identity, with the parameter count matched against the paper's
     baseline table rather than trusted from the benchmark column header */
  function describe(full) {
    var sp = splitName(full);
    var key = norm(sp.base);
    var hit = null, i;
    for (i = 0; i < BASE.length; i++) {
      if (norm(BASE[i].model) === key) { hit = BASE[i]; break; }
    }
    var size = hit ? hit.size : sp.size;
    return {
      full: full,
      base: sp.base,
      size: size || null,
      b: parseB(size),
      matched: !!hit,
      adaptive: hit ? !!hit.adaptive : null
    };
  }

  var META = TASKS[0].src.models.map(describe);
  var unmatched = META.filter(function (m) { return !m.matched; }).length;
  var sizesShown = META.filter(function (m) { return m.size; }).length;

  /* ---------------- per row analysis ---------------- */

  function analyse(t) {
    return t.src.rows.map(function (r) {
      var vals = r.vals || [];
      var best = -Infinity, i;
      for (i = 0; i < vals.length; i++) if (vals[i] > best) best = vals[i];
      var winners = [];
      for (i = 0; i < vals.length; i++) if (vals[i] === best) winners.push(i);
      var rival = -1, rivalV = -Infinity;
      for (i = 1; i < vals.length; i++) if (vals[i] > rivalV) { rivalV = vals[i]; rival = i; }
      return {
        name: r.name,
        vals: vals,
        best: best,
        winners: winners,
        usBest: winners.indexOf(0) >= 0,
        tie: winners.indexOf(0) >= 0 && winners.length > 1,
        rival: rival,
        rivalV: rivalV,
        delta: vals[0] - rivalV
      };
    });
  }

  var ANA = TASKS.map(analyse);

  /* one shared floor so the two tabs stay visually comparable */
  var lo = Infinity;
  ANA.forEach(function (rows) {
    rows.forEach(function (r) {
      r.vals.forEach(function (v) { if (v < lo) lo = v; });
    });
  });
  var FLOOR = Math.max(0, Math.floor(lo / 10) * 10);
  var TOP = 100;

  var totRows = 0, totWins = 0;
  ANA.forEach(function (rows) {
    rows.forEach(function (r) { totRows++; if (r.usBest) totWins++; });
  });
  var totLoss = totRows - totWins;

  function pct(v) {
    return Math.max(0, Math.min(100, ((v - FLOOR) / (TOP - FLOOR)) * 100));
  }

  /* ---------------- glyphs ---------------- */

  var DIA = '<svg class="gly dia" viewBox="0 0 10 10" aria-hidden="true">'
    + '<path d="M5 .5 9.5 5 5 9.5.5 5Z"></path></svg>';
  var DOWN = '<svg class="gly car" viewBox="0 0 10 10" aria-hidden="true">'
    + '<path d="M1.5 3.4 5 7 8.5 3.4" fill="none" stroke="currentColor" stroke-width="1.7"'
    + ' stroke-linecap="round" stroke-linejoin="round"></path></svg>';

  /* ---------------- shell ---------------- */

  root.appendChild(api.frag(
    '<div class="wrap" id="S_TEXT-wrap">'

    + '<div class="hd">'
    +   '<span class="eyeb">text benchmarks<i>F1</i></span>'
    +   '<span class="ctrls">'
    +     '<span class="seg" role="group" aria-label="which benchmark table to show">'
    +       TASKS.map(function (t, i) {
            return '<button type="button" class="sg' + (i === 0 ? ' sel' : '') + '"'
              + ' id="S_TEXT-tab-' + i + '" aria-pressed="' + (i === 0) + '">'
              + esc(t.tab) + '</button>';
          }).join('')
    +     '</span>'
    +     '<button type="button" class="flt" id="S_TEXT-filter" aria-pressed="false">'
    +       '<span class="box" aria-hidden="true"></span>'
    +       '<span>only rows it does not win</span>'
    +     '</button>'
    +   '</span>'
    + '</div>'

    + '<div class="lede">'
    +   '<span class="numeral" id="S_TEXT-mean">0.0</span>'
    +   '<span class="ledeside" id="S_TEXT-heroside"></span>'
    + '</div>'

    + '<div class="roster" id="S_TEXT-roster" role="group"'
    +   ' aria-label="models compared, parameter count and mean F1 over the rows shown"></div>'

    + '<div class="scale" id="S_TEXT-scale"></div>'

    + '<div class="grid" id="S_TEXT-grid"></div>'

    + '<p class="foot" id="S_TEXT-foot"></p>'

    + '</div>'
  ));

  var wrap = root.querySelector('#S_TEXT-wrap');
  var meanEl = root.querySelector('#S_TEXT-mean');
  var sideEl = root.querySelector('#S_TEXT-heroside');
  var rosterEl = root.querySelector('#S_TEXT-roster');
  var gridEl = root.querySelector('#S_TEXT-grid');
  var scaleEl = root.querySelector('#S_TEXT-scale');
  var footEl = root.querySelector('#S_TEXT-foot');
  var filtBtn = root.querySelector('#S_TEXT-filter');

  var taskIx = 0;
  var filterOn = false;
  var pinned = -1;

  /* gridlines inside every track, positions derived from the live floor */
  var TICKS = [];
  (function () {
    var v = Math.ceil((FLOOR + 1) / 10) * 10, p;
    for (; v < TOP; v += 10) {
      p = (v - FLOOR) / (TOP - FLOOR);
      if (p > 0.12 && p < 0.8) TICKS.push(v);
    }
  })();
  var trackBg = TICKS.map(function (v) {
    var p = pct(v).toFixed(3) + '%';
    return 'linear-gradient(90deg,transparent ' + p + ',rgba(255,255,255,.13) ' + p
      + ',rgba(255,255,255,.13) calc(' + p + ' + 1px),transparent calc(' + p + ' + 1px))';
  }).join(',');

  scaleEl.innerHTML = '<span class="sk">bar length</span>'
    + '<span class="rail" style="background-image:' + trackBg + '">'
    +   '<i class="lo">' + api.num(FLOOR, 0) + '</i>'
    +   TICKS.map(function (v) {
          return '<i class="tk" style="left:' + pct(v).toFixed(2) + '%">' + api.num(v, 0) + '</i>';
        }).join('')
    +   '<i class="hi">' + api.num(TOP, 0) + '</i>'
    + '</span>'
    + '<span class="sk r">F1. Cut off below ' + api.num(FLOOR, 0)
    + ', so trust the printed number, not the bar.</span>';

  /* ---------------- one benchmark block ---------------- */

  function barRow(r, i) {
    var m = META[i];
    var v = r.vals[i];
    var isUs = (i === 0);
    var isWin = r.winners.indexOf(i) >= 0;
    var cls = 'br' + (isUs ? ' us' : '') + (isWin ? ' win' : '');
    var sizeTxt = m.size ? m.size : 'size n/a';
    return '<li class="' + cls + '" data-ix="' + i + '">'
      + '<span class="sz">' + esc(sizeTxt) + '</span>'
      + '<span class="mn" title="' + esc(m.full) + '">' + esc(m.base) + '</span>'
      + '<span class="vh">' + esc(m.base) + ' at ' + esc(sizeTxt) + ', </span>'
      + '<span class="track" style="background-image:' + trackBg + '" aria-hidden="true">'
      +   '<i class="fill" data-w="' + pct(v).toFixed(2) + '" style="width:0%"></i>'
      + '</span>'
      + '<span class="v">' + api.num(v, 1) + '</span>'
      + '<span class="mk">' + (isWin ? DIA + '<b>best</b>' : '') + '</span>'
      + '</li>';
  }

  function block(r, ix, all) {
    var rival = META[r.rival] || { base: '', size: '' };
    var d = Math.abs(r.delta);
    var tag = r.usBest
      ? '<span class="tag win">' + DIA + (r.tie ? 'tied best' : 'best')
          + ', ' + api.num(d, 1) + ' clear of ' + esc(rival.size || rival.base) + '</span>'
      : '<span class="tag lose">' + DOWN + api.num(d, 1) + ' behind '
          + esc(rival.base) + ' ' + esc(rival.size || '') + '</span>';
    var last = (all.length % 2 === 1) && (ix === all.length - 1);
    return '<div class="bk ' + (r.usBest ? 'w' : 'l') + (last ? ' span2' : '')
      + '" role="group" aria-label="' + esc(r.name) + '">'
      + '<div class="bkhd"><span class="bn">' + esc(r.name) + '</span>' + tag + '</div>'
      + '<ul class="bars">'
      + r.vals.map(function (_, i) { return barRow(r, i); }).join('')
      + '</ul>'
      + '</div>';
  }

  /* ---------------- roster ---------------- */

  function roster(rows) {
    return META.map(function (m, i) {
      var s = 0;
      rows.forEach(function (r) { s += r.vals[i]; });
      var mean = rows.length ? s / rows.length : 0;
      return '<button type="button" class="mchip' + (i === 0 ? ' us' : '') + '"'
        + ' data-ix="' + i + '" aria-pressed="false">'
        + '<span class="csz">' + esc(m.size || '?') + '</span>'
        + '<span class="cnm">' + esc(m.base) + '</span>'
        + '<span class="cmn">' + api.num(mean, 1)
        + '<i> mean over the ' + rows.length + ' rows shown</i></span>'
        + '</button>';
    }).join('');
  }

  /* ---------------- paint ---------------- */

  function paint(animate) {
    var t = TASKS[taskIx];
    var all = ANA[taskIx];
    var rows = filterOn ? all.filter(function (r) { return !r.usBest; }) : all;
    var lossHere = all.filter(function (r) { return !r.usBest; }).length;
    var winHere = all.length - lossHere;

    wrap.classList.toggle('filtered', filterOn);
    filtBtn.setAttribute('aria-pressed', filterOn ? 'true' : 'false');

    if (!rows.length) {
      gridEl.innerHTML = '<p class="empty">Shieldstral is top of the table in every '
        + esc(t.noun) + ' row, so this filter leaves nothing to show.</p>';
      rosterEl.innerHTML = roster(all);
    } else {
      gridEl.innerHTML = rows.map(block).join('');
      rosterEl.innerHTML = roster(rows);
    }

    /* the big numeral is Shieldstral's mean over exactly the rows on screen */
    var use = rows.length ? rows : all;
    var s0 = 0;
    use.forEach(function (r) { s0 += r.vals[0]; });
    var mean0 = s0 / use.length;

    var bestOther = -Infinity, bestOtherIx = -1;
    META.forEach(function (m, i) {
      if (i === 0) return;
      var s = 0;
      use.forEach(function (r) { s += r.vals[i]; });
      var mu = s / use.length;
      if (mu > bestOther) { bestOther = mu; bestOtherIx = i; }
    });
    var rival = META[bestOtherIx] || { base: '', size: '', b: null };
    var gap = mean0 - bestOther;
    var level = Math.abs(gap) < 0.05;
    var ratio = (META[0].b && rival.b) ? (rival.b / META[0].b) : null;

    meanEl.textContent = api.num(mean0, 1);
    meanEl.classList.toggle('down', gap < -0.05);

    sideEl.innerHTML =
      '<b>mean F1 for ' + esc(META[0].base) + ' at ' + esc(META[0].size || 'unknown size')
      + '</b>'
      + '<span class="l1">across the <b>' + use.length + '</b> '
      + esc(t.noun) + ' row' + (use.length === 1 ? '' : 's') + ' shown'
      + (filterOn ? ', which is the losing subset only' : '') + '</span>'
      + '<span class="l2">next best is <b>' + api.num(bestOther, 1) + '</b> from '
      + esc(rival.base) + ' at <b>' + esc(rival.size || 'unknown size') + '</b>'
      + (ratio ? ', <b>' + api.num(ratio, 1) + 'x</b> the parameters' : '')
      + '<span class="gap ' + (level ? 'lv' : (gap > 0 ? 'up' : 'dn')) + '">'
      + (level ? 'level' : ((gap > 0 ? '+' : '') + api.num(gap, 1))) + '</span></span>'
      + '<span class="l3">top of the table in <b>' + winHere + '</b> of the '
      + all.length + ' ' + esc(t.noun) + ' rows, and <b>' + totWins + '</b> of the '
      + totRows + ' rows across both tables. The other <b>' + totLoss + '</b> are losses.'
      + '</span>';

    footEl.innerHTML =
      'Every value is the paper\'s reported F1 for that benchmark split, read from the data file '
      + 'at load time, not a live model call. Bars start at ' + api.num(FLOOR, 0)
      + ' rather than 0 so that gaps of a tenth of a point stay visible, which exaggerates them, '
      + 'so the printed number is the thing to trust. The mean above is computed over the rows on '
      + 'screen and is not the paper\'s headline text average of <b>' + api.num(H.textF1, 1)
      + '</b> across ' + esc(H.splits) + ' splits. Parameter counts are matched by name against '
      + 'the paper\'s baseline table, ' + sizesShown + ' of ' + META.length + ' resolved'
      + (unmatched ? ', ' + unmatched + ' fell back to the size in the column header' : '') + '.';

    bind();
    grow(animate);
  }

  /* fills go from zero to their value once the block is in the DOM */
  function grow(animate) {
    var fills = gridEl.querySelectorAll('.fill');
    function set() {
      for (var i = 0; i < fills.length; i++) {
        fills[i].style.width = fills[i].getAttribute('data-w') + '%';
      }
    }
    if (api.reduce || !animate) {
      gridEl.classList.add('nofx');
      set();
      /* let the width land before transitions are allowed back */
      requestAnimationFrame(function () { gridEl.classList.remove('nofx'); });
    } else {
      requestAnimationFrame(set);
    }
  }

  /* ---------------- isolate one model ---------------- */

  function iso(ix) {
    var rows = gridEl.querySelectorAll('.br'), i, k;
    for (i = 0; i < rows.length; i++) {
      k = +rows[i].getAttribute('data-ix');
      rows[i].classList.toggle('dim', ix >= 0 && k !== ix);
      rows[i].classList.toggle('hi', ix >= 0 && k === ix);
    }
    var chips = rosterEl.querySelectorAll('.mchip');
    for (i = 0; i < chips.length; i++) {
      k = +chips[i].getAttribute('data-ix');
      chips[i].classList.toggle('dim', ix >= 0 && k !== ix);
    }
    wrap.classList.toggle('iso', ix >= 0);
  }

  function bind() {
    var chips = rosterEl.querySelectorAll('.mchip');
    Array.prototype.forEach.call(chips, function (c) {
      var k = +c.getAttribute('data-ix');
      c.setAttribute('aria-pressed', k === pinned ? 'true' : 'false');
      c.addEventListener('mouseenter', function () { if (pinned < 0) iso(k); });
      c.addEventListener('mouseleave', function () { if (pinned < 0) iso(-1); });
      c.addEventListener('focus', function () { if (pinned < 0) iso(k); });
      c.addEventListener('blur', function () { if (pinned < 0) iso(-1); });
      c.addEventListener('click', function () {
        pinned = (pinned === k) ? -1 : k;
        Array.prototype.forEach.call(chips, function (o) {
          o.setAttribute('aria-pressed', (+o.getAttribute('data-ix')) === pinned ? 'true' : 'false');
        });
        iso(pinned);
      });
    });
    iso(pinned);
  }

  /* ---------------- controls ---------------- */

  TASKS.forEach(function (t, i) {
    root.querySelector('#S_TEXT-tab-' + i).addEventListener('click', function () {
      if (taskIx === i) return;
      taskIx = i;
      TASKS.forEach(function (_, j) {
        var b = root.querySelector('#S_TEXT-tab-' + j);
        b.classList.toggle('sel', j === i);
        b.setAttribute('aria-pressed', j === i ? 'true' : 'false');
      });
      paint(true);
    });
  });

  filtBtn.addEventListener('click', function () {
    filterOn = !filterOn;
    paint(true);
  });

  paint(true);

  return null;
};
