window.SCENES = window.SCENES || {};

/* Act IV, scene 30. Multimodal safety benchmarks.
   SS.benchmarks.multimodal, three splits by six models, drawn with the same
   grouped horizontal bar treatment as the text scene so the two read as one
   family. Shieldstral is always the first bar and the only lime one. Two rows
   are wins and one is a loss, and the loss is given the same weight as the
   wins, a rose block edge, the size of the gap spelled out in the block header,
   and a visible note that some of that split's test images were unavailable at
   evaluation time. A second view flips every bar to the signed gap against
   Shieldstral around a zero line, which puts the single negative bar on screen
   as the only thing pointing left. Nothing is hardcoded, every number is read
   from window.SS at runtime and the average is recomputed from the rows and
   checked against the paper's headline figure. */
window.SCENES['S_MM'] = function (root, api) {
  var SS = api.SS || {};
  var B = SS.benchmarks || {};
  var MM = B.multimodal;
  var H = SS.headline || {};
  var BASE = SS.baselines || [];
  var INV = SS.benchInventory || [];

  root.classList.add('sc-s_mm');

  if (!MM || !MM.models || !MM.models.length || !MM.rows || !MM.rows.length) {
    root.appendChild(api.frag('<div class="wrap"><p class="miss">'
      + 'SS.benchmarks.multimodal is not present in the data file, so this comparison '
      + 'cannot be drawn.</p></div>'));
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
  function signed(v, dp) {
    return (v >= 0 ? '+' : '-') + api.num(Math.abs(v), dp == null ? 1 : dp);
  }

  /* model identity, with the parameter count matched against the paper's
     baseline table rather than trusted from the results column header */
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
      input: hit ? hit.input : null
    };
  }

  var META = MM.models.map(describe);
  var US = META[0] || { base: 'Shieldstral', size: null, b: null };
  var matched = META.filter(function (m) { return m.matched; }).length;
  var strays = META.filter(function (m) { return !m.matched; })
    .map(function (m) { return m.full; });

  /* the paper's own evaluation inventory, so each split can say how big it is */
  function inv(name) {
    var key = norm(name), i;
    for (i = 0; i < INV.length; i++) {
      if (norm(INV[i][0]) === key) return { kind: INV[i][1], lang: INV[i][2], n: INV[i][3] };
    }
    return null;
  }

  /* caveats that belong to one split and must stay next to its bars */
  var NOTES = {
    llavaguard: 'Some of this split’s test images were no longer available at evaluation time, '
      + 'so every model in this row, Shieldstral included, is scored on the same available '
      + 'subset rather than the full inventory count.'
  };

  /* ---------------- per row analysis ---------------- */

  var ROWS = MM.rows.map(function (r) {
    var vals = r.vals || [];
    var best = -Infinity, i;
    for (i = 0; i < vals.length; i++) if (vals[i] > best) best = vals[i];
    var winners = [];
    for (i = 0; i < vals.length; i++) if (vals[i] === best) winners.push(i);
    var rival = -1, rivalV = -Infinity;
    for (i = 1; i < vals.length; i++) if (vals[i] > rivalV) { rivalV = vals[i]; rival = i; }
    var rank = 1;
    for (i = 1; i < vals.length; i++) if (vals[i] > vals[0]) rank++;
    return {
      name: r.name,
      vals: vals,
      winners: winners,
      usBest: winners.indexOf(0) >= 0,
      tie: winners.indexOf(0) >= 0 && winners.length > 1,
      rival: rival,
      rivalV: rivalV,
      delta: vals[0] - rivalV,
      rank: rank,
      inv: inv(r.name),
      note: NOTES[norm(r.name)] || null
    };
  });

  function mean(ix) {
    var s = 0;
    ROWS.forEach(function (r) { s += r.vals[ix]; });
    return s / ROWS.length;
  }

  var WINS = ROWS.filter(function (r) { return r.usBest; }).length;
  var LOSSES = ROWS.length - WINS;

  var ourMean = mean(0);
  var bestOther = -Infinity, bestOtherIx = -1;
  META.forEach(function (m, i) {
    if (i === 0) return;
    var mu = mean(i);
    if (mu > bestOther) { bestOther = mu; bestOtherIx = i; }
  });
  var runner = META[bestOtherIx] || { base: '', size: null, b: null };
  var meanGap = ourMean - bestOther;
  var ratio = (US.b && runner.b) ? (runner.b / US.b) : null;

  /* the average is recomputed here, then checked against the paper's headline */
  var headline = (typeof H.multimodalF1 === 'number') ? H.multimodalF1 : null;
  var agrees = (headline != null) && Math.abs(headline - ourMean) < 0.05;

  /* ---------------- scales ---------------- */

  var TOP = 100;
  function pct(v) { return Math.max(0, Math.min(100, (v / TOP) * 100)); }

  var dLo = 0, dHi = 0;
  ROWS.forEach(function (r) {
    r.vals.forEach(function (v, i) {
      if (i === 0) return;
      var d = r.vals[0] - v;
      if (d < dLo) dLo = d;
      if (d > dHi) dHi = d;
    });
  });
  var DLO = Math.floor(dLo / 5) * 5;
  var DHI = Math.ceil(dHi / 5) * 5;
  if (DHI - DLO < 5) DHI = DLO + 5;
  var DSPAN = DHI - DLO;
  var ZERO = ((0 - DLO) / DSPAN) * 100;
  function dwid(d) { return (Math.abs(d) / DSPAN) * 100; }

  /* ---------------- gridlines ---------------- */

  function gline(p, col) {
    var s = p.toFixed(3) + '%';
    return 'linear-gradient(90deg,transparent ' + s + ',' + col + ' ' + s + ',' + col
      + ' calc(' + s + ' + 1px),transparent calc(' + s + ' + 1px))';
  }

  var TICKS_F1 = [];
  (function () { for (var v = 20; v < TOP; v += 20) TICKS_F1.push(v); })();
  var BG_F1 = TICKS_F1.map(function (v) {
    return gline(pct(v), 'rgba(255,255,255,.13)');
  }).join(',');

  var TICKS_D = [];
  (function () {
    for (var v = Math.ceil(DLO / 10) * 10; v <= DHI; v += 10) if (v !== 0) TICKS_D.push(v);
  })();
  var BG_D = TICKS_D.map(function (v) {
    return gline(((v - DLO) / DSPAN) * 100, 'rgba(255,255,255,.1)');
  }).concat([gline(ZERO, 'rgba(255,255,255,.5)')]).join(',');

  /* ---------------- glyphs ---------------- */

  var DIA = '<svg class="gly dia" viewBox="0 0 10 10" aria-hidden="true">'
    + '<path d="M5 .5 9.5 5 5 9.5.5 5Z"></path></svg>';
  var DOWN = '<svg class="gly car" viewBox="0 0 10 10" aria-hidden="true">'
    + '<path d="M1.5 3.4 5 7 8.5 3.4" fill="none" stroke="currentColor" stroke-width="1.7"'
    + ' stroke-linecap="round" stroke-linejoin="round"></path></svg>';
  var UP = '<svg class="gly car" viewBox="0 0 10 10" aria-hidden="true">'
    + '<path d="M1.5 6.6 5 3 8.5 6.6" fill="none" stroke="currentColor" stroke-width="1.7"'
    + ' stroke-linecap="round" stroke-linejoin="round"></path></svg>';
  var INFO = '<svg class="gly inf" viewBox="0 0 12 12" aria-hidden="true">'
    + '<circle cx="6" cy="6" r="5.1" fill="none" stroke="currentColor" stroke-width="1.1"></circle>'
    + '<path d="M6 5.1v3.4M6 3.3v.9" stroke="currentColor" stroke-width="1.3"'
    + ' stroke-linecap="round"></path></svg>';

  /* ---------------- shell ---------------- */

  var VIEWS = ['score', 'gap to ' + US.base];

  root.appendChild(api.frag(
    '<div class="wrap" id="S_MM-wrap">'

    + '<div class="hd">'
    +   '<span class="eyeb">multimodal benchmarks<i>F1, ' + ROWS.length + ' splits</i></span>'
    +   '<span class="seg" role="group" aria-label="how to draw the bars">'
    +     VIEWS.map(function (v, i) {
          return '<button type="button" class="sg' + (i === 0 ? ' sel' : '') + '"'
            + ' id="S_MM-view-' + i + '" aria-pressed="' + (i === 0) + '">'
            + esc(v) + '</button>';
        }).join('')
    +   '</span>'
    + '</div>'

    + '<div class="lede">'
    +   '<span class="numeral">' + api.num(ourMean, 1) + '</span>'
    +   '<span class="ledeside" id="S_MM-side"></span>'
    + '</div>'

    + '<div class="roster" id="S_MM-roster" role="group"'
    +   ' aria-label="models compared, parameter count and mean F1 across the three splits"></div>'

    + '<div class="scale" id="S_MM-scale"></div>'

    + '<div class="grid" id="S_MM-grid"></div>'

    + '<p class="foot" id="S_MM-foot"></p>'

    + '</div>'
  ));

  var wrap = root.querySelector('#S_MM-wrap');
  var sideEl = root.querySelector('#S_MM-side');
  var rosterEl = root.querySelector('#S_MM-roster');
  var scaleEl = root.querySelector('#S_MM-scale');
  var gridEl = root.querySelector('#S_MM-grid');
  var footEl = root.querySelector('#S_MM-foot');

  var view = 0;
  var pinned = -1;

  /* ---------------- the standing text ---------------- */

  var lossRow = null;
  ROWS.forEach(function (r) { if (!r.usBest && !lossRow) lossRow = r; });

  sideEl.innerHTML =
    '<b>mean F1 for ' + esc(US.base) + ' at ' + esc(US.size || 'unknown size')
    + ' across the ' + ROWS.length + ' multimodal splits</b>'
    + '<span class="l1">next best is <b>' + api.num(bestOther, 1) + '</b> from '
    + esc(runner.base) + ' at <b>' + esc(runner.size || 'unknown size') + '</b>'
    + (ratio ? ', <b>' + api.num(ratio, 1) + 'x</b> the parameters' : '')
    + '<span class="gap ' + (meanGap >= 0 ? 'up' : 'dn') + '">' + signed(meanGap) + '</span></span>'
    + '<span class="l2">top of the table on <b>' + WINS + '</b> of ' + ROWS.length
    + ', and ' + (lossRow
        ? 'it loses <b>' + esc(lossRow.name) + '</b> by <b>'
          + api.num(Math.abs(lossRow.delta), 1) + '</b> to '
          + esc((META[lossRow.rival] || {}).full || 'the leader')
        : 'it loses none of them') + '</span>'
    + '<span class="l3">'
    + (headline == null
        ? 'no headline multimodal average in the data file to check this against'
        : (agrees
            ? 'recomputed from the ' + ROWS.length + ' rows below, agrees with the paper’s '
              + 'reported <b>' + api.num(headline, 1) + '</b>'
            : 'recomputed from the ' + ROWS.length + ' rows below as <b>' + api.num(ourMean, 1)
              + '</b>, which does not match the paper’s reported <b>'
              + api.num(headline, 1) + '</b>'))
    + '</span>';

  footEl.innerHTML =
    'Paper-reported F1, read from the data file at load time, not a live model call. Split sizes '
    + 'are the paper’s evaluation inventory. Parameter counts are matched against its baseline '
    + 'table, ' + matched + ' of ' + META.length + ' resolved'
    + (strays.length
        ? ', ' + strays.map(function (s) { return esc(s); }).join(', ')
          + ' is not in it so that size comes from the results column header'
        : '') + '.';

  /* ---------------- scale strip ---------------- */

  function paintScale() {
    if (view === 0) {
      scaleEl.innerHTML = '<span class="sk">bar length</span>'
        + '<span class="rail" style="background-image:' + BG_F1 + '">'
        +   '<i class="lo">0</i>'
        +   TICKS_F1.map(function (v) {
              return '<i class="tk" style="left:' + pct(v).toFixed(2) + '%">' + v + '</i>';
            }).join('')
        +   '<i class="hi">100</i>'
        + '</span>'
        + '<span class="sk r">F1 on the split, full axis, no truncation</span>';
    } else {
      scaleEl.innerHTML = '<span class="sk">bar length</span>'
        + '<span class="rail" style="background-image:' + BG_D + '">'
        +   '<i class="lo">' + signed(DLO, 0) + '</i>'
        +   '<i class="tk zz" style="left:' + ZERO.toFixed(2) + '%">0</i>'
        +   '<i class="hi">' + signed(DHI, 0) + '</i>'
        + '</span>'
        + '<span class="sk r">F1 points, ' + esc(US.base) + ' minus that model. Left of the zero '
        + 'line means ' + esc(US.base) + ' loses.</span>';
    }
  }

  /* ---------------- bars ---------------- */

  function scoreBar(r, i) {
    var m = META[i];
    var v = r.vals[i];
    var isUs = (i === 0);
    var isWin = r.winners.indexOf(i) >= 0;
    var sizeTxt = m.size ? m.size : 'size n/a';
    return '<li class="br' + (isUs ? ' us' : '') + (isWin ? ' win' : '') + '" data-ix="' + i + '">'
      + '<span class="sz">' + esc(sizeTxt) + '</span>'
      + '<span class="mn" title="' + esc(m.full) + '">' + esc(m.base) + '</span>'
      + '<span class="vh">' + esc(m.base) + ' at ' + esc(sizeTxt) + ' scores '
      +   api.num(v, 1) + (isWin ? ', best in this split' : '') + '. </span>'
      + '<span class="track" style="background-image:' + BG_F1 + '" aria-hidden="true">'
      +   '<i class="fill" data-l="0" data-w="' + pct(v).toFixed(2) + '" style="left:0;width:0%"></i>'
      + '</span>'
      + '<span class="val">' + api.num(v, 1) + '</span>'
      + '<span class="mk">' + (isWin ? DIA + '<b>best</b>' : '') + '</span>'
      + '</li>';
  }

  function gapBar(r, i) {
    var m = META[i];
    var d = r.vals[0] - r.vals[i];
    var up = d >= 0;
    var w = dwid(d);
    var left = up ? ZERO : ZERO - w;
    var sizeTxt = m.size ? m.size : 'size n/a';
    return '<li class="br gp' + (up ? '' : ' bad') + '" data-ix="' + i + '">'
      + '<span class="sz">' + esc(sizeTxt) + '</span>'
      + '<span class="mn" title="' + esc(m.full) + ' scored ' + api.num(r.vals[i], 1) + '">'
      +   esc(m.base) + '</span>'
      + '<span class="vh">against ' + esc(m.base) + ' at ' + esc(sizeTxt) + ', which scored '
      +   api.num(r.vals[i], 1) + ', ' + esc(US.base) + ' is '
      +   api.num(Math.abs(d), 1) + ' points ' + (up ? 'ahead' : 'behind') + '. </span>'
      + '<span class="track" style="background-image:' + BG_D + '" aria-hidden="true">'
      +   '<i class="fill" data-l="' + left.toFixed(2) + '" data-w="' + w.toFixed(2) + '"'
      +     ' style="left:' + ZERO.toFixed(2) + '%;width:0%"></i>'
      + '</span>'
      + '<span class="val">' + signed(d) + '</span>'
      + '<span class="mk">' + (up ? UP + '<b>ahead</b>' : DOWN + '<b>behind</b>') + '</span>'
      + '</li>';
  }

  function block(r) {
    var rival = META[r.rival] || { base: '', size: null, full: '' };
    var d = Math.abs(r.delta);
    var tag = r.usBest
      ? '<span class="tag win">' + DIA + (r.tie ? 'tied best' : 'best of ' + r.vals.length)
          + ', ' + api.num(d, 1) + ' clear of ' + esc(rival.base) + '</span>'
      : '<span class="tag lose">' + DOWN + 'loses by ' + api.num(d, 1) + ' to '
          + esc(rival.full) + ', ranked ' + r.rank + ' of ' + r.vals.length + '</span>';

    var meta = [];
    if (r.inv) {
      meta.push(api.num(r.inv.n, 0) + ' items');
      if (r.inv.kind) meta.push(esc(r.inv.kind));
    }
    if (view === 1) meta.push('reference ' + esc(US.base) + ' ' + api.num(r.vals[0], 1));

    var list = (view === 0)
      ? r.vals.map(function (_, i) { return scoreBar(r, i); }).join('')
      : r.vals.map(function (_, i) { return i === 0 ? '' : gapBar(r, i); }).join('');

    return '<div class="bk ' + (r.usBest ? 'w' : 'l') + '" role="group" aria-label="'
      + esc(r.name) + '">'
      + '<div class="bkhd">'
      +   '<span class="bn">' + esc(r.name)
      +     (meta.length
              ? '<i title="split size and task type as listed in the paper’s evaluation'
                + ' inventory">' + meta.join(' &middot; ') + '</i>'
              : '') + '</span>'
      +   tag
      + '</div>'
      + '<ul class="bars">' + list + '</ul>'
      + (r.note ? '<p class="note">' + INFO + '<span>' + esc(r.note) + '</span></p>' : '')
      + '</div>';
  }

  /* ---------------- roster ---------------- */

  function roster() {
    return META.map(function (m, i) {
      return '<button type="button" class="mchip' + (i === 0 ? ' us' : '') + '"'
        + ' data-ix="' + i + '" aria-pressed="' + (i === pinned) + '">'
        + '<span class="csz">' + esc(m.size || '?') + '</span>'
        + '<span class="cnm">' + esc(m.base) + '</span>'
        + '<span class="cmn">' + api.num(mean(i), 1)
        + '<i> mean F1 across the ' + ROWS.length + ' splits</i></span>'
        + '</button>';
    }).join('');
  }

  /* ---------------- paint ---------------- */

  function paint(animate) {
    wrap.classList.toggle('gapview', view === 1);
    paintScale();
    gridEl.innerHTML = ROWS.map(block).join('');
    rosterEl.innerHTML = roster();
    bind();
    grow(animate);
  }

  /* bars are written at zero, the layout is forced, then the real widths land,
     which gives the transition something to run from without depending on a
     frame callback that a hidden or backgrounded tab would never fire */
  function grow(animate) {
    var still = api.reduce || !animate;
    var fills = gridEl.querySelectorAll('.fill'), i;
    if (still) gridEl.classList.add('nofx');
    void gridEl.offsetHeight;
    for (i = 0; i < fills.length; i++) {
      fills[i].style.left = fills[i].getAttribute('data-l') + '%';
      fills[i].style.width = fills[i].getAttribute('data-w') + '%';
    }
    if (still) {
      void gridEl.offsetHeight;
      gridEl.classList.remove('nofx');
    }
  }

  /* ---------------- isolate one model ---------------- */

  function iso(ix) {
    var rows = gridEl.querySelectorAll('.br'), i, k;
    for (i = 0; i < rows.length; i++) {
      k = +rows[i].getAttribute('data-ix');
      rows[i].classList.toggle('dim', ix >= 0 && k !== ix);
      rows[i].classList.toggle('lit', ix >= 0 && k === ix);
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

  VIEWS.forEach(function (v, i) {
    root.querySelector('#S_MM-view-' + i).addEventListener('click', function () {
      if (view === i) return;
      view = i;
      VIEWS.forEach(function (_, j) {
        var b = root.querySelector('#S_MM-view-' + j);
        b.classList.toggle('sel', j === i);
        b.setAttribute('aria-pressed', j === i ? 'true' : 'false');
      });
      paint(true);
    });
  });

  paint(true);

  return null;
};
