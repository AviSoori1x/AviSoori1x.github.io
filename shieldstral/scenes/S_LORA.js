window.SCENES = window.SCENES || {};

/* Act III, scene 23. LoRA against full SFT.
   SS.loraVsSft holds two ablation validation sets, each with the same two runs
   and the same four metrics. The point of the figure is that the two sets
   disagree about the winner and that every gap is tiny, so the picture is a
   pair of dumbbell plots on a shared per-set score axis, plus the gap on the
   metric in focus as a big numeral.
   Every score, every gap, every tally and every run name is read from
   window.SS at runtime. Nothing here is a live model call. */
window.SCENES['S_LORA'] = function (root, api) {
  var SS = api.SS || {};
  var D = SS.loraVsSft || {};
  var COLS = (D.cols || []).slice();

  root.className = 'sc-s_lora';

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* Display names for the two validation sets. The scores, the run names and
     the metric names all come from SS, only these two words are ours, and the
     SS key is printed under each one so the provenance is visible. */
  var LABEL = { aegis: 'Aegis v2', taxonomy: 'Taxonomy' };

  var SETS = [];
  ['aegis', 'taxonomy'].forEach(function (k) {
    var rows = D[k];
    if (rows && rows.length >= 2) {
      SETS.push({ key: k, title: LABEL[k] || k, rows: rows });
    }
  });

  if (!COLS.length || !SETS.length) {
    root.appendChild(api.frag('<div class="wrap"><div class="hd">'
      + '<span class="eyebrow">lora against full sft</span>'
      + '<span class="hnote">SS.loraVsSft is empty, nothing to draw</span>'
      + '</div></div>').firstChild);
    return;
  }

  var NM = COLS.length;

  /* ---------- read the table ---------- */
  function pickLora(rows) {
    for (var i = 0; i < rows.length; i++) {
      if (/lora/i.test(String(rows[i].name || ''))) return i;
    }
    return 0;
  }

  var LORA_NAME = '', SFT_NAME = '';
  SETS.forEach(function (s) {
    var li = pickLora(s.rows);
    var si = li === 0 ? 1 : 0;
    s.lora = s.rows[li];
    s.sft = s.rows[si];
    LORA_NAME = LORA_NAME || String(s.lora.name || 'LoRA');
    SFT_NAME = SFT_NAME || String(s.sft.name || 'Full SFT');

    var all = [], m;
    s.m = [];
    for (m = 0; m < NM; m++) {
      var a = Number((s.lora.vals || [])[m]);
      var b = Number((s.sft.vals || [])[m]);
      var ok = isFinite(a) && isFinite(b);
      if (ok) { all.push(a); all.push(b); }
      s.m.push({
        a: ok ? a : null,
        b: ok ? b : null,
        gap: ok ? Math.abs(a - b) : null,
        loraLeads: ok ? a >= b : null
      });
    }
    /* one score axis per set, padded a point past the extremes so the two
       markers on every row are compared against the same ruler */
    var lo = Math.floor(Math.min.apply(null, all)) - 1;
    var hi = Math.ceil(Math.max.apply(null, all)) + 1;
    s.lo = lo;
    s.hi = hi;
    s.pos = function (v) { return ((v - lo) / (hi - lo)) * 100; };

    /* how many of the metrics each run takes on this set */
    var w = 0;
    for (m = 0; m < NM; m++) if (s.m[m].loraLeads === true) w++;
    s.loraWins = w;
    s.sftWins = NM - w;
  });

  /* the widest disagreement anywhere in the table */
  var maxGap = 0, maxGapWhere = '';
  SETS.forEach(function (s) {
    for (var m = 0; m < NM; m++) {
      if (s.m[m].gap != null && s.m[m].gap > maxGap + 1e-9) {
        maxGap = s.m[m].gap;
        maxGapWhere = String(COLS[m]) + ' on ' + s.title;
      }
    }
  });

  /* ---------- markup ---------- */
  var chips = '', i, m;
  for (m = 0; m < NM; m++) {
    chips += '<button type="button" class="chp" role="radio" aria-checked="false"'
      + ' tabindex="-1" id="S_LORA-chp-' + m + '">' + esc(COLS[m]) + '</button>';
  }

  function panel(s, si) {
    var rows = '', mm, r, p1, p2, l, w;
    for (mm = 0; mm < NM; mm++) {
      r = s.m[mm];
      if (r.a == null) {
        rows += '<div class="row" id="S_LORA-r-' + si + '-' + mm + '">'
          + '<div class="rh"><span class="rlab">' + esc(COLS[mm]) + '</span>'
          + '<span class="rnums"><span class="ng">not published</span></span></div></div>';
        continue;
      }
      p1 = s.pos(r.a);
      p2 = s.pos(r.b);
      l = Math.min(p1, p2);
      w = Math.abs(p1 - p2);
      rows += '<div class="row" id="S_LORA-r-' + si + '-' + mm + '">'
        + '<div class="rh">'
        +   '<span class="rlab">' + esc(COLS[mm]) + '</span>'
        +   '<span class="rnums">'
        +     '<span class="nv"><i class="sw lora" aria-hidden="true"></i><b>'
        +       api.num(r.a, 1) + '</b></span>'
        +     '<span class="nv"><i class="sw sft" aria-hidden="true"></i><b>'
        +       api.num(r.b, 1) + '</b></span>'
        +     '<span class="ng">gap <b>' + api.num(r.gap, 1) + '</b></span>'
        +   '</span>'
        + '</div>'
        + '<div class="rtrk"><div class="rin">'
        +   '<i class="conn" style="left:' + l.toFixed(2) + '%;width:'
        +     w.toFixed(2) + '%"></i>'
        +   '<i class="mk sft" style="left:' + p2.toFixed(2) + '%"></i>'
        +   '<i class="mk lora" style="left:' + p1.toFixed(2) + '%"></i>'
        + '</div></div>'
        + '</div>';
    }

    return '<section class="pnl">'
      + '<div class="ptop">'
      +   '<b class="pname">' + esc(s.title) + '</b>'
      +   '<span class="ptag">ablation validation set</span>'
      + '</div>'
      /* no aria-live here, the figure cycles on its own and would talk over
         the reader. Every score sits in the rows below as plain text. */
      + '<div class="pgap">'
      +   '<b class="gnum" id="S_LORA-gap-' + si + '">0.0</b>'
      +   '<span class="gside">'
      +     '<span class="gpts">points</span>'
      +     '<span class="gwho" id="S_LORA-who-' + si + '">'
      +       '<i class="sw" aria-hidden="true"></i><em></em></span>'
      +     '<span class="gmet" id="S_LORA-met-' + si + '"></span>'
      +   '</span>'
      + '</div>'
      + '<div class="rows">' + rows + '</div>'
      + '<div class="pax" aria-hidden="true">'
      +   '<span>' + s.lo + '</span>'
      +   '<span>' + api.num((s.lo + s.hi) / 2, 0) + '</span>'
      +   '<span>' + s.hi + '</span>'
      + '</div>'
      + '<div class="paxl">SS.loraVsSft.' + esc(s.key)
      +   ', one ruler for all four rows</div>'
      + '</section>';
  }

  var panels = '';
  for (i = 0; i < SETS.length; i++) panels += panel(SETS[i], i);

  /* the verdict sentence, counted out of the same table */
  var vparts = SETS.map(function (s) {
    var lead = s.loraWins > s.sftWins ? LORA_NAME : SFT_NAME;
    var n = Math.max(s.loraWins, s.sftWins);
    return '<b>' + esc(lead) + '</b> takes ' + n + ' of ' + NM + ' on ' + esc(s.title);
  }).join(', ');

  root.appendChild(api.frag(
    '<div class="wrap">'

    + '<div class="hd">'
    +   '<span class="eyebrow">lora against full sft</span>'
    +   '<span class="hnote">two ways to update the weights, on the two sets the choice '
    +   'was made on</span>'
    + '</div>'

    + '<div class="bar">'
    +   '<div class="lgd">'
    +     '<span class="lg"><i class="sw lora" aria-hidden="true"></i>'
    +       esc(LORA_NAME) + '</span>'
    +     '<span class="lg"><i class="sw sft" aria-hidden="true"></i>'
    +       esc(SFT_NAME) + '</span>'
    +   '</div>'
    +   '<div class="mrow">'
    +     '<span class="mlab" id="S_LORA-mlab">metric in focus</span>'
    +     '<div class="chips" role="radiogroup" aria-labelledby="S_LORA-mlab"'
    +       ' id="S_LORA-chips">' + chips + '</div>'
    +   '</div>'
    + '</div>'

    + '<div class="grid">' + panels + '</div>'

    + '<div class="vd">'
    +   '<span class="vk">the call</span>'
    +   '<p class="vt">' + vparts + ', and the widest gap anywhere in the table is '
    +     '<b>' + api.num(maxGap, 1) + '</b> points (' + esc(maxGapWhere) + '). '
    +     'The report treats that as no significant difference and takes <b>'
    +     esc(LORA_NAME) + '</b> for the training efficiency.</p>'
    + '</div>'

    + '<div class="foot">'
    +   '<span class="gt">Both are ablation validation sets, not the headline benchmarks of '
    +   'Act IV. Every score, gap and tally is read from SS.loraVsSft at runtime, no live '
    +   'model call.</span>'
    +   '<span class="hint" id="S_LORA-hint">cycling on its own, click or use arrow keys to '
    +   'take over</span>'
    + '</div>'

    + '</div>'
  ).firstChild);

  /* ---------- handles ---------- */
  var chipEls = [], gapEls = [], whoEls = [], metEls = [], rowEls = [];
  for (m = 0; m < NM; m++) chipEls.push(root.querySelector('#S_LORA-chp-' + m));
  for (i = 0; i < SETS.length; i++) {
    gapEls.push(root.querySelector('#S_LORA-gap-' + i));
    whoEls.push(root.querySelector('#S_LORA-who-' + i));
    metEls.push(root.querySelector('#S_LORA-met-' + i));
    var rr = [];
    for (m = 0; m < NM; m++) rr.push(root.querySelector('#S_LORA-r-' + i + '-' + m));
    rowEls.push(rr);
  }
  var hint = root.querySelector('#S_LORA-hint');
  var chipBox = root.querySelector('#S_LORA-chips');

  var cur = -1;
  var auto = true;

  function setMetric(idx, fromUser) {
    if (idx === cur || idx < 0 || idx >= NM) return;
    var j, k;
    for (j = 0; j < NM; j++) {
      chipEls[j].setAttribute('aria-checked', j === idx ? 'true' : 'false');
      chipEls[j].tabIndex = j === idx ? 0 : -1;
    }
    for (k = 0; k < SETS.length; k++) {
      var s = SETS[k];
      var r = s.m[idx];
      for (j = 0; j < NM; j++) {
        if (rowEls[k][j]) rowEls[k][j].classList.toggle('on', j === idx);
      }
      if (r.a == null) {
        gapEls[k].textContent = 'n/a';
        whoEls[k].querySelector('em').textContent = 'not published';
        whoEls[k].className = 'gwho';
        metEls[k].textContent = 'on ' + String(COLS[idx]);
        continue;
      }
      gapEls[k].textContent = api.num(r.gap, 1);
      gapEls[k].classList.toggle('lo', r.loraLeads === true);
      whoEls[k].className = 'gwho ' + (r.loraLeads ? 'lo' : 'sf');
      whoEls[k].querySelector('em').textContent =
        (r.loraLeads ? LORA_NAME : SFT_NAME) + ' ahead';
      metEls[k].textContent = 'on ' + String(COLS[idx]);
    }
    if (fromUser && auto) {
      auto = false;
      hint.textContent = 'manual, arrow keys move the focus';
    }
    cur = idx;
  }

  chipEls.forEach(function (b, j) {
    b.addEventListener('click', function () { setMetric(j, true); });
  });
  chipBox.addEventListener('keydown', function (e) {
    var key = e.key, nx = -1;
    if (key === 'ArrowLeft' || key === 'ArrowUp') nx = (cur - 1 + NM) % NM;
    else if (key === 'ArrowRight' || key === 'ArrowDown') nx = (cur + 1) % NM;
    else if (key === 'Home') nx = 0;
    else if (key === 'End') nx = NM - 1;
    if (nx < 0) return;
    e.preventDefault();
    setMetric(nx, true);
    chipEls[nx].focus();
  });

  /* open on the summary metric if the table has one */
  var start = 0;
  for (m = 0; m < NM; m++) if (/^f1/i.test(String(COLS[m]))) start = m;
  setMetric(start, false);

  var running = false, nextAt = null;
  return {
    start: function () { running = true; nextAt = null; },
    stop: function () { running = false; },
    tick: function (t) {
      if (!running || !auto || api.reduce) return;
      if (nextAt === null) { nextAt = t + 3.0; return; }
      if (t >= nextAt) {
        nextAt = t + 3.0;
        setMetric((cur + 1) % NM, false);
      }
    }
  };
};
