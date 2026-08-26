window.SCENES = window.SCENES || {};

/* Act III, scene 24. The two checkpoints as opposing forces.
   SS.merge.rows carries one row per candidate checkpoint, each with a score
   vector on two ablation validation sets. Rows P and PG are the two ends of
   the argument, and the figure is a pair of tug of war ropes on a shared gap
   scale: the knot on each rope is dragged away from even by the difference
   between the two reported F1 scores, toward whichever checkpoint is better
   on that set. The two ropes end up pointing opposite ways, which is the
   whole reason neither checkpoint ships.
   Every score, every gap and the metric name are read from window.SS at
   runtime. Nothing here is a live model call. */
window.SCENES['S_TWOCKPT'] = function (root, api) {
  var SS = api.SS || {};
  var M = SS.merge || {};
  var COLS = (M.cols || []).slice();
  var ROWS = M.rows || [];

  root.classList.add('sc-s_twockpt');

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function findRow(nm) {
    for (var i = 0; i < ROWS.length; i++) {
      if (String(ROWS[i].name == null ? '' : ROWS[i].name).trim() === nm) return ROWS[i];
    }
    return null;
  }

  var P = findRow('P');
  var PG = findRow('PG');
  var f1 = COLS.indexOf('F1');
  if (f1 < 0) f1 = COLS.length - 1;

  function usable(r) {
    return r && r.aegis && r.taxonomy && f1 >= 0
      && isFinite(Number(r.aegis[f1])) && isFinite(Number(r.taxonomy[f1]));
  }

  if (!usable(P) || !usable(PG)) {
    root.appendChild(api.frag('<div class="wrap"><p class="miss">'
      + 'window.SS.merge does not carry both checkpoint rows P and PG with a '
      + 'score on each validation set, so this figure cannot be drawn.'
      + '</p></div>'));
    return null;
  }

  var METRIC = String(COLS[f1] || 'F1').replace(/\.$/, '');

  /* The two ablation validation sets. Only the display wording is ours, the
     keys are the ones the score vectors are filed under in SS.merge.rows. */
  var SETS = [
    { key: 'aegis', name: 'Aegis v2', sub: 'the standard public benchmark' },
    { key: 'taxonomy', name: 'fine grained taxonomy', sub: 'unseen fine grained policies' }
  ];

  var maxAbs = 0;
  SETS.forEach(function (s) {
    s.p = Number(P[s.key][f1]);
    s.g = Number(PG[s.key][f1]);
    s.gap = s.p - s.g;                 /* positive means P is ahead */
    s.mag = Math.abs(s.gap);
    s.win = s.gap >= 0 ? 'P' : 'PG';
    s.wcls = s.gap >= 0 ? 'p' : 'pg';
    s.vecP = P[s.key] || [];
    s.vecG = PG[s.key] || [];
    if (s.mag > maxAbs) maxAbs = s.mag;
  });

  /* one shared gap scale, so a 1.4 rope and a 23.3 rope are directly comparable */
  var STEP = 5;
  var SCALE = Math.max(STEP, Math.ceil((maxAbs * 1.15) / STEP) * STEP);
  var HALF = 44;                       /* percent of the track for a full SCALE of gap */

  function pctFor(gap) { return 50 - (gap / SCALE) * HALF; }

  /* ---------- provenance for the two names, straight out of SS ---------- */
  /* The stage ablation table lists the same two score vectors under readable
     names, so the training mix label under P and PG is data, not a claim. */
  function stageName(vec) {
    var rows = (SS.stageAblation || {}).rows || [];
    for (var i = 0; i < rows.length; i++) {
      var v = rows[i].vals || [];
      if (v.length !== vec.length) continue;
      var same = true;
      for (var j = 0; j < v.length; j++) {
        if (Number(v[j]) !== Number(vec[j])) { same = false; break; }
      }
      if (same) return String(rows[i].name == null ? '' : rows[i].name);
    }
    return '';
  }
  var descP = stageName(SETS[1].vecP);
  var descG = stageName(SETS[1].vecG);

  /* ---------- pieces ---------- */

  function chev(dir) {
    var d = dir === 'left' ? 'M10 3 5 8l5 5' : 'M6 3l5 5-5 5';
    return '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="' + d + '" fill="none"'
      + ' stroke="currentColor" stroke-width="2.4" stroke-linecap="round"'
      + ' stroke-linejoin="round"></path></svg>';
  }

  function ticks() {
    var out = '', v;
    for (v = -SCALE; v <= SCALE + 1e-9; v += STEP) {
      if (Math.abs(v) < 1e-9) continue;
      var pct = 50 + (v / SCALE) * HALF;
      out += '<i class="tk' + (Math.abs(v % 10) < 1e-9 ? ' maj' : '')
        + '" style="left:' + pct.toFixed(3) + '%"></i>';
    }
    return out;
  }

  function axis() {
    var out = '', v;
    for (v = -SCALE; v <= SCALE + 1e-9; v += 10) {
      var pct = 50 + (v / SCALE) * HALF;
      var t = Math.abs(v) < 1e-9 ? 'even' : String(Math.abs(v));
      out += '<i style="left:' + pct.toFixed(3) + '%">' + t + '</i>';
    }
    return out;
  }

  function vecLine(tag, cls, vec) {
    var cells = '';
    for (var i = 0; i < COLS.length; i++) {
      cells += '<span class="c' + (i === f1 ? ' hi' : '') + '"><i>' + esc(COLS[i]) + '</i>'
        + api.num(Number(vec[i]), 1) + '</span>';
    }
    return '<span class="vec"><b class="tag ' + cls + '">' + esc(tag) + '</b>' + cells + '</span>';
  }

  function rope(s, ix) {
    var side = s.win === 'P' ? 'toleft' : 'toright';
    var lab = 'On the ' + s.name + ' ablation validation set, checkpoint P scores '
      + api.num(s.p, 1) + ' ' + METRIC + ' and checkpoint PG scores ' + api.num(s.g, 1)
      + '. ' + s.win + ' is ahead by ' + api.num(s.mag, 1) + ' points.';

    return '<section class="rope">'
      + '<div class="rhd">'
      +   '<span class="eyeb">ablation validation set</span>'
      +   '<b class="rnm">' + esc(s.name) + '</b>'
      +   '<span class="rsub">' + esc(s.sub) + '</span>'
      +   '<span class="rmet">' + esc(METRIC) + ', gap scale ' + SCALE + ' each way</span>'
      + '</div>'
      + '<div class="rbody">'
      +   '<div class="end left">'
      +     '<span class="tag p">P</span>'
      +     '<span class="sc ' + (s.win === 'P' ? 'p win' : 'lose') + '">'
      +       api.num(s.p, 1) + '</span>'
      +     '<span class="cap">' + (s.win === 'P' ? 'ahead' : 'behind') + '</span>'
      +   '</div>'
      +   '<div class="mid">'
      +     '<div class="callzone">'
      +       '<div class="call ' + side + ' ' + s.wcls + '">'
      +         '<b class="gap" id="S_TWOCKPT-gap' + ix + '">' + api.num(0, 1) + '</b>'
      +         '<span class="glab">' + esc(METRIC) + ' pulled to ' + esc(s.win) + '</span>'
      +       '</div>'
      +     '</div>'
      +     '<div class="track" role="img" aria-label="' + esc(lab) + '">'
      +       '<i class="line"></i>' + ticks() + '<i class="ctr"></i>'
      +       '<i class="pull ' + s.wcls + '" id="S_TWOCKPT-pull' + ix + '"></i>'
      +       '<span class="knot ' + s.wcls + '" id="S_TWOCKPT-knot' + ix + '">'
      +         chev(s.win === 'P' ? 'left' : 'right') + '</span>'
      +     '</div>'
      +     '<div class="axis" aria-hidden="true">' + axis() + '</div>'
      +   '</div>'
      +   '<div class="end right">'
      +     '<span class="tag pg">PG</span>'
      +     '<span class="sc ' + (s.win === 'PG' ? 'pg win' : 'lose') + '">'
      +       api.num(s.g, 1) + '</span>'
      +     '<span class="cap">' + (s.win === 'PG' ? 'ahead' : 'behind') + '</span>'
      +   '</div>'
      + '</div>'
      + '<div class="vecs">' + vecLine('P', 'p', s.vecP) + vecLine('PG', 'pg', s.vecG) + '</div>'
      + '</section>';
  }

  function verdict(tag, cls) {
    var out = '<p class="vline"><b class="tag ' + cls + '">' + esc(tag) + '</b>';
    SETS.forEach(function (s, i) {
      var won = s.win === tag;
      var d = (won ? '+' : '-') + api.num(s.mag, 1);
      out += '<span class="cl">' + (i ? 'and ' : '') + (won ? 'wins' : 'loses') + ' '
        + esc(s.name) + ' by <b class="' + (won ? 'w' : 'l') + '">' + d + '</b></span>';
    });
    return out + '</p>';
  }

  /* ---------- shell ---------- */

  root.appendChild(api.frag(
    '<div class="wrap">'

    + '<div class="hd">'
    +   '<span class="eyebrow">the two candidate checkpoints, before any merge</span>'
    +   '<button type="button" class="rep" id="S_TWOCKPT-replay" aria-label="replay the pull">'
    +     '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M13 8a5 5 0 1 1-1.6-3.6"'
    +     ' fill="none" stroke="currentColor" stroke-width="1.6"></path>'
    +     '<path d="M13.4 1.6V5H10" fill="none" stroke="currentColor" stroke-width="1.6"></path>'
    +     '</svg><span>pull again</span>'
    +   '</button>'
    + '</div>'

    + '<div class="idrow">'
    +   '<span class="id p"><b class="tag p">P</b>'
    +     (descP ? '<i>' + esc(descP) + '</i>' : '<i>public safety data only</i>') + '</span>'
    +   '<span class="vs">pulling against</span>'
    +   '<span class="id pg"><b class="tag pg">PG</b>'
    +     (descG ? '<i>' + esc(descG) + '</i>' : '<i>plus generated taxonomy data</i>') + '</span>'
    + '</div>'

    + SETS.map(rope).join('')

    + '<div class="verdict">'
    +   verdict('P', 'p')
    +   verdict('PG', 'pg')
    +   '<p class="vsum">The two knots are dragged opposite ways on the same scale. Each '
    +   'checkpoint is the better one on exactly one of the two ablation validation sets, so '
    +   'neither is the one to ship.</p>'
    + '</div>'

    + '<p class="foot">Scores are the paper\'s reported ablation validation numbers, read from '
    + 'window.SS.merge rows P and PG. Each knot sits at the difference between two published '
    + 'numbers, not at the output of a live model call.</p>'

    + '</div>'
  ));

  /* ---------- the pull ---------- */

  var UI = SETS.map(function (s, ix) {
    return {
      s: s,
      knot: root.querySelector('#S_TWOCKPT-knot' + ix),
      pull: root.querySelector('#S_TWOCKPT-pull' + ix),
      gap: root.querySelector('#S_TWOCKPT-gap' + ix)
    };
  });

  function place(u, e) {
    var pct = 50 + (pctFor(u.s.gap) - 50) * e;
    u.knot.style.left = pct.toFixed(3) + '%';
    if (pct <= 50) {
      u.pull.style.left = pct.toFixed(3) + '%';
      u.pull.style.width = (50 - pct).toFixed(3) + '%';
    } else {
      u.pull.style.left = '50%';
      u.pull.style.width = (pct - 50).toFixed(3) + '%';
    }
    u.gap.textContent = api.num(u.s.mag * e, 1);
  }

  var DUR = 1.25, LAG = 0.22, t0 = null, playing = false, done = false;

  function render(p) {
    UI.forEach(function (u, ix) {
      var t = (p * (DUR + LAG * (UI.length - 1)) - LAG * ix) / DUR;
      t = t < 0 ? 0 : (t > 1 ? 1 : t);
      place(u, 1 - Math.pow(1 - t, 3));
    });
    done = (p >= 1);
  }

  render(api.reduce ? 1 : 0);

  var rep = root.querySelector('#S_TWOCKPT-replay');
  if (api.reduce) rep.style.display = 'none';
  rep.addEventListener('click', function () { t0 = null; playing = true; render(0); });

  return {
    start: function () { if (!done) { t0 = null; playing = true; } },
    stop: function () { playing = false; },
    tick: function (sec) {
      if (!playing) return;
      if (t0 == null) t0 = sec;
      var p = (sec - t0) / (DUR + LAG * (UI.length - 1));
      if (p >= 1) { p = 1; playing = false; }
      render(p);
    }
  };
};
