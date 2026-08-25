(function(){
  var ID = 'w-ablation';
  var root = document.getElementById(ID);
  if (!root) return;

  var SS = window.SS;
  if (!SS || !SS.stageAblation || !SS.merge) return;

  var SA = SS.stageAblation;
  var MG = SS.merge;
  var cols = SA.cols || [];
  var saRows = SA.rows || [];
  var mgRows = MG.rows || [];
  if (!cols.length || !saRows.length || !mgRows.length) return;

  var seg = document.getElementById(ID + '-seg');
  var chart = document.getElementById(ID + '-chart');
  var scale = document.getElementById(ID + '-scale');
  var callout = document.getElementById(ID + '-callout');
  var line = document.getElementById(ID + '-line');
  if (!seg || !chart || !scale || !callout || !line) return;

  /* the caveat block is created if the host markup predates it, so a stale page
     degrades to a chart without caveats instead of blanking the whole figure */
  var foot = document.getElementById(ID + '-foot');
  if (!foot && line.parentNode){
    foot = document.createElement('p');
    foot.className = 'wa-foot';
    foot.id = ID + '-foot';
    line.parentNode.appendChild(foot);
  }

  var MERGE_KEY = '0.6PG+0.3P+0.1I';
  var PUBLIC_KEY = 'P';
  /* label map only, no data: spells out the abbreviated column headers for prose */
  var LONG = { 'Acc.':'accuracy', 'Prec.':'precision', 'Rec.':'recall', 'F1':'F1' };
  var DOMAIN = 100;
  var CARET_UP = 'M4 1.1 L7.3 6.6 L0.7 6.6 Z';
  var CARET_DOWN = 'M4 6.9 L0.7 1.4 L7.3 1.4 Z';

  function esc(s){
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }
  function fmt(n){ return Number(n).toFixed(1); }
  function pad(n){ return (n < 10 ? '0' : '') + n; }
  function pct(v){ return Math.round(Math.max(0, Math.min(DOMAIN, Number(v))) / DOMAIN * 1e6) / 1e4; }
  function r4(n){ return Math.round(n * 1e4) / 1e4; }
  function cap(s){ return s.charAt(0).toUpperCase() + s.slice(1); }
  function longName(c){ return LONG[c] || c; }
  function joinList(a){
    if (a.length < 2) return a.join('');
    return a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1];
  }

  /* the final row of the ablation is the merged checkpoint, read from SS.merge */
  var mergeRow = null, publicRow = null;
  for (var m = 0; m < mgRows.length; m++){
    if (mgRows[m].name === MERGE_KEY) mergeRow = mgRows[m];
    if (mgRows[m].name === PUBLIC_KEY) publicRow = mgRows[m];
  }
  if (!mergeRow) mergeRow = mgRows[mgRows.length - 1];
  var mgCols = MG.cols || cols;

  var stages = [];
  saRows.forEach(function(r){
    stages.push({
      name: r.name,
      grp: 'single checkpoints, no merging',
      mix: '',
      get: function(mi){ return r.vals[mi]; }
    });
  });
  stages.push({
    name: 'Merge of the two checkpoints with the base instruct model',
    grp: 'after SLERP weight merging',
    mix: mergeRow.name,
    sub: 'not another round of training. The weights of the earlier checkpoints are interpolated with the base instruct model at the ratio in the chip.',
    get: function(mi){
      var k = mgCols.indexOf(cols[mi]);
      if (k < 0) k = mi;
      return mergeRow.taxonomy[k];
    }
  });
  /* the second stage is the generalisation result, worth one line of context */
  if (stages.length > 1) stages[1].sub = 'the evaluation taxonomy was designed separately, and no leaf maps one to one';

  /* ---------- metric buttons ---------- */
  var sel = cols.indexOf('F1');
  if (sel < 0) sel = cols.length - 1;

  var segHtml = '';
  cols.forEach(function(c, i){
    segHtml += '<button type="button" class="wa-opt" role="radio" id="' + ID + '-opt-' + i +
      '" aria-checked="false" tabindex="-1" aria-label="' + esc(longName(c)) + '">' + esc(c) + '</button>';
  });
  seg.innerHTML = segHtml;
  var opts = [];
  cols.forEach(function(c, i){ opts.push(document.getElementById(ID + '-opt-' + i)); });

  /* ---------- rows ---------- */
  var rowHtml = '';
  var lastGrp = null;
  stages.forEach(function(s, i){
    if (s.grp !== lastGrp){
      rowHtml += '<p class="wa-grp' + (lastGrp === null ? '' : ' is-later') + '">' + esc(s.grp) + '</p>';
      lastGrp = s.grp;
    }
    rowHtml += '<div class="wa-row' + (s.mix ? ' is-merge' : '') + '" id="' + ID + '-row-' + i + '">' +
      '<div class="wa-head">' +
        '<span class="wa-idx">' + pad(i + 1) + '</span>' +
        '<span class="wa-name">' + esc(s.name) + '</span>' +
        (s.mix ? '<code class="wa-mix">' + esc(s.mix) + '</code>' : '') +
      '</div>' +
      (s.sub ? '<p class="wa-sub">' + esc(s.sub) + '</p>' : '') +
      '<div class="wa-plot">' +
        '<div class="wa-track">' +
          '<span class="wa-bar" id="' + ID + '-bar-' + i + '"></span>' +
          '<span class="wa-gain" id="' + ID + '-gain-' + i + '" hidden></span>' +
          '<span class="wa-loss" id="' + ID + '-loss-' + i + '" hidden></span>' +
          '<span class="wa-carry" id="' + ID + '-carry-' + i + '" hidden></span>' +
          '<span class="wa-zero" id="' + ID + '-zero-' + i + '" hidden></span>' +
          '<span class="wa-zerotxt" id="' + ID + '-zt-' + i + '" hidden>never predicts a violation</span>' +
        '</div>' +
        '<div class="wa-lane">' +
          '<span class="wa-brk" id="' + ID + '-brk-' + i + '" hidden></span>' +
          '<span class="wa-chip" id="' + ID + '-chip-' + i + '">' +
            '<span class="wa-sr" id="' + ID + '-chipsr-' + i + '"></span>' +
            '<svg class="wa-car" viewBox="0 0 8 8" aria-hidden="true" focusable="false">' +
              '<path id="' + ID + '-car-' + i + '" d="' + CARET_UP + '"></path>' +
            '</svg>' +
            '<span id="' + ID + '-chipt-' + i + '"></span>' +
          '</span>' +
        '</div>' +
      '</div>' +
      '<div class="wa-val"><span id="' + ID + '-v-' + i + '"></span>' +
        '<span class="wa-sr" id="' + ID + '-vsr-' + i + '"></span></div>' +
    '</div>';
  });
  chart.innerHTML = rowHtml;

  var el = {};
  ['row','bar','gain','loss','carry','zero','zt','brk','chip','chipsr','car','chipt','v','vsr'].forEach(function(k){
    el[k] = stages.map(function(s, i){ return document.getElementById(ID + '-' + k + '-' + i); });
  });

  /* ---------- axis ---------- */
  var steps = 4;
  var axHtml = '';
  for (var t = 0; t <= steps; t++){
    var p = t / steps * 100;
    var cls = 'wa-tk' + (t === 0 ? ' is-first' : '') + (t === steps ? ' is-last' : '');
    var style = t === 0 ? 'left:0' : (t === steps ? 'right:0' : 'left:' + p + '%;transform:translateX(-50%)');
    axHtml += '<span class="' + cls + '" style="' + style + '">' + Math.round(p / 100 * DOMAIN) + '</span>';
  }
  scale.innerHTML = axHtml;

  /* ---------- static callout, built from the base row ---------- */
  (function(){
    var base = saRows[0].vals || [];
    var live = [], dead = [], deadVal = null;
    cols.forEach(function(c, i){
      if (Number(base[i]) === 0){ dead.push(esc(longName(c))); deadVal = base[i]; }
      else live.push(esc(longName(c)) + ' of <span class="wa-num">' + fmt(base[i]) + '</span>');
    });
    if (!dead.length || !live.length){ callout.hidden = true; return; }
    callout.innerHTML = '<span class="wa-k">Read stage ' + pad(1) + ' carefully.</span> It posts ' +
      joinList(live) + ', yet ' + joinList(dead) + ' are all <span class="wa-num">' + fmt(deadVal) +
      '</span>. The untrained base model never predicts a violation, so the only thing it gets right is content that was safe to begin with. A high score on one metric can mean the classifier is not working at all.';
  })();

  /* ---------- what this chart leaves out, all numbers from SS ---------- */
  (function(){
    if (!foot) return;
    var bits = [];
    var f1 = mgCols.indexOf('F1');
    if (f1 < 0) f1 = mgCols.length - 1;
    if (publicRow && publicRow !== mergeRow && mergeRow.aegis && publicRow.aegis){
      var mA = Number(mergeRow.aegis[f1]), pA = Number(publicRow.aegis[f1]);
      if (isFinite(mA) && isFinite(pA)){
        bits.push('The merge is not free. On the Aegis v2 validation set it scores <span class="wa-num">' +
          fmt(mA) + '</span> ' + esc(mgCols[f1]) + ' against <span class="wa-num">' + fmt(pA) +
          '</span> for the public-data checkpoint, so it ' +
          (mA < pA ? 'gives back <span class="wa-num">' + fmt(pA - mA) + '</span> there'
                   : 'holds its ground there') +
          ' to win the taxonomy set shown above.');
      }
    }
    var hl = SS.headline || {};
    if (typeof hl.adaptabilityF1 === 'number'){
      bits.push('This is a validation set, not the headline benchmark. The <span class="wa-num">' +
        fmt(hl.adaptabilityF1) + '</span> ' + esc(cols[cols.length - 1]) +
        ' quoted for the released model elsewhere in this post is measured on the adaptability benchmark, which is a different set of samples.');
    }
    if (!bits.length){ foot.hidden = true; return; }
    foot.innerHTML = bits.join(' ');
  })();

  /* ---------- render ---------- */
  function render(mi){
    var prev = null;
    var deltas = [];

    stages.forEach(function(s, i){
      var raw = Number(s.get(mi));
      var v = isFinite(raw) ? raw : 0;
      var w = pct(v);
      var isZero = v === 0;

      var lo = w, hi = w, d = null;
      if (prev !== null){
        d = v - prev.v;
        lo = Math.min(prev.w, w);
        hi = Math.max(prev.w, w);
        deltas.push({ i: i, d: d });
      }

      el.bar[i].style.width = r4(prev === null ? w : lo) + '%';

      var up = d !== null && d >= 0;
      el.gain[i].hidden = !(d !== null && d > 0);
      if (!el.gain[i].hidden){
        el.gain[i].style.left = r4(lo) + '%';
        el.gain[i].style.width = r4(hi - lo) + '%';
      }
      el.loss[i].hidden = !(d !== null && d < 0);
      if (!el.loss[i].hidden){
        el.loss[i].style.left = r4(lo) + '%';
        el.loss[i].style.width = r4(hi - lo) + '%';
      }
      el.carry[i].hidden = !(d !== null && d < 0);
      if (!el.carry[i].hidden) el.carry[i].style.left = r4(hi) + '%';

      el.zero[i].hidden = !isZero;
      el.zt[i].hidden = !(isZero && i === 0);
      el.row[i].className = 'wa-row' + (s.mix ? ' is-merge' : '') + (isZero ? ' is-zero' : '');

      el.v[i].textContent = fmt(v);
      el.vsr[i].textContent = ' ' + longName(cols[mi]);

      var chip = el.chip[i];
      if (d === null){
        chip.className = 'wa-chip is-base';
        el.chipsr[i].textContent = '';
        el.chipt[i].textContent = 'baseline';
        el.brk[i].hidden = true;
        chip.style.left = '0%';
        chip.style.transform = 'none';
      } else {
        chip.className = 'wa-chip ' + (up ? 'is-up' : 'is-down');
        el.car[i].setAttribute('d', up ? CARET_UP : CARET_DOWN);
        el.chipsr[i].textContent = 'change ';
        el.chipt[i].textContent = (up ? '+' : '-') + fmt(Math.abs(d));
        el.brk[i].hidden = false;
        el.brk[i].style.left = r4(lo) + '%';
        el.brk[i].style.width = r4(hi - lo) + '%';
        var mid = (lo + hi) / 2;
        if (mid <= 14){ chip.style.left = '0%'; chip.style.transform = 'none'; }
        else if (mid >= 86){ chip.style.left = '100%'; chip.style.transform = 'translateX(-100%)'; }
        else { chip.style.left = r4(mid) + '%'; chip.style.transform = 'translateX(-50%)'; }
      }

      prev = { v: v, w: w };
    });

    var big = null, drops = [];
    deltas.forEach(function(x){
      if (!big || Math.abs(x.d) > Math.abs(big.d)) big = x;
      if (x.d < 0) drops.push(x);
    });

    var txt = '';
    if (big){
      txt = 'On ' + longName(cols[mi]) + ' the largest step is ' + (big.d >= 0 ? '+' : '-') + fmt(Math.abs(big.d)) +
        ' into stage ' + pad(big.i + 1) + '. ';
      if (drops.length){
        txt += cap(joinList(drops.map(function(x){
          return 'stage ' + pad(x.i + 1) + ' gives back ' + fmt(Math.abs(x.d));
        }))) + '.';
      } else {
        txt += 'No stage loses ground.';
      }
    }
    line.textContent = txt;
  }

  /* ---------- interaction ---------- */
  function select(i, focus){
    sel = i;
    opts.forEach(function(b, k){
      var on = k === i;
      b.setAttribute('aria-checked', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1;
    });
    if (focus && opts[i]) opts[i].focus();
    render(i);
  }

  opts.forEach(function(b, i){
    if (!b) return;
    b.addEventListener('click', function(){ select(i, false); });
    b.addEventListener('keydown', function(ev){
      var k = ev.key, n = null;
      if (k === 'ArrowRight' || k === 'ArrowDown') n = (i + 1) % opts.length;
      else if (k === 'ArrowLeft' || k === 'ArrowUp') n = (i - 1 + opts.length) % opts.length;
      else if (k === 'Home') n = 0;
      else if (k === 'End') n = opts.length - 1;
      if (n === null) return;
      ev.preventDefault();
      select(n, true);
    });
  });

  select(sel, false);
})();
