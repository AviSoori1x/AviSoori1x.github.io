/* w-merge: the five measured SLERP recipes, plotted on the two validation sets. */
(function () {
  var root = document.getElementById('w-merge');
  if (!root || !window.SS || !SS.merge) return;

  var COLS = SS.merge.cols;                 // ["Acc.","Prec.","Rec.","F1"]
  var F1 = COLS.indexOf('F1');
  var ROWS = SS.merge.rows;
  var FINAL = '0.6PG+0.3P+0.1I';

  var COMPONENT = {
    PG: { fill: 'var(--accent)', label: 'PG, public plus generated' },
    P:  { fill: 'var(--muted)',  label: 'P, public only' },
    I:  { fill: 'var(--rule)',   label: 'I, Ministral-3B-Instruct' }
  };

  /* "0.6PG+0.3P+0.1I" -> [{k:'PG',w:0.6}, ...] */
  function mixOf(name) {
    return name.split('+').map(function (term) {
      var m = /^([0-9.]*)(PG|P|I)$/.exec(term.trim());
      if (!m) return null;
      return { k: m[2], w: m[1] === '' ? 1 : parseFloat(m[1]) };
    }).filter(Boolean);
  }

  /* where each point's text label sits, so the two near-coincident points stay legible */
  var PLACE = {
    'P':               { dx: -11, dy: 4,   anchor: 'end' },
    '0.9P+0.1I':       { dx: -11, dy: 4,   anchor: 'end' },
    'PG':              { dx: -11, dy: 4,   anchor: 'end' },
    '0.9PG+0.1I':      { dx: 11,  dy: 4,   anchor: 'start' },
    '0.6PG+0.3P+0.1I': { dx: -11, dy: -9,  anchor: 'end' }
  };

  var selected = ROWS.reduce(function (acc, r, i) { return r.name === FINAL ? i : acc; }, 0);

  var picker = document.getElementById('w-merge-picker');
  var countEl = document.getElementById('w-merge-count');
  var plot = document.getElementById('w-merge-plot');
  var nameEl = document.getElementById('w-merge-name');
  var mixEl = document.getElementById('w-merge-mix');
  var setsEl = document.getElementById('w-merge-sets');

  /* ---------------- picker ---------------- */
  var chips = ROWS.map(function (r, i) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'wm-chip';
    b.setAttribute('role', 'radio');
    b.id = 'w-merge-chip-' + i;
    b.innerHTML = '<span>' + r.name + '</span>' +
      (r.name === FINAL ? '<span class="wm-chiptag">shipped</span>' : '');
    b.addEventListener('click', function () { select(i); });
    b.addEventListener('keydown', function (e) {
      var d = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1
            : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0;
      if (!d) return;
      e.preventDefault();
      var n = (i + d + ROWS.length) % ROWS.length;
      select(n);
      chips[n].focus();
    });
    picker.appendChild(b);
    return b;
  });
  countEl.textContent = ROWS.length + ' measured recipes, no interpolation';

  /* ---------------- scatter geometry ---------------- */
  var W = 620, H = 350;
  var M = { t: 26, r: 22, b: 46, l: 54 };
  var xs = ROWS.map(function (r) { return r.aegis[F1]; });
  var ys = ROWS.map(function (r) { return r.taxonomy[F1]; });

  function nice(lo, hi, padLo, padHi) {
    return { lo: Math.floor((lo - padLo)), hi: Math.ceil((hi + padHi)) };
  }
  var xr = nice(Math.min.apply(null, xs), Math.max.apply(null, xs), 0.8, 0.8);
  var yr = nice(Math.min.apply(null, ys), Math.max.apply(null, ys), 4, 4);

  function px(v) { return M.l + (v - xr.lo) / (xr.hi - xr.lo) * (W - M.l - M.r); }
  function py(v) { return H - M.b - (v - yr.lo) / (yr.hi - yr.lo) * (H - M.t - M.b); }

  function ticks(lo, hi, want) {
    var span = hi - lo;
    var raw = span / want;
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var step = [1, 2, 2.5, 5, 10].reduce(function (best, m) {
      return Math.abs(m * mag - raw) < Math.abs(best * mag - raw) ? m : best;
    }, 1) * mag;
    var out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      out.push(Math.round(v * 100) / 100);
    }
    return out;
  }

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function drawPlot() {
    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' +
      'Scatter of the five measured merge recipes. Horizontal axis is Aegis v2 validation F1, ' +
      'vertical axis is fine-grained taxonomy validation F1. Both are ablation validation ' +
      'sets, not the headline benchmarks.">';

    s += '<rect class="wm-field" x="' + M.l + '" y="' + M.t + '" width="' + (W - M.l - M.r) +
         '" height="' + (H - M.t - M.b) + '"/>';

    ticks(yr.lo, yr.hi, 5).forEach(function (v) {
      var y = py(v);
      s += '<line class="wm-grid" x1="' + M.l + '" y1="' + y + '" x2="' + (W - M.r) + '" y2="' + y + '"/>';
      s += '<text class="wm-tick" x="' + (M.l - 9) + '" y="' + (y + 3.5) +
           '" font-size="10" text-anchor="end">' + v + '</text>';
    });
    ticks(xr.lo, xr.hi, 5).forEach(function (v) {
      var x = px(v);
      s += '<line class="wm-grid" x1="' + x + '" y1="' + M.t + '" x2="' + x + '" y2="' + (H - M.b) + '"/>';
      s += '<text class="wm-tick" x="' + x + '" y="' + (H - M.b + 15) +
           '" font-size="10" text-anchor="middle">' + v + '</text>';
    });

    s += '<line class="wm-spine" x1="' + M.l + '" y1="' + (H - M.b) + '" x2="' + (W - M.r) +
         '" y2="' + (H - M.b) + '"/>';
    s += '<line class="wm-spine" x1="' + M.l + '" y1="' + M.t + '" x2="' + M.l + '" y2="' + (H - M.b) + '"/>';

    // both axes are zoomed well away from zero, so mark the break explicitly
    s += '<line class="wm-break" x1="' + (M.l - 5) + '" y1="' + (H - M.b + 5) + '" x2="' +
         (M.l + 3) + '" y2="' + (H - M.b - 5) + '"/>';
    s += '<line class="wm-break" x1="' + (M.l - 1) + '" y1="' + (H - M.b + 5) + '" x2="' +
         (M.l + 7) + '" y2="' + (H - M.b - 5) + '"/>';

    s += '<text class="wm-axis" x="' + ((M.l + W - M.r) / 2) + '" y="' + (H - 9) +
         '" font-size="9.5" text-anchor="middle">Aegis v2 validation F1, axis does not start at zero</text>';
    s += '<text class="wm-axis" x="' + (-(M.t + H - M.b) / 2) + '" y="14" font-size="9.5" ' +
         'text-anchor="middle" transform="rotate(-90)">Taxonomy validation F1, ablation set</text>';

    // "better this way" cue toward the top right
    var hx = W - M.r - 96, hy = M.t + 16;
    s += '<path class="wm-hintarrow" d="M' + hx + ' ' + (hy + 12) + ' L' + (hx + 52) + ' ' + hy + '"/>';
    s += '<path class="wm-hinthead" d="M' + (hx + 52) + ' ' + hy + ' l-8 1.2 l1.6 5.4 z"/>';
    s += '<text class="wm-hint" x="' + (hx + 56) + '" y="' + (hy + 12) + '" font-size="9">better</text>';

    var sel = ROWS[selected];
    var sx = px(sel.aegis[F1]), sy = py(sel.taxonomy[F1]);
    s += '<line class="wm-cross" x1="' + M.l + '" y1="' + sy + '" x2="' + sx + '" y2="' + sy + '"/>';
    s += '<line class="wm-cross" x1="' + sx + '" y1="' + sy + '" x2="' + sx + '" y2="' + (H - M.b) + '"/>';

    ROWS.forEach(function (r, i) {
      var x = px(r.aegis[F1]), y = py(r.taxonomy[F1]);
      var pl = PLACE[r.name] || { dx: 11, dy: 4, anchor: 'start' };
      var isSel = i === selected;
      var isFinal = r.name === FINAL;

      s += '<g class="wm-pt" tabindex="0" role="button" ' +
           'aria-label="' + esc(r.name) + ', Aegis v2 F1 ' + r.aegis[F1] +
           ', taxonomy F1 ' + r.taxonomy[F1] + '" data-i="' + i + '">';
      if (isSel) s += '<circle class="wm-ring" cx="' + x + '" cy="' + y + '" r="10"/>';
      s += '<circle class="wm-dot' + (isFinal ? ' is-final' : '') + '" cx="' + x + '" cy="' + y + '" r="5"/>';
      s += '<text class="wm-plab' + (isSel ? ' is-sel' : '') + '" x="' + (x + pl.dx) + '" y="' +
           (y + pl.dy) + '" font-size="10.5" text-anchor="' + pl.anchor + '">' + esc(r.name) + '</text>';
      if (isFinal) {
        s += '<text class="wm-ptag" x="' + (x + pl.dx) + '" y="' + (y + pl.dy - 11) +
             '" font-size="7.5" text-anchor="' + pl.anchor + '">shipped</text>';
      }
      s += '<circle class="wm-hit" cx="' + x + '" cy="' + y + '" r="15"/></g>';
    });

    s += '</svg>';
    plot.innerHTML = s;

    Array.prototype.forEach.call(plot.querySelectorAll('.wm-pt'), function (g) {
      var i = parseInt(g.getAttribute('data-i'), 10);
      g.addEventListener('click', function () { select(i); });
      g.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(i); }
      });
    });
  }

  /* ---------------- readout ---------------- */
  function drawReadout() {
    var r = ROWS[selected];

    nameEl.innerHTML = '<span>' + esc(r.name) + '</span>' +
      (r.name === FINAL ? '<span class="wm-nametag">final model</span>' : '');

    var mix = mixOf(r.name);
    var bar = mix.map(function (c) {
      return '<span class="wm-seg" style="width:' + (c.w * 100) + '%;background:' +
             COMPONENT[c.k].fill + '"></span>';
    }).join('');
    var keys = mix.map(function (c) {
      return '<span class="wm-key"><i class="wm-sw" style="background:' + COMPONENT[c.k].fill +
             '"></i>' + c.w.toFixed(1) + ' ' + esc(COMPONENT[c.k].label) + '</span>';
    }).join('');
    mixEl.innerHTML = '<div class="wm-mixbar">' + bar + '</div><div class="wm-mixkeys">' + keys + '</div>';

    var sets = [
      { lab: 'Aegis v2 validation set', vals: r.aegis },
      { lab: 'Taxonomy validation, ablation set', vals: r.taxonomy }
    ];
    setsEl.innerHTML = sets.map(function (set) {
      var rows = COLS.map(function (c, ci) {
        var v = set.vals[ci];
        return '<div class="wm-m' + (ci === F1 ? ' wm-m-f1' : '') + '">' +
          '<span class="wm-mk">' + esc(c) + '</span>' +
          '<span class="wm-track"><span class="wm-fill" style="width:' + v + '%"></span></span>' +
          '<span class="wm-mv">' + v.toFixed(1) + '</span></div>';
      }).join('');
      return '<div><div class="wm-setlab">' + esc(set.lab) + '</div>' + rows + '</div>';
    }).join('');
  }

  /* ---------------- LoRA vs full SFT ---------------- */
  function drawLora() {
    var host = document.getElementById('w-merge-lora');
    if (!host || !SS.loraVsSft) return;
    var cols = SS.loraVsSft.cols;
    var sets = [
      { lab: 'Aegis v2 validation', rows: SS.loraVsSft.aegis },
      { lab: 'Taxonomy validation, ablation set', rows: SS.loraVsSft.taxonomy }
    ];
    host.innerHTML = sets.map(function (set) {
      var head = '<thead><tr><th>' + esc(set.lab) + '</th>' +
        cols.map(function (c) { return '<th>' + esc(c) + '</th>'; }).join('') + '</tr></thead>';
      var body = '<tbody>' + set.rows.map(function (r) {
        return '<tr><th>' + esc(r.name) + '</th>' +
          r.vals.map(function (v) { return '<td>' + v.toFixed(1) + '</td>'; }).join('') + '</tr>';
      }).join('') + '</tbody>';
      return '<table>' + head + body + '</table>';
    }).join('');

    // state the gap rather than asserting it, so the number cannot drift from the data
    var a = SS.loraVsSft.aegis, t = SS.loraVsSft.taxonomy;
    var gapA = (a[1].vals[F1] - a[0].vals[F1]);
    var gapT = (t[0].vals[F1] - t[1].vals[F1]);
    document.getElementById('w-merge-loranote').textContent =
      'Full SFT is ' + gapA.toFixed(1) + ' F1 ahead on Aegis v2 validation, LoRA is ' +
      gapT.toFixed(1) + ' ahead on the taxonomy validation set. The report treats that as no ' +
      'significant difference and adopts LoRA for training efficiency.';
  }

  function select(i) {
    selected = i;
    chips.forEach(function (c, ci) { c.setAttribute('aria-checked', ci === i ? 'true' : 'false'); });
    chips.forEach(function (c, ci) { c.tabIndex = ci === i ? 0 : -1; });
    drawPlot();
    drawReadout();
  }

  select(selected);
  drawLora();
})();
