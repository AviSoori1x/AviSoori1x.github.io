window.SCENES = window.SCENES || {};

/* Act III, scene 25. The SLERP merge.
   The five measured merge recipes in SS.merge.rows as points on two axes,
   Aegis v2 validation F1 against fine-grained taxonomy validation F1. Both
   coordinates, the four-metric readout and the column names are read from
   SS.merge at runtime. The mix bar is parsed out of the recipe name itself,
   so 0.6PG+0.3P+0.1I draws its own proportions.
   No curve is fitted and no weight slider is offered: only these five
   recipes exist, anything between them would be invented. */
window.SCENES['S_MERGE'] = function (root, api) {
  var SS = api.SS || {};
  var el = api.el;
  var num = api.num;

  var M = SS.merge || {};
  var COLS = (M.cols || []).slice();
  var ROWS = (M.rows || []).filter(function (r) {
    return r && r.name && r.aegis && r.taxonomy;
  });

  var wrap = el('div', 'sc-s_merge');
  root.appendChild(wrap);

  if (ROWS.length < 2 || !COLS.length) {
    wrap.appendChild(el('p', 'miss',
      'SS.merge does not carry the measured recipes, so this plot cannot be drawn.'));
    return null;
  }

  /* the F1 column, located by name rather than assumed to be last */
  var f1i = COLS.length - 1;
  COLS.forEach(function (c, i) {
    if (String(c).toLowerCase().replace(/[^a-z0-9]/g, '') === 'f1') f1i = i;
  });

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /* ---------------- parse the recipe name into a weight mix ---------------- */
  var COMP = {
    PG: { cls: 'cpg', full: 'public plus generated taxonomy checkpoint' },
    P: { cls: 'cp', full: 'public safety data checkpoint' },
    I: { cls: 'ci', full: 'Ministral-3B-Instruct, the base model' }
  };

  function parseMix(name) {
    var out = [], at = {};
    String(name).split('+').forEach(function (t) {
      var m = /^\s*([0-9]*\.?[0-9]*)\s*([A-Za-z]+)\s*$/.exec(t);
      if (!m) return;
      var w = (m[1] === '' || m[1] === '.') ? 1 : parseFloat(m[1]);
      if (!isFinite(w) || w <= 0) w = 1;
      var k = m[2].toUpperCase();
      if (at[k] != null) { out[at[k]].w += w; return; }
      at[k] = out.length;
      out.push({ k: k, w: w });
    });
    if (!out.length) out.push({ k: String(name), w: 1 });
    var s = out.reduce(function (a, b) { return a + b.w; }, 0) || 1;
    out.forEach(function (o) { o.p = o.w / s; o.cls = (COMP[o.k] || {}).cls || 'cx'; });
    return out;
  }

  var PT = ROWS.map(function (r, i) {
    return {
      i: i,
      name: String(r.name),
      mix: parseMix(r.name),
      aegis: r.aegis,
      tax: r.taxonomy,
      x: Number(r.aegis[f1i]),
      y: Number(r.taxonomy[f1i])
    };
  }).filter(function (p) { return isFinite(p.x) && isFinite(p.y); });

  /* the shipped model is the one recipe that blends all three ingredients */
  var FIN = 0, best = -1;
  PT.forEach(function (p, i) { if (p.mix.length > best) { best = p.mix.length; FIN = i; } });
  PT.forEach(function (p, i) { p.fin = (i === FIN); });

  var xs = PT.map(function (p) { return p.x; });
  var ys = PT.map(function (p) { return p.y; });
  var xlo = Math.min.apply(null, xs), xhi = Math.max.apply(null, xs);
  var ylo = Math.min.apply(null, ys), yhi = Math.max.apply(null, ys);
  var xspan = xhi - xlo, yspan = yhi - ylo;
  var xpad = Math.max(xspan * 0.18, 0.25), ypad = Math.max(yspan * 0.14, 0.25);
  var X0 = xlo - xpad, X1 = xhi + xpad, Y0 = ylo - ypad, Y1 = yhi + ypad;

  function ticks(a, b, want) {
    var span = b - a;
    if (!(span > 0)) return { v: [a], dp: 1 };
    var raw = span / want;
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var n = raw / mag;
    var step = (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * mag;
    var out = [], t = Math.ceil(a / step) * step, guard = 0;
    while (t <= b + 1e-9 && guard++ < 40) { out.push(Math.round(t * 1e6) / 1e6); t += step; }
    /* one decimal count for the whole axis, so 87 and 87.5 do not sit side by side */
    return { v: out, dp: step >= 1 ? 0 : (step >= 0.1 ? 1 : 2) };
  }

  var XT = ticks(X0, X1, 4), YT = ticks(Y0, Y1, 4);

  /* ---------------- header ---------------- */
  var hd = el('div', 'hd');
  hd.appendChild(el('span', 'eyy', 'slerp merge · ' + PT.length + ' measured recipes'));
  var autoBtn = el('button', 'aut');
  autoBtn.type = 'button';
  autoBtn.appendChild(el('span', 'dot'));
  autoBtn.appendChild(el('span', 'atx', 'auto on'));
  autoBtn.setAttribute('aria-pressed', 'true');
  autoBtn.setAttribute('aria-label', 'Step through the five recipes automatically');
  hd.appendChild(autoBtn);
  wrap.appendChild(hd);

  /* ---------------- chart ---------------- */
  var chart = el('div', 'chart');
  wrap.appendChild(chart);

  var zoom = el('p', 'zoom');
  zoom.appendChild(el('b', null, 'The two axes are not on the same scale. '));
  zoom.appendChild(document.createTextNode(
    'All ' + PT.length + ' recipes sit within ' + num(xspan, 1)
    + ' F1 points of each other on Aegis v2, while taxonomy F1 spans '
    + num(yspan, 1) + '. The horizontal axis is zoomed hard, read the tick values.'));
  wrap.appendChild(zoom);

  var G_WIDE = { W: 860, H: 372, ml: 80, mr: 24, mt: 30, mb: 64, fT: 12, fL: 13, fA: 12.5, fN: 11, r: 7.5, rf: 12.5, yt: ['fine-grained taxonomy', 'validation F1'] };
  var G_NAR = { W: 372, H: 372, ml: 46, mr: 12, mt: 26, mb: 58, fT: 9.5, fL: 10, fA: 9.5, fN: 8.5, r: 6, rf: 9.5, yt: ['taxonomy val F1'] };
  var narrow = matchMedia('(max-width: 46rem)');
  var sel = 0;

  function drawChart() {
    var G = narrow.matches ? G_NAR : G_WIDE;
    var pw = G.W - G.ml - G.mr, ph = G.H - G.mt - G.mb;
    var px = function (v) { return G.ml + (v - X0) / (X1 - X0) * pw; };
    var py = function (v) { return G.mt + (1 - (v - Y0) / (Y1 - Y0)) * ph; };
    var s = '';

    /* grid and ticks */
    XT.v.forEach(function (t) {
      var x = px(t).toFixed(1);
      s += '<line class="gl" x1="' + x + '" y1="' + G.mt + '" x2="' + x + '" y2="' + (G.mt + ph) + '"></line>';
      s += '<text class="tk" x="' + x + '" y="' + (G.mt + ph + G.fT + 8)
        + '" text-anchor="middle" font-size="' + G.fT + '">' + t.toFixed(XT.dp) + '</text>';
    });
    YT.v.forEach(function (t) {
      var y = py(t).toFixed(1);
      s += '<line class="gl" x1="' + G.ml + '" y1="' + y + '" x2="' + (G.ml + pw) + '" y2="' + y + '"></line>';
      s += '<text class="tk" x="' + (G.ml - 9) + '" y="' + y
        + '" text-anchor="end" dy=".33em" font-size="' + G.fT + '">' + t.toFixed(YT.dp) + '</text>';
    });

    /* axes */
    s += '<line class="ax" x1="' + G.ml + '" y1="' + (G.mt + ph) + '" x2="' + (G.ml + pw) + '" y2="' + (G.mt + ph) + '"></line>';
    s += '<line class="ax" x1="' + G.ml + '" y1="' + G.mt + '" x2="' + G.ml + '" y2="' + (G.mt + ph) + '"></line>';

    /* what the reader should know before reading a single point */
    s += '<text class="note" x="' + (G.ml + 9) + '" y="' + (G.mt + G.fN + 3)
      + '" font-size="' + G.fN + '">' + PT.length + ' measured points, no curve fitted</text>';
    s += '<text class="note" x="' + (G.ml + pw - 6) + '" y="' + (G.mt + G.fN + 3)
      + '" text-anchor="end" font-size="' + G.fN + '">up and to the right is better</text>';

    /* selection crosshair, drawn under the points */
    var cur = PT[sel];
    if (cur) {
      var cx = px(cur.x), cy = py(cur.y);
      s += '<g class="sel">';
      s += '<line class="ch" x1="' + cx.toFixed(1) + '" y1="' + cy.toFixed(1) + '" x2="' + cx.toFixed(1) + '" y2="' + (G.mt + ph) + '"></line>';
      s += '<line class="ch" x1="' + G.ml + '" y1="' + cy.toFixed(1) + '" x2="' + cx.toFixed(1) + '" y2="' + cy.toFixed(1) + '"></line>';
      var cw = G.fT * 3.1, chh = G.fT + 8;
      var bx = Math.min(Math.max(cx, G.ml + cw / 2), G.ml + pw - cw / 2);
      s += '<rect class="chip" x="' + (bx - cw / 2).toFixed(1) + '" y="' + (G.mt + ph + 3)
        + '" width="' + cw.toFixed(1) + '" height="' + chh + '" rx="3"></rect>';
      s += '<text class="chv" x="' + bx.toFixed(1) + '" y="' + (G.mt + ph + 3 + chh / 2)
        + '" text-anchor="middle" dy=".34em" font-size="' + G.fT + '">' + num(cur.x, 1) + '</text>';
      var by = Math.min(Math.max(cy, G.mt + chh / 2), G.mt + ph - chh / 2);
      s += '<rect class="chip" x="' + (G.ml - cw - 5).toFixed(1) + '" y="' + (by - chh / 2).toFixed(1)
        + '" width="' + cw.toFixed(1) + '" height="' + chh + '" rx="3"></rect>';
      s += '<text class="chv" x="' + (G.ml - 5 - cw / 2).toFixed(1) + '" y="' + by.toFixed(1)
        + '" text-anchor="middle" dy=".34em" font-size="' + G.fT + '">' + num(cur.y, 1) + '</text>';
      s += '</g>';
    }

    /* the five points */
    PT.forEach(function (p, i) {
      var x = px(p.x), y = py(p.y);
      var r = p.fin ? G.rf : G.r;
      var left = (p.x - X0) / (X1 - X0) > 0.55;
      var lx = left ? x - r - 11 : x + r + 11;
      var anc = left ? 'end' : 'start';
      s += '<g class="pt' + (p.fin ? ' fin' : '') + (i === sel ? ' on' : '') + '">';
      if (p.fin) s += '<circle class="halo" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + (r + 7) + '"></circle>';
      if (i === sel) s += '<circle class="ring" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + (r + 6) + '"></circle>';
      s += '<circle class="dot" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + r + '"></circle>';
      if (p.fin) {
        s += '<text class="tag" x="' + lx.toFixed(1) + '" y="' + (y - G.fL + 1).toFixed(1)
          + '" text-anchor="' + anc + '" font-size="' + (G.fN - 0.5) + '">FINAL MODEL</text>';
        s += '<text class="nm" x="' + lx.toFixed(1) + '" y="' + (y + G.fL * 0.62).toFixed(1)
          + '" text-anchor="' + anc + '" font-size="' + G.fL + '">' + esc(p.name) + '</text>';
      } else {
        s += '<text class="nm" x="' + lx.toFixed(1) + '" y="' + y.toFixed(1)
          + '" text-anchor="' + anc + '" dy=".34em" font-size="' + G.fL + '">' + esc(p.name) + '</text>';
      }
      s += '</g>';
    });

    /* hit targets last so they sit on top */
    PT.forEach(function (p, i) {
      s += '<circle class="hit" data-i="' + i + '" cx="' + px(p.x).toFixed(1)
        + '" cy="' + py(p.y).toFixed(1) + '" r="' + Math.max(18, G.rf + 8) + '"></circle>';
    });

    /* axis titles */
    s += '<text class="axt" x="' + (G.ml + pw / 2) + '" y="' + (G.H - 12)
      + '" text-anchor="middle" font-size="' + G.fA + '">Aegis v2 validation F1</text>';
    G.yt.forEach(function (line, k) {
      s += '<text class="axt" transform="translate(' + (G.fA + 1 + k * (G.fA + 2.5)) + ','
        + (G.mt + ph / 2) + ') rotate(-90)" text-anchor="middle" font-size="' + G.fA + '">'
        + esc(line) + '</text>';
    });

    var desc = PT.map(function (p) {
      return p.name + (p.fin ? ', the final model,' : '') + ' at Aegis F1 ' + num(p.x, 1)
        + ' and taxonomy F1 ' + num(p.y, 1);
    }).join('. ');

    chart.innerHTML = '<svg class="plot" viewBox="0 0 ' + G.W + ' ' + G.H + '" role="img" '
      + 'aria-label="Scatter of ' + PT.length + ' measured merge recipes. Horizontal axis '
      + 'Aegis v2 validation F1, vertical axis fine-grained taxonomy validation F1. '
      + esc(desc) + '.">' + s + '</svg>';
  }

  chart.addEventListener('click', function (e) {
    var n = e.target;
    while (n && n !== chart) {
      if (n.getAttribute && n.getAttribute('data-i') != null) {
        pick(Number(n.getAttribute('data-i')), true);
        return;
      }
      n = n.parentNode;
    }
  });

  /* ---------------- selector strip ---------------- */
  function wfmt(p) {
    if (p >= 0.995) return '1.0';
    return (Math.round(p * 100) / 100).toFixed(2).replace(/0$/, '');
  }

  function mixBar(p, cls) {
    var b = el('div', cls);
    if (cls === 'mix') {
      b.setAttribute('role', 'img');
      b.setAttribute('aria-label', 'Merge mix, ' + p.mix.map(function (m) {
        return m.k + ' ' + num(m.p, 2);
      }).join(', '));
    } else {
      b.setAttribute('aria-hidden', 'true');
    }
    p.mix.forEach(function (m) {
      var g = el('span', 'sg ' + m.cls);
      g.style.width = (m.p * 100).toFixed(2) + '%';
      g.title = m.k + ' ' + num(m.p, 2) + ', ' + ((COMP[m.k] || {}).full || 'component');
      if (cls === 'mix') {
        g.appendChild(el('span', 'sgk', m.k));
        g.appendChild(el('span', 'sgw', wfmt(m.p)));
      }
      b.appendChild(g);
    });
    return b;
  }

  var strip = el('div', 'strip');
  strip.setAttribute('role', 'group');
  strip.setAttribute('aria-label', 'Choose one of the ' + PT.length + ' measured recipes');
  var btns = PT.map(function (p, i) {
    var b = el('button', 'rcp' + (p.fin ? ' isfin' : ''));
    b.type = 'button';
    b.setAttribute('aria-pressed', 'false');
    b.setAttribute('aria-label', 'Recipe ' + p.name + (p.fin ? ', the final model' : '')
      + ', Aegis F1 ' + num(p.x, 1) + ', taxonomy F1 ' + num(p.y, 1));
    var top = el('div', 'rtop');
    top.appendChild(el('span', 'rn', p.name));
    if (p.fin) top.appendChild(el('span', 'rfin', 'final'));
    b.appendChild(top);
    b.appendChild(mixBar(p, 'mini'));
    var f = el('div', 'rf');
    f.appendChild(el('span', null, 'x ' + num(p.x, 1)));
    f.appendChild(el('span', null, 'y ' + num(p.y, 1)));
    b.appendChild(f);
    b.addEventListener('click', function () { pick(i, true); });
    strip.appendChild(b);
    return b;
  });
  wrap.appendChild(strip);

  /* ---------------- readout ---------------- */
  var read = el('div', 'read');

  var pickCard = el('div', 'pick');
  var ptop = el('div', 'ptop');
  var pname = el('span', 'pname', '');
  ptop.appendChild(pname);
  var ptag = el('span', 'ptag', '');
  ptop.appendChild(ptag);
  pickCard.appendChild(ptop);
  var mixHost = el('div', 'mixhost');
  pickCard.appendChild(mixHost);
  var mixLeg = el('div', 'mixleg');
  ['PG', 'P', 'I'].forEach(function (k) {
    var lg = el('span', 'lg ' + COMP[k].cls);
    lg.appendChild(el('b', null, k));
    lg.appendChild(document.createTextNode(' ' + COMP[k].full));
    mixLeg.appendChild(lg);
  });
  pickCard.appendChild(mixLeg);
  read.appendChild(pickCard);

  var stats = el('div', 'stats');
  var head = el('div', 'srow shead');
  head.appendChild(el('span', 'slab', ''));
  COLS.forEach(function (c, i) {
    if (i === f1i) return;
    head.appendChild(el('span', 'sc', String(c)));
  });
  head.appendChild(el('span', 'sf1h', String(COLS[f1i])));
  stats.appendChild(head);

  function statRow(axis, title) {
    var r = el('div', 'srow');
    var lab = el('span', 'slab');
    lab.appendChild(el('i', 'axn', axis));
    lab.appendChild(document.createTextNode(title));
    r.appendChild(lab);
    var cells = [];
    COLS.forEach(function (c, i) {
      if (i === f1i) return;
      var n = el('span', 'sc', '');
      n.setAttribute('data-c', String(c));
      r.appendChild(n);
      cells.push({ i: i, node: n });
    });
    var box = el('span', 'sf1');
    var big = el('span', 'big', '');
    box.appendChild(big);
    var dl = el('em', 'dl', '');
    box.appendChild(dl);
    r.appendChild(box);
    stats.appendChild(r);
    return { cells: cells, big: big, dl: dl };
  }

  var rowX = statRow('x axis', 'Aegis v2 validation');
  var rowY = statRow('y axis', 'fine-grained taxonomy validation');
  read.appendChild(stats);
  wrap.appendChild(read);

  /* ---------------- footer ---------------- */
  var foot = el('div', 'foot');
  foot.appendChild(el('b', null, 'Measured recipes only. '));
  foot.appendChild(document.createTextNode(
    'These ' + PT.length + ' merges were run and scored. Nothing was measured between them, so '
    + 'this figure fits no curve and gives you no weight slider: a point you could drag to would '
    + 'be invented, not observed. '));
  var ad = (SS.headline || {}).adaptabilityF1;
  foot.appendChild(el('b', null, 'Different evaluation. '));
  foot.appendChild(document.createTextNode(
    'Both axes are merge-ablation validation sets'
    + (ad == null ? '' : ', not the adaptability benchmark that the ' + num(ad, 1)
      + ' figure elsewhere in this guide comes from') + '.'));
  wrap.appendChild(foot);

  var live = el('span', 'sr');
  live.setAttribute('aria-live', 'polite');
  wrap.appendChild(live);

  /* ---------------- state ---------------- */
  var auto = !api.reduce;
  var lastT = 0;

  function delta(v, ref) {
    var d = v - ref;
    if (Math.abs(d) < 0.05) return { t: 'level with the final model', c: 'z' };
    return { t: (d > 0 ? '+' : '-') + num(Math.abs(d), 1) + ' vs final', c: d > 0 ? 'up' : 'dn' };
  }

  function paint() {
    var p = PT[sel], f = PT[FIN];
    drawChart();

    btns.forEach(function (b, i) {
      b.classList.toggle('on', i === sel);
      b.setAttribute('aria-pressed', i === sel ? 'true' : 'false');
    });

    pname.textContent = p.name;
    ptag.textContent = p.fin ? 'the model that shipped' : 'not shipped';
    ptag.classList.toggle('is', !!p.fin);
    mixHost.textContent = '';
    mixHost.appendChild(mixBar(p, 'mix'));

    [[rowX, p.aegis, f.aegis], [rowY, p.tax, f.tax]].forEach(function (pair) {
      var row = pair[0], vals = pair[1], fv = pair[2];
      row.cells.forEach(function (c) { c.node.textContent = num(vals[c.i], 1); });
      row.big.textContent = num(vals[f1i], 1);
      if (p.fin) {
        row.dl.textContent = 'the final model';
        row.dl.className = 'dl z';
      } else {
        var d = delta(Number(vals[f1i]), Number(fv[f1i]));
        row.dl.textContent = d.t;
        row.dl.className = 'dl ' + d.c;
      }
    });

    live.textContent = p.name + (p.fin ? ', the final model. ' : '. ')
      + 'Aegis v2 validation F1 ' + num(p.aegis[f1i], 1)
      + ', taxonomy validation F1 ' + num(p.tax[f1i], 1) + '.';
  }

  function pick(i, byUser) {
    if (i < 0 || i >= PT.length) return;
    sel = i;
    if (byUser) setAuto(false);
    paint();
  }

  function setAuto(on) {
    auto = on;
    autoBtn.classList.toggle('off', !on);
    autoBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    autoBtn.querySelector('.atx').textContent = on ? 'auto on' : 'auto off';
  }

  autoBtn.addEventListener('click', function () { setAuto(!auto); lastT = 0; });
  strip.addEventListener('keydown', function (e) {
    var d = (e.key === 'ArrowRight' || e.key === 'ArrowDown') ? 1
      : (e.key === 'ArrowLeft' || e.key === 'ArrowUp') ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    var n = (sel + d + PT.length) % PT.length;
    pick(n, true);
    btns[n].focus();
  });
  if (narrow.addEventListener) narrow.addEventListener('change', drawChart);
  else if (narrow.addListener) narrow.addListener(drawChart);

  sel = FIN;
  if (api.reduce) setAuto(false);
  paint();

  return {
    start: function () { drawChart(); },
    stop: function () {},
    tick: function (t) {
      if (!auto) return;
      if (!lastT) { lastT = t; return; }
      if (t - lastT > 3.2) { lastT = t; sel = (sel + 1) % PT.length; paint(); }
    }
  };
};
