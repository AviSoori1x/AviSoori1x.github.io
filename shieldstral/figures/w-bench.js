(function () {
  var ID = 'w-bench';
  var root = document.getElementById(ID);
  if (!root) return;

  var SS = (typeof window !== 'undefined' && window.SS) ? window.SS : null;
  if (!SS) return;

  var B = SS.benchmarks || {};
  var H = SS.headline || {};
  var ROSTER = Array.isArray(SS.baselines) ? SS.baselines : [];
  var INV = Array.isArray(SS.benchInventory) ? SS.benchInventory : [];
  var KEYS = Object.keys(B);
  if (!KEYS.length) return;

  var NS = 'http://www.w3.org/2000/svg';

  /* ---------- tiny helpers ---------- */

  function pick(sfx) { return document.getElementById(ID + '-' + sfx); }

  function mk(tag, cls, txt) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt !== undefined && txt !== null) n.textContent = String(txt);
    return n;
  }

  function sv(tag, attrs) {
    var n = document.createElementNS(NS, tag);
    if (attrs) {
      for (var k in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, k) && attrs[k] !== null) {
          n.setAttribute(k, String(attrs[k]));
        }
      }
    }
    return n;
  }

  function clear(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function f1(v) { return (Math.round(v * 10) / 10).toFixed(1); }

  function thou(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }

  function norm(s) { return String(s).toLowerCase().replace(/[^a-z0-9]/g, ''); }

  /* "Nemotron-3.5-Content-Safety-4B" -> base + size, so the size can survive truncation */
  function splitSize(name) {
    var m = /^(.*?)[-\s](\d+(?:\.\d+)?[BbMm])$/.exec(String(name));
    if (m) return { base: m[1], size: m[2].toUpperCase() };
    return { base: String(name), size: '' };
  }

  function commonPrefix(a, b) {
    var i = 0, n = Math.min(a.length, b.length);
    while (i < n && a.charAt(i) === b.charAt(i)) i++;
    return i;
  }

  /* match a chart model name against the roster in SS.baselines by name prefix */
  function rosterFor(name) {
    var nb = norm(splitSize(name).base);
    var best = null, score = -1;
    ROSTER.forEach(function (r) {
      var nr = norm(r.model);
      var L = commonPrefix(nb, nr);
      var s = (nb === nr) ? 1000 : L;
      if (L >= 6 && s > score) { score = s; best = r; }
    });
    return best;
  }

  function sizeOf(name) {
    var r = rosterFor(name);
    if (r && r.size) return String(r.size);
    return splitSize(name).size;
  }

  function policyOf(name) {
    var r = rosterFor(name);
    if (!r) return null;
    return r.adaptive ? 'adaptive' : 'fixed';
  }

  function isOurs(name) { return /shieldstral/i.test(String(name)); }

  function pretty(name) {
    var s = splitSize(name);
    var sz = sizeOf(name);
    return sz ? (s.base + ' ' + sz) : s.base;
  }

  /* key -> human label, derived from the key itself */
  function keyLabel(k) {
    var s = String(k).replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/\s*Card$/, '');
    return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
  }

  /* ---------- analysis, all derived from SS ---------- */

  function analyze(key) {
    var t = B[key] || {};
    var models = Array.isArray(t.models) ? t.models : [];
    var rows = Array.isArray(t.rows) ? t.rows : [];

    var si = 0;
    for (var i = 0; i < models.length; i++) { if (isOurs(models[i])) { si = i; break; } }
    var order = [si];
    models.forEach(function (_, j) { if (j !== si) order.push(j); });

    var an = rows.map(function (r) {
      var vals = Array.isArray(r.vals) ? r.vals : [];
      var max = null;
      vals.forEach(function (v) {
        if (v === null || v === undefined) return;
        if (max === null || v > max) max = v;
      });
      var ours = (vals[si] === null || vals[si] === undefined) ? null : vals[si];
      var best = (ours !== null && max !== null && Math.abs(ours - max) < 1e-9);
      return {
        name: r.name,
        vals: vals,
        max: max,
        ours: ours,
        isBest: best,
        gap: (ours !== null && max !== null) ? (max - ours) : null
      };
    });

    return { key: key, models: models, order: order, si: si, rows: an };
  }

  var ANALYSIS = {};
  KEYS.forEach(function (k) { ANALYSIS[k] = analyze(k); });

  var TOTAL = (function () {
    var n = 0, best = 0, close = 0;
    KEYS.forEach(function (k) {
      ANALYSIS[k].rows.forEach(function (r) {
        n++;
        if (r.isBest) best++;
        if (r.gap !== null && r.gap <= 1.0 + 1e-9) close++;
      });
    });
    return { rows: n, best: best, close: close };
  })();

  /* mean per model in a table, used only to name the runner up */
  function meanOf(key, idx) {
    var s = 0, n = 0;
    (ANALYSIS[key].rows || []).forEach(function (r) {
      var v = r.vals[idx];
      if (v !== null && v !== undefined) { s += v; n++; }
    });
    return n ? s / n : null;
  }

  /* the 20B baseline from the roster, named without hardcoding it */
  function big20(needReason) {
    var hit = null;
    ROSTER.forEach(function (r) {
      if (String(r.size) !== '20B') return;
      if (needReason && !/reason/i.test(String(r.output || ''))) return;
      if (!needReason && !r.adaptive) return;
      if (!hit) hit = r;
    });
    return hit ? (r0(hit)) : null;
  }
  function r0(r) { return r.model + ' ' + r.size; }

  /* whichever non Shieldstral model in the multimodal table matches multimodalNextBest */
  function multimodalRunnerUp() {
    if (!ANALYSIS.multimodal || H.multimodalNextBest === undefined) return null;
    var A = ANALYSIS.multimodal, hit = null;
    A.models.forEach(function (m, i) {
      if (i === A.si) return;
      var mu = meanOf('multimodal', i);
      if (mu === null) return;
      if (Math.abs(Math.round(mu * 10) / 10 - H.multimodalNextBest) < 0.051 && !hit) hit = pretty(m);
    });
    return hit;
  }

  /* ---------- headline strip ---------- */

  function verb(v, cmp) {
    if (v === undefined || cmp === undefined) return { word: '', state: 'tie', tag: '' };
    if (v > cmp + 1e-9) return { word: 'Ahead of', state: 'lead', tag: 'lead' };
    if (v < cmp - 1e-9) return { word: 'Behind', state: 'behind', tag: 'behind' };
    return { word: 'Level with', state: 'tie', tag: 'level' };
  }

  function statCell(value, kicker, cmp, who) {
    if (value === undefined) return null;
    var st = verb(value, cmp);
    var cell = mk('div', 'wb-stat');
    var n = mk('div', 'wb-n' + (st.state === 'lead' ? ' is-lead' : ''));
    n.appendChild(mk('span', null, f1(value)));
    n.appendChild(mk('span', 'wb-tag' + (st.state === 'lead' ? ' is-lead' : (st.state === 'behind' ? ' is-behind' : '')), st.tag));
    cell.appendChild(n);
    cell.appendChild(mk('div', 'wb-k', kicker));
    if (cmp !== undefined) {
      cell.appendChild(mk('div', 'wb-s', st.word + ' ' + (who ? who : 'the next model') + ' at ' + f1(cmp)));
    }
    return cell;
  }

  function buildStats() {
    var host = pick('stats');
    if (!host) return;
    clear(host);
    var cells = [
      statCell(H.textF1, 'Overall text F1', H.textTiedWith, big20(false)),
      statCell(H.multimodalF1, 'Multimodal F1', H.multimodalNextBest, multimodalRunnerUp()),
      statCell(H.adaptabilityF1, 'Policy adaptability F1', H.adaptabilityBest, big20(true))
    ];
    cells.forEach(function (c) { if (c) host.appendChild(c); });
  }

  /* ---------- tabs ---------- */

  var state = { tab: KEYS[0], lossOnly: false };
  var tabBtns = [];

  function buildTabs() {
    var host = pick('tabs');
    if (!host) return;
    clear(host);
    tabBtns = [];
    KEYS.forEach(function (k, i) {
      var b = mk('button', 'wb-tab');
      b.setAttribute('type', 'button');
      b.setAttribute('role', 'tab');
      b.setAttribute('id', ID + '-tab-' + i);
      b.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
      b.setAttribute('tabindex', i === 0 ? '0' : '-1');
      b.appendChild(mk('span', null, keyLabel(k)));
      b.appendChild(mk('span', 'wb-tabn', ANALYSIS[k].rows.length));
      b.addEventListener('click', function () { select(i, false); });
      b.addEventListener('keydown', function (ev) {
        var d = 0;
        if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') d = 1;
        else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') d = -1;
        else if (ev.key === 'Home') { if (ev.preventDefault) ev.preventDefault(); select(0, true); return; }
        else if (ev.key === 'End') { if (ev.preventDefault) ev.preventDefault(); select(KEYS.length - 1, true); return; }
        if (!d) return;
        if (ev.preventDefault) ev.preventDefault();
        select((i + d + KEYS.length) % KEYS.length, true);
      });
      tabBtns.push(b);
      host.appendChild(b);
    });
  }

  function select(i, moveFocus) {
    state.tab = KEYS[i];
    tabBtns.forEach(function (b, j) {
      b.setAttribute('aria-selected', j === i ? 'true' : 'false');
      b.setAttribute('tabindex', j === i ? '0' : '-1');
    });
    if (moveFocus && tabBtns[i] && tabBtns[i].focus) tabBtns[i].focus();
    var panel = pick('panel');
    if (panel) panel.setAttribute('aria-labelledby', ID + '-tab-' + i);
    render();
  }

  /* ---------- legend ---------- */

  function buildLegend() {
    var host = pick('legend');
    if (!host) return;
    clear(host);
    var A = ANALYSIS[state.tab];
    A.order.forEach(function (mi, slot) {
      var name = A.models[mi];
      var ours = isOurs(name);
      var item = mk('div', 'wb-li' + (ours ? ' is-ours' : ''));
      item.appendChild(mk('span', 'wb-sw wb-s' + Math.min(slot, 5)));
      item.appendChild(mk('span', null, splitSize(name).base));
      var sz = sizeOf(name);
      if (sz) item.appendChild(mk('span', 'wb-lsz', sz));
      var pol = policyOf(name);
      item.appendChild(mk(
        'span',
        'wb-pol' + (pol === 'adaptive' ? ' is-ad' : ''),
        pol === null ? 'not in roster' : (pol === 'adaptive' ? 'adaptive' : 'fixed taxonomy')
      ));
      host.appendChild(item);
    });
  }

  /* ---------- chart ---------- */

  function trunc(s, n) {
    if (n < 3) return '';
    return s.length <= n ? s : (s.slice(0, n - 1) + '…');
  }

  function visibleRows() {
    var A = ANALYSIS[state.tab];
    return A.rows.filter(function (r) { return !state.lossOnly || !r.isBest; });
  }

  function renderChart() {
    var host = pick('chart');
    var empty = pick('empty');
    if (!host) return;
    clear(host);

    var A = ANALYSIS[state.tab];
    var rows = visibleRows();

    if (empty) {
      if (!rows.length) {
        empty.hidden = false;
        empty.textContent = 'Shieldstral is best in every row of this table.';
        return;
      }
      empty.hidden = true;
      empty.textContent = '';
    }
    if (!rows.length) return;

    var W = Math.round(host.clientWidth || 0) || 640;
    if (W < 240) W = 240;

    var narrow = W < 470;
    var fsLab = narrow ? 8.5 : 9.5;
    var fsVal = narrow ? 8.5 : 9.5;
    var barH = 11;
    var barGap = 3;
    var step = barH + barGap;
    var headH = narrow ? 17 : 18;
    var rowGap = 11;
    var valW = narrow ? 28 : 32;
    var bestW = 38;

    var labelW = narrow
      ? Math.min(116, Math.max(62, Math.round(W * 0.33)))
      : Math.min(184, Math.max(120, Math.round(W * 0.28)));
    var trackX = labelW + 8;
    var trackW = W - trackX - 4 - valW - 6 - bestW;
    if (trackW < 66) {
      labelW = Math.max(56, labelW - (66 - trackW));
      trackX = labelW + 8;
      trackW = W - trackX - 4 - valW - 6 - bestW;
    }
    if (trackW < 24) trackW = 24;

    var valEnd = trackX + trackW + 4 + valW;
    var bestX = valEnd + 6;
    var maxChars = Math.floor(labelW / (fsLab * 0.605));

    function xv(v) { return trackX + (Math.max(0, Math.min(100, v)) / 100) * trackW; }

    var svg = sv('svg', { xmlns: NS });
    var gGrid = sv('g', null);
    svg.appendChild(gGrid);

    var y = 5;
    var top = y;

    rows.forEach(function (r) {
      var g = sv('g', null);

      var hb = y + headH - 6;
      var tn = sv('text', { x: 0, y: hb, class: 'wb-rowname' });
      tn.textContent = r.name;
      g.appendChild(tn);

      var note = r.isBest ? 'best' : (r.gap === null ? 'no score' : ('gap ' + f1(r.gap)));
      var tg = sv('text', { x: W, y: hb, class: 'wb-rowgap', 'text-anchor': 'end' });
      tg.textContent = note;
      g.appendChild(tg);

      g.appendChild(sv('line', { x1: 0, y1: y + headH - 2.5, x2: W, y2: y + headH - 2.5, class: 'wb-underline' }));
      y += headH;

      A.order.forEach(function (mi, slot) {
        var name = A.models[mi];
        var v = r.vals[mi];
        var has = (v !== null && v !== undefined);
        var ours = isOurs(name);
        var best = has && r.max !== null && Math.abs(v - r.max) < 1e-9;
        var by = y;
        var mid = by + barH / 2 + fsLab * 0.36;

        var bar = sv('g', null);
        var ttl = sv('title', null);
        ttl.textContent = pretty(name) + ' on ' + r.name + ': F1 ' + (has ? f1(v) : 'not reported') + (best ? ' (best in row)' : '');
        bar.appendChild(ttl);

        var lab = sv('text', {
          x: 0, y: mid,
          class: 'wb-mlabel' + (ours ? ' is-ours' : ''),
          'font-size': fsLab
        });
        var sz = sizeOf(name);
        var szTxt = sz ? (' ' + sz) : '';
        var baseTxt = trunc(splitSize(name).base, Math.max(3, maxChars - szTxt.length));
        var t1 = sv('tspan', null);
        t1.textContent = baseTxt;
        lab.appendChild(t1);
        if (szTxt) {
          var t2 = sv('tspan', { class: 'wb-msz' });
          t2.textContent = szTxt;
          lab.appendChild(t2);
        }
        bar.appendChild(lab);

        bar.appendChild(sv('rect', { x: trackX, y: by, width: trackW, height: barH, class: 'wb-track' }));

        if (has) {
          bar.appendChild(sv('rect', {
            x: trackX, y: by,
            width: Math.max(1, xv(v) - trackX),
            height: barH,
            class: 'b' + Math.min(slot, 5)
          }));
        }

        var val = sv('text', {
          x: valEnd, y: mid,
          class: 'wb-val' + (ours ? ' is-ours' : '') + (has ? '' : ' is-na'),
          'text-anchor': 'end',
          'font-size': fsVal
        });
        val.textContent = has ? f1(v) : 'n/a';
        bar.appendChild(val);

        if (best) {
          var my = by + barH / 2;
          bar.appendChild(sv('path', {
            d: 'M' + bestX + ',' + (my - 3.6) + ' L' + (bestX + 5.6) + ',' + my + ' L' + bestX + ',' + (my + 3.6) + ' Z',
            class: 'wb-mark'
          }));
          var bt = sv('text', { x: bestX + 8.5, y: mid, class: 'wb-best' });
          bt.textContent = 'best';
          bar.appendChild(bt);
        }

        g.appendChild(bar);
        y += step;
      });

      y += rowGap;
      svg.appendChild(g);
    });

    var bottom = y - rowGap + 3;

    [0, 25, 50, 75, 100].forEach(function (t) {
      var x = xv(t);
      gGrid.appendChild(sv('line', {
        x1: x, y1: top, x2: x, y2: bottom,
        class: (t === 0 || t === 100) ? 'wb-axis' : 'wb-grid'
      }));
    });
    gGrid.appendChild(sv('line', { x1: trackX, y1: bottom, x2: trackX + trackW, y2: bottom, class: 'wb-axis' }));

    var tickY = bottom + 11;
    [0, 50, 100].forEach(function (t) {
      var tx = sv('text', {
        x: xv(t), y: tickY, class: 'wb-tick',
        'text-anchor': t === 0 ? 'start' : (t === 100 ? 'end' : 'middle')
      });
      tx.textContent = String(t);
      gGrid.appendChild(tx);
    });
    var unit = sv('text', { x: 0, y: tickY, class: 'wb-unit' });
    unit.textContent = 'F1';
    gGrid.appendChild(unit);

    var Htotal = tickY + 5;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + Htotal);
    svg.setAttribute('width', W);
    svg.setAttribute('height', Htotal);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'F1 by benchmark and model for ' + keyLabel(state.tab) + '. The same numbers are in the table that follows.');
    host.appendChild(svg);
  }

  /* ---------- tally, screen reader mirror ---------- */

  function renderTally() {
    var host = pick('tally');
    if (!host) return;
    clear(host);
    var A = ANALYSIS[state.tab];
    var best = 0;
    A.rows.forEach(function (r) { if (r.isBest) best++; });

    function chunk(pre, a, b, post) {
      host.appendChild(mk('span', null, pre));
      host.appendChild(mk('b', null, a + ' of ' + b));
      if (post) host.appendChild(mk('span', null, post));
    }
    chunk('Shieldstral is best in ', best, A.rows.length, ' rows here, ');
    chunk('', TOTAL.best, TOTAL.rows, ' across all tables. ');
    chunk('Within 1.0 F1 of the best in ', TOTAL.close, TOTAL.rows, '.');
  }

  function renderSR() {
    var host = pick('sr');
    if (!host) return;
    clear(host);
    var A = ANALYSIS[state.tab];
    var rows = visibleRows();
    var t = mk('table');
    var cap = mk('caption', null, keyLabel(state.tab) + ', F1 per model');
    t.appendChild(cap);
    var thead = mk('thead');
    var htr = mk('tr');
    htr.appendChild(mk('th', null, 'Benchmark'));
    A.order.forEach(function (mi) { htr.appendChild(mk('th', null, pretty(A.models[mi]))); });
    thead.appendChild(htr);
    t.appendChild(thead);
    var tb = mk('tbody');
    rows.forEach(function (r) {
      var tr = mk('tr');
      tr.appendChild(mk('th', null, r.name));
      A.order.forEach(function (mi) {
        var v = r.vals[mi];
        var has = (v !== null && v !== undefined);
        var best = has && r.max !== null && Math.abs(v - r.max) < 1e-9;
        tr.appendChild(mk('td', null, (has ? f1(v) : 'not reported') + (best ? ' best' : '')));
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    host.appendChild(t);
  }

  /* ---------- reference tables ---------- */

  function buildInventory() {
    var t = pick('inv');
    var sum = pick('sum-inv');
    if (!t) return;
    clear(t);
    var total = 0;
    INV.forEach(function (r) { total += Number(r[3]) || 0; });
    if (sum) {
      clear(sum);
      sum.appendChild(mk('span', null, 'Benchmark inventory'));
      sum.appendChild(mk('span', 'wb-tabn', INV.length + ' sets, ' + thou(total) + ' samples'));
    }
    var thead = mk('thead');
    var htr = mk('tr');
    ['Dataset', 'Split coverage', 'Languages', 'Samples'].forEach(function (h, i) {
      htr.appendChild(mk('th', i === 3 ? 'wb-num' : null, h));
    });
    thead.appendChild(htr);
    t.appendChild(thead);
    var tb = mk('tbody');
    INV.forEach(function (r) {
      var tr = mk('tr');
      tr.appendChild(mk('td', null, r[0]));
      tr.appendChild(mk('td', null, r[1]));
      tr.appendChild(mk('td', null, r[2]));
      tr.appendChild(mk('td', 'wb-num', thou(r[3])));
      tb.appendChild(tr);
    });
    t.appendChild(tb);
  }

  function buildRoster() {
    var t = pick('ros');
    var sum = pick('sum-ros');
    if (!t) return;
    clear(t);
    var list = ROSTER.slice().sort(function (a, b) {
      return (isOurs(b.model) ? 1 : 0) - (isOurs(a.model) ? 1 : 0);
    });
    var ad = 0;
    ROSTER.forEach(function (r) { if (r.adaptive) ad++; });
    if (sum) {
      clear(sum);
      sum.appendChild(mk('span', null, 'Model roster'));
      sum.appendChild(mk('span', 'wb-tabn', ROSTER.length + ' models, ' + ad + ' policy adaptive'));
    }
    var thead = mk('thead');
    var htr = mk('tr');
    ['Model', 'Size', 'Input', 'Output', 'Policy'].forEach(function (h) { htr.appendChild(mk('th', null, h)); });
    thead.appendChild(htr);
    t.appendChild(thead);
    var tb = mk('tbody');
    list.forEach(function (r) {
      var tr = mk('tr', isOurs(r.model) ? 'is-ours' : null);
      tr.appendChild(mk('td', null, r.model));
      tr.appendChild(mk('td', null, r.size));
      tr.appendChild(mk('td', null, r.input));
      tr.appendChild(mk('td', null, r.output));
      var td = mk('td');
      td.appendChild(mk('span', 'wb-flag' + (r.adaptive ? ' is-ad' : ''), r.adaptive ? 'adaptive' : 'fixed taxonomy'));
      tr.appendChild(td);
      tb.appendChild(tr);
    });
    t.appendChild(tb);
  }

  /* ---------- wiring ---------- */

  function render() {
    buildLegend();
    renderChart();
    renderTally();
    renderSR();
  }

  var only = pick('only');
  if (only) {
    only.addEventListener('click', function () {
      state.lossOnly = !state.lossOnly;
      only.setAttribute('aria-pressed', state.lossOnly ? 'true' : 'false');
      render();
    });
  }

  buildStats();
  buildTabs();
  buildInventory();
  buildRoster();
  select(0, false);

  var pending = null;
  function onResize() {
    if (pending) return;
    pending = true;
    var run = function () { pending = null; renderChart(); };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run);
    else run();
  }

  if (typeof ResizeObserver === 'function') {
    var ro = new ResizeObserver(onResize);
    var chartHost = pick('chart');
    if (chartHost) ro.observe(chartHost);
  } else if (typeof window !== 'undefined' && window.addEventListener) {
    window.addEventListener('resize', onResize);
  }
})();
