(function () {
  var ID = 'w-divergence';
  var root = document.getElementById(ID);
  if (!root) return;

  var SS = (typeof window !== 'undefined' && window.SS) ? window.SS : null;
  if (!SS) return;

  var H = SS.headline || {};
  var CMP = Array.isArray(SS.taxCompare) ? SS.taxCompare : [];
  var DIV = Array.isArray(SS.divergence) ? SS.divergence : [];
  var TAX = Array.isArray(SS.evalTaxonomy) ? SS.evalTaxonomy : [];

  var NS = 'http://www.w3.org/2000/svg';

  function pick(id) { return document.getElementById(ID + '-' + id); }

  function el(tag, cls, txt) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt !== undefined && txt !== null) n.textContent = String(txt);
    return n;
  }

  function sv(tag, attrs) {
    var n = document.createElementNS(NS, tag);
    if (attrs) {
      for (var k in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, k)) n.setAttribute(k, String(attrs[k]));
      }
    }
    return n;
  }

  function clear(n) { while (n && n.firstChild) n.removeChild(n.firstChild); }

  /* ------------------------------------------------------------------
     Part 1: comb diagrams of each taxonomy's shape
     ------------------------------------------------------------------ */

  /* One leaf category = one unit of width. Both combs are drawn at the same
     scale, so a longer run really does mean more leaf categories. */
  var CW = 360, CH = 28, CPAD = 4;
  var TICK_Y = 3, TICK_H = 14, SUB_Y = 20, SC_Y = 24, TICK_FRAC = 0.62;
  var SC_GAP_U = 0.5;

  function combSvg(label) {
    var s = sv('svg', {
      viewBox: '0 0 ' + CW + ' ' + CH,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      focusable: 'false'
    });
    s.setAttribute('aria-label', label);
    return s;
  }

  /* Width in leaf-units of the grouped evaluation run. */
  function taxUnits(tax) {
    var u = 0;
    tax.forEach(function (sc, i) {
      if (i) u += SC_GAP_U;
      (Array.isArray(sc.subs) ? sc.subs : []).forEach(function (sub) {
        u += (Array.isArray(sub.leaves) ? sub.leaves : []).length;
      });
    });
    return u;
  }

  /* Flat run: used where the paper does not publish the internal grouping. */
  function flatComb(n, scale, label) {
    var s = combSvg(label);
    if (!n || n < 1) return s;
    var w = scale * TICK_FRAC;
    for (var i = 0; i < n; i++) {
      s.appendChild(sv('rect', {
        x: (CPAD + i * scale).toFixed(2),
        y: TICK_Y, width: w.toFixed(2), height: TICK_H, 'class': 'wd-tick-t'
      }));
    }
    s.appendChild(sv('line', {
      x1: CPAD, y1: SC_Y, x2: (CPAD + (n - 1) * scale + w).toFixed(2), y2: SC_Y,
      'class': 'wd-base-dash'
    }));
    return s;
  }

  /* Two-tier run built from the real evaluation taxonomy tree: a bracket per
     subcategory on the upper rule, a bracket per super class on the lower. */
  function groupedComb(tax, scale, label) {
    var s = combSvg(label);
    var w = scale * TICK_FRAC;
    var cursor = 0, ticks = [], subBars = [], scBars = [];
    tax.forEach(function (sc, i) {
      if (i) cursor += SC_GAP_U;
      var scStart = cursor;
      (Array.isArray(sc.subs) ? sc.subs : []).forEach(function (sub) {
        var leaves = Array.isArray(sub.leaves) ? sub.leaves : [];
        if (!leaves.length) return;
        var subStart = cursor;
        leaves.forEach(function () { ticks.push(cursor); cursor += 1; });
        subBars.push([subStart, cursor - 1]);
      });
      if (cursor > scStart) scBars.push([scStart, cursor - 1]);
    });
    if (!ticks.length) return s;

    ticks.forEach(function (u) {
      s.appendChild(sv('rect', {
        x: (CPAD + u * scale).toFixed(2),
        y: TICK_Y, width: w.toFixed(2), height: TICK_H, 'class': 'wd-tick-e'
      }));
    });
    subBars.forEach(function (g) {
      s.appendChild(sv('line', {
        x1: (CPAD + g[0] * scale).toFixed(2), y1: SUB_Y,
        x2: (CPAD + g[1] * scale + w).toFixed(2), y2: SUB_Y, 'class': 'wd-base'
      }));
    });
    scBars.forEach(function (g) {
      s.appendChild(sv('line', {
        x1: (CPAD + g[0] * scale).toFixed(2), y1: SC_Y,
        x2: (CPAD + g[1] * scale + w).toFixed(2), y2: SC_Y, 'class': 'wd-base-sc'
      }));
    });
    return s;
  }

  function combBlock(swatchCls, name, count, svgNode, note) {
    var wrap = el('div', 'wd-comb');
    var lab = el('div', 'wd-comblab');
    var sw = el('i', 'wd-sw ' + swatchCls);
    sw.setAttribute('aria-hidden', 'true');
    lab.appendChild(sw);
    lab.appendChild(el('span', 'wd-combname', name));
    if (count) lab.appendChild(el('span', 'wd-combcount', count));
    wrap.appendChild(lab);
    wrap.appendChild(svgNode);
    if (note) wrap.appendChild(el('p', 'wd-combnote', note));
    return wrap;
  }

  function cmpRow(re) {
    for (var i = 0; i < CMP.length; i++) {
      if (re.test(String(CMP[i].aspect || ''))) return CMP[i];
    }
    return null;
  }

  function evalCounts() {
    var subs = 0, leaves = 0;
    TAX.forEach(function (sc) {
      var ss = Array.isArray(sc.subs) ? sc.subs : [];
      subs += ss.length;
      ss.forEach(function (sub) {
        leaves += (Array.isArray(sub.leaves) ? sub.leaves : []).length;
      });
    });
    return { supers: TAX.length, subs: subs, leaves: leaves };
  }

  function buildCombs() {
    var host = pick('combs');
    if (!host) return;
    clear(host);

    var groupRow = cmpRow(/subcategor/i);
    var ec = evalCounts();

    var trainLeaves = typeof H.trainLeaves === 'number' ? H.trainLeaves : null;
    var trainSupers = typeof H.trainSupers === 'number' ? H.trainSupers : null;

    var eUnits = ec.leaves ? taxUnits(TAX) : 0;
    var maxUnits = Math.max(trainLeaves || 0, eUnits, 1);
    var scale = (CW - 2 * CPAD) / maxUnits;

    var key = el('p', 'wd-key',
      'One tick is one leaf category. Both rows use the same tick spacing, so the longer row is the bigger taxonomy.');
    host.appendChild(key);

    if (trainLeaves) {
      var tCount = [];
      if (trainSupers) tCount.push(trainSupers + ' super classes');
      tCount.push(trainLeaves + ' leaf categories');
      var tNote = 'Subcategories per super class: ' +
        (groupRow ? groupRow.train : 'not recorded') +
        '. The paper gives these counts but never lists the training tree, so the ticks here stand for leaf categories in no particular order and the row is drawn flat. Read the flat rule as unknown grouping, not as even grouping.';
      host.appendChild(combBlock(
        'wd-sw-t', 'Training taxonomy', tCount.join(' / '),
        flatComb(trainLeaves, scale,
          'Training taxonomy: a flat run of ' + trainLeaves + ' leaf ticks under one dashed rule, because the paper does not publish where the group boundaries fall.'),
        tNote
      ));
    }

    if (ec.leaves) {
      var eCount = ec.supers + ' super classes / ' + ec.subs + ' subcategories / ' + ec.leaves + ' leaf categories';
      var eNote = 'Subcategories per super class: ' + (groupRow ? groupRow.eval : 'not recorded') +
        '. This row is the real tree from the paper: upper brackets are subcategories, lower brackets are super classes.';
      host.appendChild(combBlock(
        'wd-sw-e', 'Evaluation taxonomy', eCount,
        groupedComb(TAX, scale,
          'Evaluation taxonomy: ' + ec.leaves + ' leaf ticks bracketed into ' + ec.subs +
          ' subcategories of two, grouped again into ' + ec.supers + ' super classes. The run is shorter than the training run because it has fewer leaf categories.'),
        eNote
      ));
    }
  }

  /* ------------------------------------------------------------------
     Part 1b: aspect-by-aspect table
     ------------------------------------------------------------------ */

  function buildTable() {
    var tbl = pick('tbl');
    if (!tbl) return;
    clear(tbl);

    var head = el('div', 'wd-tr wd-thead');
    head.setAttribute('role', 'row');
    var h0 = el('span', 'wd-th wd-asp', 'Aspect');
    h0.setAttribute('role', 'columnheader');
    var h1 = el('span', 'wd-th wd-th-t');
    h1.setAttribute('role', 'columnheader');
    var sw1 = el('i', 'wd-sw wd-sw-t');
    sw1.setAttribute('aria-hidden', 'true');
    h1.appendChild(sw1);
    h1.appendChild(document.createTextNode('Training'));
    var h2 = el('span', 'wd-th wd-th-e');
    h2.setAttribute('role', 'columnheader');
    var sw2 = el('i', 'wd-sw wd-sw-e');
    sw2.setAttribute('aria-hidden', 'true');
    h2.appendChild(sw2);
    h2.appendChild(document.createTextNode('Evaluation'));
    head.appendChild(h0);
    head.appendChild(h1);
    head.appendChild(h2);
    tbl.appendChild(head);

    CMP.forEach(function (r) {
      var tr = el('div', 'wd-tr');
      tr.setAttribute('role', 'row');
      var a = el('span', 'wd-asp', r.aspect);
      a.setAttribute('role', 'rowheader');
      tr.appendChild(a);

      /* The inline Training / Evaluation labels only appear in the stacked
         narrow layout; the column headers already say it to a screen reader. */
      var lt = el('span', 'wd-cl', 'Training');
      lt.setAttribute('aria-hidden', 'true');
      var ct = el('span', 'wd-cell wd-cell-t');
      ct.setAttribute('role', 'cell');
      ct.appendChild(lt);
      ct.appendChild(document.createTextNode(String(r.train)));
      tr.appendChild(ct);

      var le = el('span', 'wd-cl', 'Evaluation');
      le.setAttribute('aria-hidden', 'true');
      var ce = el('span', 'wd-cell wd-cell-e');
      ce.setAttribute('role', 'cell');
      ce.appendChild(le);
      ce.appendChild(document.createTextNode(String(r.eval)));
      tr.appendChild(ce);

      tbl.appendChild(tr);
    });
  }

  function buildSeg() {
    var seg = pick('seg');
    var tbl = pick('tbl');
    if (!seg || !tbl) return;
    clear(seg);

    var modes = [
      { key: 'both', label: 'Both', aria: 'Show both columns' },
      { key: 't', label: 'Training', aria: 'Show the training column only' },
      { key: 'e', label: 'Evaluation', aria: 'Show the evaluation column only' }
    ];
    var btns = [];

    function apply(key) {
      tbl.className = 'wd-tbl' + (key === 'both' ? '' : ' wd-only-' + key);
      btns.forEach(function (b) {
        b.setAttribute('aria-pressed', b.getAttribute('data-key') === key ? 'true' : 'false');
      });
    }

    modes.forEach(function (m) {
      var b = el('button', null, m.label);
      b.type = 'button';
      b.setAttribute('data-key', m.key);
      b.setAttribute('aria-pressed', m.key === 'both' ? 'true' : 'false');
      b.setAttribute('aria-label', m.aria);
      b.addEventListener('click', function () { apply(m.key); });
      btns.push(b);
      seg.appendChild(b);
    });
    apply('both');
  }

  /* ------------------------------------------------------------------
     Part 2: domain divergence explorer
     ------------------------------------------------------------------ */

  /* Leaf counts are read out of the SS description strings themselves, never
     assumed. Returns null when the string states no counterpart. A string that
     enumerates several bracketed groups, as the crime row does, keeps the split
     so the drawing can show it. */
  function leafSpec(s) {
    if (!s) return null;
    var str = String(s);
    var parens = str.match(/\((\d+)\)/g);
    if (parens && parens.length) {
      return parens.map(function (p) { return parseInt(p.replace(/[()]/g, ''), 10); });
    }
    var m = str.match(/(\d+)\s+leaves?/i);
    if (m) return [parseInt(m[1], 10)];
    return null;
  }

  function total(groups) {
    return (groups || []).reduce(function (a, b) { return a + b; }, 0);
  }

  var BW = 360, BH = 68, BX0 = 8, BX1 = 352, GAP_U = 0.85;
  var TOP_Y = 3, BOT_Y = 46, BAR_H = 13, MID_Y = 31, GRP_Y = 63;

  function runUnits(groups) {
    if (!groups || !groups.length) return 0;
    return total(groups) + GAP_U * (groups.length - 1);
  }

  function countPhrase(groups) {
    if (!groups) return 'no counterpart';
    var n = total(groups);
    var base = n + (n === 1 ? ' leaf' : ' leaves');
    return groups.length > 1 ? base + ' in ' + groups.length + ' groups' : base;
  }

  function bandSvg(gT, gE) {
    var lab = 'Training side: ' + countPhrase(gT) +
      '. Evaluation side: ' + countPhrase(gE) +
      '. Both rows use the same bar spacing, so the longer row has more leaf categories. ' +
      'They are drawn offset and unjoined because no leaf maps one to one.';
    var s = sv('svg', {
      viewBox: '0 0 ' + BW + ' ' + BH,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      focusable: 'false'
    });
    s.setAttribute('aria-label', lab);

    var span = BX1 - BX0;
    /* Same pitch on both rows, so run length reads as the leaf count.
       The lower row sits half a pitch across so the two never line up. */
    var slots = Math.max(runUnits(gT), runUnits(gE), 6) + 0.5;
    var pitch = span / slots;
    var w = Math.min(16, Math.max(3, pitch * 0.42));

    function row(groups, y, cls, shift, dropFrom, dropTo, ruleY) {
      if (!groups || !groups.length) return;
      var u = 0;
      groups.forEach(function (g, gi) {
        if (gi) u += GAP_U;
        var first = u;
        for (var i = 0; i < g; i++) {
          var x = BX0 + shift + pitch * u;
          u += 1;
          s.appendChild(sv('rect', {
            x: x.toFixed(2), y: y, width: w.toFixed(2), height: BAR_H, 'class': cls
          }));
          s.appendChild(sv('line', {
            x1: (x + w / 2).toFixed(2), y1: dropFrom,
            x2: (x + w / 2).toFixed(2), y2: dropTo, 'class': 'wd-drop'
          }));
        }
        if (ruleY && groups.length > 1) {
          s.appendChild(sv('line', {
            x1: (BX0 + shift + pitch * first).toFixed(2), y1: ruleY,
            x2: (BX0 + shift + pitch * (u - 1) + w).toFixed(2), y2: ruleY,
            'class': 'wd-base-sc'
          }));
        }
      });
    }

    row(gT, TOP_Y, 'wd-tick-t', 0, TOP_Y + BAR_H, MID_Y - 5, 0);

    if (gE && total(gE)) {
      row(gE, BOT_Y, 'wd-tick-e', pitch / 2, MID_Y + 5, BOT_Y, GRP_Y);
    } else {
      s.appendChild(sv('rect', {
        x: (BX0 + pitch / 2).toFixed(2), y: BOT_Y,
        width: Math.max(pitch * 1.5, 44).toFixed(2),
        height: BAR_H, 'class': 'wd-slot'
      }));
    }

    s.appendChild(sv('line', {
      x1: BX0, y1: MID_Y, x2: BX1, y2: MID_Y, 'class': 'wd-mid'
    }));
    return s;
  }

  function sideBlock(swatchCls, name, count, absent, desc, descCls) {
    var wrap = el('div', 'wd-side');
    var lab = el('div', 'wd-sidelab');
    var sw = el('i', 'wd-sw ' + swatchCls);
    sw.setAttribute('aria-hidden', 'true');
    lab.appendChild(sw);
    lab.appendChild(el('span', 'wd-sidename', name));
    lab.appendChild(el('span', 'wd-sidecount' + (absent ? ' wd-absent' : ''), count));
    wrap.appendChild(lab);
    wrap.appendChild(el('p', 'wd-desc ' + descCls, desc));
    return wrap;
  }

  function renderPanel(idx) {
    var panel = pick('panel');
    if (!panel) return;
    var d = DIV[idx];
    if (!d) return;

    var gT = leafSpec(d.train);
    var gE = leafSpec(d.eval);

    clear(panel);

    panel.appendChild(sideBlock(
      'wd-sw-t', 'Training taxonomy',
      gT === null ? 'count not stated' : countPhrase(gT),
      false, d.train, 'wd-desc-t'
    ));

    var band = el('div', 'wd-band');
    band.appendChild(el('p', 'wd-bandlab',
      gE === null ? 'nothing on the other side' : 'no leaf maps one to one'));
    band.appendChild(bandSvg(gT, gE));
    panel.appendChild(band);

    panel.appendChild(sideBlock(
      'wd-sw-e', 'Evaluation taxonomy',
      gE === null ? 'no counterpart' : countPhrase(gE),
      gE === null, d.eval, 'wd-desc-e'
    ));

    var why = el('div', 'wd-why');
    why.appendChild(el('span', 'wd-whylab', 'Why they diverge'));
    why.appendChild(el('span', 'wd-whytxt', d.why));
    panel.appendChild(why);

    panel.appendChild(el('p', 'wd-src',
      'Counts and wording are the paper’s own divergence table. Each bar stands for one leaf category; where the paper does not name them, the bars are unlabelled placeholders and their order carries no meaning.'));
  }

  function buildChips() {
    var chips = pick('chips');
    var panel = pick('panel');
    if (!chips || !panel || !DIV.length) return;
    clear(chips);

    var btns = [];

    function select(i, focus) {
      btns.forEach(function (b, j) {
        var on = i === j;
        b.setAttribute('aria-selected', on ? 'true' : 'false');
        b.setAttribute('tabindex', on ? '0' : '-1');
      });
      panel.setAttribute('aria-labelledby', ID + '-tab-' + i);
      renderPanel(i);
      if (focus && btns[i]) btns[i].focus();
    }

    DIV.forEach(function (d, i) {
      var b = el('button', 'wd-chip');
      b.type = 'button';
      b.id = ID + '-tab-' + i;
      b.setAttribute('role', 'tab');
      b.setAttribute('aria-controls', ID + '-panel');
      b.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
      b.setAttribute('tabindex', i === 0 ? '0' : '-1');
      var mark = el('span', 'wd-chipmark', String(i + 1).length === 1 ? '0' + (i + 1) : String(i + 1));
      mark.setAttribute('aria-hidden', 'true');
      b.appendChild(mark);
      b.appendChild(document.createTextNode(String(d.domain)));
      b.addEventListener('click', function () { select(i, false); });
      b.addEventListener('keydown', function (ev) {
        var k = ev.key, n = btns.length, next = -1;
        if (k === 'ArrowRight' || k === 'ArrowDown') next = (i + 1) % n;
        else if (k === 'ArrowLeft' || k === 'ArrowUp') next = (i - 1 + n) % n;
        else if (k === 'Home') next = 0;
        else if (k === 'End') next = n - 1;
        if (next >= 0) { ev.preventDefault(); select(next, true); }
      });
      btns.push(b);
      chips.appendChild(b);
    });

    select(0, false);
  }

  /* ------------------------------------------------------------------
     Standing note
     ------------------------------------------------------------------ */

  function buildStanding() {
    var n = pick('standing');
    if (!n) return;
    clear(n);

    var supers = typeof H.evalSupers === 'number' ? H.evalSupers : evalCounts().supers;
    var loose = typeof H.looseCounterparts === 'number' ? H.looseCounterparts : null;
    var lead = loose !== null
      ? loose + ' of the ' + supers + ' evaluation super classes'
      : 'Most of the ' + supers + ' evaluation super classes';

    n.appendChild(el('strong', null, lead + ' have a loose counterpart in the training taxonomy, but no leaf category maps one to one.'));
    n.appendChild(document.createTextNode(
      ' That is the point of building the evaluation taxonomy separately: a score on it cannot be explained away as the model recognising a label it was trained on.'
    ));
  }

  buildCombs();
  buildTable();
  buildSeg();
  buildChips();
  buildStanding();
})();
