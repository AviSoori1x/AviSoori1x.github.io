window.SCENES = window.SCENES || {};

/* S_ANCESTOR, act 2, beat 18. Free positives up the tree.
   A three level tree is drawn straight from window.SS.evalTaxonomy, super class
   then subcategory then leaf. Pick a leaf and the path up to its super class
   lights, and the positive rows that path buys stack on the right, one per
   level, all labelled yes. The counter keeps the ledger honest, one LLM call
   against N rows.
   Every count on screen is counted from SS.evalTaxonomy at runtime, the yes
   label comes from SS.fig3.positive.label, the sampling temperature from
   SS.headline.genTemp, and the training tree sizes from
   SS.headline.trainSupers and SS.headline.trainLeaves.
   The query wording is composed for this figure and says so on screen. */
window.SCENES['S_ANCESTOR'] = function (root, api) {
  var SS = api.SS || {};
  var TAX = SS.evalTaxonomy || [];
  var head = SS.headline || {};
  var F = SS.fig3 || {};
  var POS = F.positive || {};

  var ID = 'S_ANCESTOR';
  var DOT = '·';
  var YES = String(POS.label == null ? 'yes' : POS.label).toLowerCase();
  var TEMP = head.genTemp;
  var TRAIN_SC = head.trainSupers;
  var TRAIN_LF = head.trainLeaves;

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ---------- flatten the tree, counts are counted, never typed ---------- */
  var LEAVES = [];
  var nSupers = TAX.length;
  var nSubs = 0;
  TAX.forEach(function (sc, i) {
    var subs = sc.subs || [];
    nSubs += subs.length;
    subs.forEach(function (sb, j) {
      (sb.leaves || []).forEach(function (lf, k) {
        LEAVES.push({
          sc: i, sub: j, leaf: k,
          scId: sc.id, scName: sc.name,
          subName: sb.name,
          id: lf.id, name: lf.name
        });
      });
    });
  });
  var nLeaves = LEAVES.length;
  var DEPTH = 3;                       /* super class, subcategory, leaf */
  var totalRows = nLeaves * DEPTH;

  /* ---------- tree markup ---------- */
  function twisty() {
    return '<svg class="tw" viewBox="0 0 12 12" aria-hidden="true">'
      + '<path d="M4 2.4 L8.4 6 L4 9.6"></path></svg>';
  }
  function tickSvg() {
    return '<svg class="tick" viewBox="0 0 20 20" aria-hidden="true">'
      + '<path d="M4 10.4 L8.2 14.6 L16 6.1"></path></svg>';
  }

  var treeHtml = TAX.map(function (sc, i) {
    var subs = sc.subs || [];
    var count = 0;
    subs.forEach(function (sb) { count += (sb.leaves || []).length; });

    var kids = subs.map(function (sb, j) {
      var lv = (sb.leaves || []).map(function (lf, k) {
        return '<button type="button" class="leaf" role="treeitem" aria-selected="false"'
          + ' aria-level="3" aria-posinset="' + (k + 1) + '"'
          + ' aria-setsize="' + (sb.leaves || []).length + '"'
          + ' tabindex="-1" id="' + ID + '-lf-' + i + '-' + j + '-' + k + '">'
          + '<span class="mk" aria-hidden="true"></span>'
          + '<span class="lnm">' + esc(lf.name) + '</span>'
          + '<span class="lid">' + esc(lf.id) + '</span>'
          + '<span class="rkb" aria-hidden="true">1</span>'
          + '</button>';
      }).join('');
      return '<div class="subwrap" id="' + ID + '-sw-' + i + '-' + j + '">'
        + '<button type="button" class="subb" role="treeitem" aria-selected="false"'
        + ' aria-level="2" aria-posinset="' + (j + 1) + '"'
        + ' aria-setsize="' + subs.length + '" aria-expanded="true"'
        + ' tabindex="-1" id="' + ID + '-sb-' + i + '-' + j + '">'
        + '<span class="mk" aria-hidden="true"></span>'
        + '<span class="snm">' + esc(sb.name) + '</span>'
        + '<span class="rkb" aria-hidden="true">2</span>'
        + '</button>'
        + '<div class="leaves" role="group">' + lv + '</div>'
        + '</div>';
    }).join('');

    return '<div class="grp" id="' + ID + '-grp-' + i + '">'
      + '<button type="button" class="scb" role="treeitem" aria-expanded="false"'
      + ' aria-level="1" aria-posinset="' + (i + 1) + '"'
      + ' aria-setsize="' + TAX.length + '"'
      + ' aria-selected="false" tabindex="-1" id="' + ID + '-sc-' + i + '">'
      + twisty()
      + '<span class="scnm">' + esc(sc.name) + '</span>'
      + '<span class="scid">' + esc(sc.id) + '</span>'
      + '<span class="sccnt">' + count + '</span>'
      + '<span class="rkb" aria-hidden="true">3</span>'
      + '</button>'
      + '<div class="kids" role="group">' + kids + '</div>'
      + '</div>';
  }).join('');

  /* ---------- the three rows one leaf buys ---------- */
  function rowHtml(i) {
    return '<div class="row" id="' + ID + '-row-' + i + '">'
      + '<span class="gut">'
      +   '<span class="rk">' + (i + 1) + '</span>'
      +   '<span class="cost" id="' + ID + '-cost-' + i + '"></span>'
      + '</span>'
      + '<span class="mid">'
      +   '<span class="lvl" id="' + ID + '-lvl-' + i + '"></span>'
      +   '<span class="cnm" id="' + ID + '-nm-' + i + '"></span>'
      +   '<span class="q" id="' + ID + '-q-' + i + '"></span>'
      + '</span>'
      + '<span class="ans">' + tickSvg()
      +   '<b>' + esc(YES) + '</b><i>label</i>'
      + '</span>'
      + '</div>';
  }

  var whichNote =
    'This is the <b class="ev">evaluation</b> taxonomy, ' + nSupers + ' super classes, '
    + nSubs + ' subcategories, ' + nLeaves + ' leaves, the one tree the paper prints in full. '
    + 'The trick itself runs on the <b>training</b> taxonomy, a different tree of '
    + (TRAIN_SC != null ? TRAIN_SC : 'its own') + ' super classes and '
    + (TRAIN_LF != null ? TRAIN_LF : 'its own') + ' leaves. Same shape, other labels.';

  /* classList.add, never root.className, the engine's own .scene class lives here too */
  root.classList.add('sc-s_ancestor');
  root.appendChild(api.frag(
    '<div class="wrap">'

    + '<div class="hd">'
    +   '<span class="eyebrow">free positives</span>'
    +   '<span class="hnote">violate a leaf and you violate every ancestor of it</span>'
    + '</div>'

    + '<div class="cols">'

    /* ---- left, the tree ---- */
    + '<div class="left">'
    +   '<div class="lhd">'
    +     '<span class="klab hot">pick a leaf</span>'
    +     '<span class="pin">' + nLeaves + ' leaves</span>'
    +   '</div>'
    +   '<div class="tree" id="' + ID + '-tree" role="tree"'
    +     ' aria-label="evaluation taxonomy, super class, subcategory, leaf">'
    +     treeHtml
    +   '</div>'
    +   '<p class="which">' + whichNote + '</p>'
    + '</div>'

    /* ---- right, the rows that leaf buys ---- */
    + '<div class="right">'
    +   '<div class="call">'
    +     '<div class="callbar">'
    +       '<span class="klab hot">llm rewrite</span>'
    +       '<span class="pin">1 call'
    +         (TEMP != null ? (' ' + DOT + ' temp ' + TEMP) : '') + '</span>'
    +     '</div>'
    +     '<p class="calltxt">target <b id="' + ID + '-target"></b></p>'
    +     '<p class="cout">one unsafe document out, reused unchanged by every row below</p>'
    +   '</div>'

    +   '<div class="rows" id="' + ID + '-rows" aria-live="off"'
    +     ' aria-label="the rows this leaf produces">'
    +     rowHtml(0) + rowHtml(1) + rowHtml(2)
    +   '</div>'
    + '</div>'

    + '</div>'

    + '<div class="ledger">'
    +   '<span class="cell one"><span class="big amb">1</span>'
    +     '<span class="lab">LLM call<i>one rewrite</i></span></span>'
    +   '<svg class="arw" viewBox="0 0 34 20" aria-hidden="true">'
    +     '<path d="M2 10 H27 M21 4.4 L27.4 10 L21 15.6"></path></svg>'
    +   '<span class="cell two"><span class="big lim" id="' + ID + '-n">0</span>'
    +     '<span class="lab">positive rows<i id="' + ID + '-free">stacking</i></span></span>'
    +   '<p class="scale">Every leaf sits ' + DEPTH + ' levels deep, so one rewrite yields '
    +     'one row per level. The paper reports 11 training super classes and 73 training '
    +     'leaves but does not publish the training tree, so no total is computed here.</p>'
    + '</div>'

    + '<div class="foot">'
    +   '<span class="gt">Labels are the taxonomy ground truth of the pipeline, not a live '
    +     'model call. The query wording is composed for this figure.</span>'
    +   '<span class="hint" id="' + ID + '-hint">walking the tree on its own, click or use '
    +     'arrow keys</span>'
    + '</div>'

    + '</div>'
  ).firstChild);

  /* ---------- handles ---------- */
  var treeEl = root.querySelector('#' + ID + '-tree');
  var rowsEl = root.querySelector('#' + ID + '-rows');
  var grpEls = [], scEls = [], swEls = {}, sbEls = {}, lfEls = {};
  TAX.forEach(function (sc, i) {
    grpEls.push(root.querySelector('#' + ID + '-grp-' + i));
    scEls.push(root.querySelector('#' + ID + '-sc-' + i));
    (sc.subs || []).forEach(function (sb, j) {
      swEls[i + '.' + j] = root.querySelector('#' + ID + '-sw-' + i + '-' + j);
      sbEls[i + '.' + j] = root.querySelector('#' + ID + '-sb-' + i + '-' + j);
      (sb.leaves || []).forEach(function (lf, k) {
        lfEls[i + '.' + j + '.' + k] = root.querySelector('#' + ID + '-lf-' + i + '-' + j + '-' + k);
      });
    });
  });

  var rowEls = [0, 1, 2].map(function (i) { return root.querySelector('#' + ID + '-row-' + i); });
  var lvlEls = [0, 1, 2].map(function (i) { return root.querySelector('#' + ID + '-lvl-' + i); });
  var nmEls = [0, 1, 2].map(function (i) { return root.querySelector('#' + ID + '-nm-' + i); });
  var qEls = [0, 1, 2].map(function (i) { return root.querySelector('#' + ID + '-q-' + i); });
  var costEls = [0, 1, 2].map(function (i) { return root.querySelector('#' + ID + '-cost-' + i); });
  var nEl = root.querySelector('#' + ID + '-n');
  var freeEl = root.querySelector('#' + ID + '-free');
  var targetEl = root.querySelector('#' + ID + '-target');
  var hintEl = root.querySelector('#' + ID + '-hint');

  /* ---------- state ---------- */
  var cursor = 0;
  var revealed = 0;
  var auto = true;
  var timers = [];
  var openSc = -1;

  function clearTimers() {
    timers.forEach(function (t) { clearTimeout(t); });
    timers = [];
  }

  function setRevealed(n) {
    revealed = n;
    for (var i = 0; i < DEPTH; i++) rowEls[i].classList.toggle('lit', i < n);
    nEl.textContent = String(n);
    nEl.classList.toggle('zero', n === 0);
    freeEl.textContent = n === 0
      ? 'stacking'
      : (n === 1 ? '1 paid for by the call' : ((n - 1) + ' of them free'));
    /* the tree badges only claim a rank once the row exists */
    var L = LEAVES[cursor];
    if (!L) return;
    var lf = lfEls[L.sc + '.' + L.sub + '.' + L.leaf];
    var sb = sbEls[L.sc + '.' + L.sub];
    var sc = scEls[L.sc];
    if (lf) lf.classList.toggle('counted', n >= 1);
    if (sb) sb.classList.toggle('counted', n >= 2);
    if (sc) sc.classList.toggle('counted', n >= 3);
  }

  function openSuper(i) {
    if (openSc === i) return;
    openSc = i;
    grpEls.forEach(function (g, k) { g.classList.toggle('open', k === i); });
    scEls.forEach(function (b, k) { b.setAttribute('aria-expanded', k === i ? 'true' : 'false'); });
  }

  function query(name) {
    return 'Does this content violate "' + name + '"?';
  }

  function paint(L) {
    /* clear every path mark, then set this one */
    Object.keys(lfEls).forEach(function (k) {
      lfEls[k].classList.remove('sel', 'counted');
      lfEls[k].setAttribute('aria-selected', 'false');
    });
    Object.keys(sbEls).forEach(function (k) {
      sbEls[k].classList.remove('path', 'counted');
      sbEls[k].setAttribute('aria-selected', 'false');
    });
    Object.keys(swEls).forEach(function (k) { swEls[k].classList.remove('path'); });
    scEls.forEach(function (b) { b.classList.remove('path', 'counted'); });

    var lf = lfEls[L.sc + '.' + L.sub + '.' + L.leaf];
    var sb = sbEls[L.sc + '.' + L.sub];
    var sw = swEls[L.sc + '.' + L.sub];
    if (lf) { lf.classList.add('sel'); lf.setAttribute('aria-selected', 'true'); }
    if (sb) sb.classList.add('path');
    if (sw) sw.classList.add('path');
    scEls[L.sc].classList.add('path');

    targetEl.textContent = L.name + ' ' + DOT + ' ' + L.id;

    var spec = [
      { lvl: 'leaf category ' + DOT + ' 1 of ' + nLeaves, name: L.name, cost: 'generated' },
      { lvl: 'subcategory ' + DOT + ' 1 of ' + nSubs, name: L.subName, cost: 'free' },
      { lvl: 'super class ' + DOT + ' 1 of ' + nSupers, name: L.scName, cost: 'free' }
    ];
    for (var i = 0; i < DEPTH; i++) {
      lvlEls[i].textContent = spec[i].lvl;
      nmEls[i].textContent = spec[i].name;
      qEls[i].textContent = query(spec[i].name);
      costEls[i].textContent = spec[i].cost;
      rowEls[i].classList.toggle('freebie', spec[i].cost === 'free');
    }
  }

  function stackNow() {
    clearTimers();
    if (api.reduce) { setRevealed(DEPTH); return; }
    setRevealed(1);
    timers.push(setTimeout(function () { setRevealed(2); }, 260));
    timers.push(setTimeout(function () { setRevealed(3); }, 520));
  }

  function selectLeaf(idx, fromUser) {
    if (idx < 0 || idx >= nLeaves) return;
    cursor = idx;
    var L = LEAVES[idx];
    openSuper(L.sc);
    paint(L);
    if (fromUser) {
      if (auto) {
        auto = false;
        hintEl.textContent = 'manual ' + DOT + ' arrow keys walk the tree, enter picks';
        /* quiet while it walks itself, announced once the reader is driving */
        rowsEl.setAttribute('aria-live', 'polite');
      }
      stackNow();
    }
  }

  function leafIndexOf(i, j, k) {
    for (var n = 0; n < nLeaves; n++) {
      var L = LEAVES[n];
      if (L.sc === i && L.sub === j && L.leaf === k) return n;
    }
    return -1;
  }
  function firstLeafOfSub(i, j) { return leafIndexOf(i, j, 0); }
  function firstLeafOfSuper(i) { return leafIndexOf(i, 0, 0); }

  /* ---------- clicks ---------- */
  scEls.forEach(function (b, i) {
    b.addEventListener('click', function () {
      var n = firstLeafOfSuper(i);
      if (n >= 0) selectLeaf(n, true);
      focusRow(b);
    });
  });
  Object.keys(sbEls).forEach(function (key) {
    var p = key.split('.');
    sbEls[key].addEventListener('click', function () {
      var n = firstLeafOfSub(+p[0], +p[1]);
      if (n >= 0) selectLeaf(n, true);
      focusRow(sbEls[key]);
    });
  });
  Object.keys(lfEls).forEach(function (key) {
    var p = key.split('.');
    lfEls[key].addEventListener('click', function () {
      selectLeaf(leafIndexOf(+p[0], +p[1], +p[2]), true);
      focusRow(lfEls[key]);
    });
  });

  /* ---------- keyboard, roving tabindex over the visible rows ---------- */
  function visibleRows() {
    var out = [];
    TAX.forEach(function (sc, i) {
      out.push(scEls[i]);
      if (i !== openSc) return;
      (sc.subs || []).forEach(function (sb, j) {
        out.push(sbEls[i + '.' + j]);
        (sb.leaves || []).forEach(function (lf, k) {
          out.push(lfEls[i + '.' + j + '.' + k]);
        });
      });
    });
    return out;
  }
  function focusRow(el) {
    var vis = visibleRows();
    vis.forEach(function (v) { v.tabIndex = (v === el) ? 0 : -1; });
    if (vis.indexOf(el) < 0) el.tabIndex = 0;
  }
  treeEl.addEventListener('keydown', function (e) {
    var vis = visibleRows();
    var at = vis.indexOf(document.activeElement);
    var k = e.key, next = -1;
    if (k === 'ArrowDown') next = (at < 0) ? 0 : Math.min(at + 1, vis.length - 1);
    else if (k === 'ArrowUp') next = (at < 0) ? 0 : Math.max(at - 1, 0);
    else if (k === 'Home') next = 0;
    else if (k === 'End') next = vis.length - 1;
    else if (k === 'ArrowRight') {
      if (at >= 0 && vis[at].classList.contains('scb') && vis[at] !== scEls[openSc]) {
        var s = scEls.indexOf(vis[at]);
        var n0 = firstLeafOfSuper(s);
        if (n0 >= 0) selectLeaf(n0, true);
        e.preventDefault();
        return;
      }
      next = (at < 0) ? 0 : Math.min(at + 1, vis.length - 1);
    } else if (k === 'ArrowLeft') {
      if (at >= 0 && !vis[at].classList.contains('scb')) {
        next = vis.indexOf(scEls[openSc]);
      } else next = (at < 0) ? 0 : Math.max(at - 1, 0);
    } else if (k === 'Enter' || k === ' ' || k === 'Spacebar') {
      return; /* buttons fire their own click */
    } else return;
    e.preventDefault();
    if (next < 0 || next >= vis.length) return;
    focusRow(vis[next]);
    vis[next].focus();
  });

  /* ---------- first paint ---------- */
  selectLeaf(0, false);
  setRevealed(api.reduce ? DEPTH : 0);
  focusRow(scEls[0]);
  if (api.reduce) hintEl.textContent = 'click a leaf, or use the arrow keys';

  /* ---------- ambient walk, so a scroller sees the mechanic ---------- */
  var running = false, nextAt = null;
  return {
    start: function () {
      running = true;
      nextAt = null;
      if (auto) { selectLeaf(0, false); setRevealed(0); }
      /* a reader who scrolled away mid stack comes back to the finished stack */
      else if (revealed < DEPTH) setRevealed(DEPTH);
    },
    stop: function () { running = false; clearTimers(); },
    tick: function (t) {
      if (!running || !auto || api.reduce) return;
      if (nextAt === null) { nextAt = t + 0.55; return; }
      if (t < nextAt) return;
      if (revealed < DEPTH) {
        setRevealed(revealed + 1);
        nextAt = t + (revealed < DEPTH ? 0.8 : 2.5);
      } else {
        selectLeaf((cursor + 1) % nLeaves, false);
        setRevealed(0);
        nextAt = t + 0.6;
      }
    }
  };
};
