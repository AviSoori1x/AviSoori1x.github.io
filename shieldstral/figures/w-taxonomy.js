/* w-taxonomy: explorable Appendix B evaluation taxonomy with per category F1.
   Every displayed value is read from window.SS at runtime. */
(function () {
  var ID = 'w-taxonomy';
  var root = document.getElementById(ID);
  if (!root) return;

  var SS = window.SS;
  if (!SS || !SS.evalTaxonomy || !SS.taxonomyModels) return;

  var MODELS = SS.taxonomyModels;
  var TAX = SS.evalTaxonomy;

  /* Notes for rows the paper leaves blank. Prefer strings supplied by the data
     file; the fallbacks below carry no figures of their own. */
  var DATA_NOTES = SS.taxonomyNotes || {};
  var FALLBACK_NOTES = {
    'Drug Operations': 'This subcategory has no row in the published results table.',
    'Physical Property': 'No scores are reported for this row: there were too few evaluation samples for the F1 to be reliable.'
  };
  var GENERIC_NOTE = 'No scores are reported for this row.';

  /* ---------- indices ---------- */
  var SIDX = -1, i;
  for (i = 0; i < MODELS.length; i++) { if (/shieldstral/i.test(MODELS[i])) SIDX = i; }
  if (SIDX < 0) SIDX = MODELS.length - 1;

  var cmpIdx = -1;
  for (i = 0; i < MODELS.length; i++) { if (/gpt-?oss/i.test(MODELS[i]) && i !== SIDX) cmpIdx = i; }
  if (cmpIdx < 0) { for (i = 0; i < MODELS.length; i++) { if (i !== SIDX) { cmpIdx = i; break; } } }
  if (cmpIdx < 0) return;

  /* ---------- flatten the taxonomy into a tree of uniform nodes ---------- */
  function slug(s) { return String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-'); }

  function build(src, level, parentKey, order) {
    var node = {
      name: src.name,
      id: src.id || '',
      f1: (src.f1 && src.f1.length) ? src.f1 : null,
      level: level,
      order: order,
      kids: []
    };
    node.key = (parentKey ? parentKey + '.' : '') + level + '-' + order + '-' + slug(src.name);
    var kids = src.subs || src.leaves || [];
    for (var k = 0; k < kids.length; k++) node.kids.push(build(kids[k], level + 1, node.key, k));
    return node;
  }

  var TREE = [];
  for (i = 0; i < TAX.length; i++) TREE.push(build(TAX[i], 1, '', i));

  var ALL = [];
  (function collect(list) {
    for (var j = 0; j < list.length; j++) { ALL.push(list[j]); collect(list[j].kids); }
  })(TREE);

  function val(node, idx) {
    if (!node.f1) return null;
    var v = node.f1[idx];
    return (typeof v === 'number') ? v : null;
  }

  function noteFor(node) {
    if (typeof DATA_NOTES[node.name] === 'string') return DATA_NOTES[node.name];
    var base = FALLBACK_NOTES[node.name] || GENERIC_NOTE;
    var scored = 0;
    for (var j = 0; j < node.kids.length; j++) if (node.kids[j].f1) scored++;
    if (scored === 1) {
      base += ' Its single leaf category is scored, so expand the row to see it.';
    } else if (scored > 1) {
      base += ' Its ' + scored + ' leaf categories are scored, so expand the row to see them.';
    }
    return base;
  }

  /* ---------- state ---------- */
  var mode = 'tax';
  var open = {};
  var pendingFocus = null;

  /* ---------- markup helpers ---------- */
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function fmt(v) { return v.toFixed(1); }

  var TICKS = '<rect class="tx-tk" x="24.85" y="0.4" width="0.3" height="5.2"/>' +
              '<rect class="tx-tk" x="49.85" y="0.4" width="0.3" height="5.2"/>' +
              '<rect class="tx-tk" x="74.85" y="0.4" width="0.3" height="5.2"/>';

  function bar(v, kind) {
    var s = '<svg class="tx-svg" viewBox="0 0 100 6" preserveAspectRatio="none" aria-hidden="true" focusable="false">' +
            '<rect class="tx-trk" x="0" y="0.4" width="100" height="5.2"/>' + TICKS;
    if (v !== null && v > 0) {
      if (kind === 's') {
        s += '<rect class="tx-sbar" x="0" y="0.4" width="' + v + '" height="5.2"/>';
      } else {
        s += '<rect class="tx-cbar" x="0.3" y="0.9" width="' + Math.max(v - 0.6, 0.4) +
             '" height="4.2" vector-effect="non-scaling-stroke"/>';
      }
    }
    return s + '</svg>';
  }

  function barLine(v, kind, modelName) {
    if (v === null) {
      return '<div class="tx-nrline"><span class="tx-sr">' + esc(modelName) + ': </span>not reported</div>';
    }
    return bar(v, kind) +
      '<span class="tx-num"><span class="tx-sr">' + esc(modelName) + ' F1 </span>' + fmt(v) + '</span>';
  }

  var CARET = '<span class="tx-caret" aria-hidden="true"><svg viewBox="0 0 8 8" width="7" height="7" focusable="false"><path d="M1.4 0.4 L6.6 4 L1.4 7.6 Z" fill="currentColor"/></svg></span>';
  var DOT = '<span class="tx-dot" aria-hidden="true"><svg viewBox="0 0 8 8" width="7" height="7" focusable="false"><rect x="2.6" y="2.6" width="2.8" height="2.8" fill="currentColor"/></svg></span>';

  function rowHtml(node, cIdx) {
    var s = val(node, SIDX);
    var c = val(node, cIdx);
    var hasKids = node.kids.length > 0;
    var isOpen = !!open[node.key];
    var cls = 'tx-row tx-l' + node.level + (isOpen ? ' tx-open' : '');
    var label = (node.id ? '<span class="tx-id">' + esc(node.id) + '</span>' : '') +
                '<span class="tx-name">' + esc(node.name) + '</span>';

    var hit;
    if (hasKids) {
      hit = '<button type="button" class="tx-hit" aria-expanded="' + (isOpen ? 'true' : 'false') + '"' +
            (isOpen ? ' aria-controls="' + ID + '-g-' + esc(node.key) + '"' : '') + '>' +
            CARET + label + '</button>';
    } else {
      hit = '<div class="tx-hit tx-static">' + DOT + label + '</div>';
    }

    var bars, why = '';
    if (node.f1 === null || (s === null && c === null)) {
      bars = '<div class="tx-bars"><div class="tx-nr">not reported in the paper</div></div>';
      why = '<p class="tx-why">' + esc(noteFor(node)) + '</p>';
    } else {
      bars = '<div class="tx-bars">' +
             barLine(s, 's', MODELS[SIDX]) +
             barLine(c, 'c', MODELS[cIdx]) +
             '</div>';
    }

    var gap;
    if (s === null || c === null) {
      gap = '<div class="tx-gap tx-na"><span class="tx-sr">gap not available</span>' +
            '<span aria-hidden="true">n/a</span></div>';
    } else {
      var g = s - c;
      var gcls = g > 0.05 ? 'tx-up' : (g < -0.05 ? 'tx-dn' : '');
      var txt = (g >= 0 ? '+' : '-') + Math.abs(g).toFixed(1);
      gap = '<div class="tx-gap ' + gcls + '"><span class="tx-sr">gap </span>' + txt + '</div>';
    }

    var out = '<div class="' + cls + '" data-key="' + esc(node.key) + '">' + hit + bars + gap + why + '</div>';
    if (hasKids && isOpen) {
      out += '<div class="tx-kids" id="' + ID + '-g-' + esc(node.key) + '">' +
             listHtml(node.kids, cIdx) + '</div>';
    }
    return out;
  }

  function sortKey(node, cIdx) {
    if (mode === 'ss') return val(node, SIDX);
    var s = val(node, SIDX), c = val(node, cIdx);
    return (s === null || c === null) ? null : s - c;
  }

  function listHtml(list, cIdx) {
    var arr = list.slice();
    if (mode !== 'tax') {
      arr.sort(function (a, b) {
        var ka = sortKey(a, cIdx), kb = sortKey(b, cIdx);
        if (ka === null && kb === null) return a.order - b.order;
        if (ka === null) return 1;
        if (kb === null) return -1;
        if (ka !== kb) return mode === 'gap' ? ka - kb : kb - ka;
        return a.order - b.order;
      });
    }
    var out = '';
    for (var j = 0; j < arr.length; j++) out += rowHtml(arr[j], cIdx);
    return out;
  }

  /* ---------- derived summary ---------- */
  function statsHtml(cIdx) {
    var n = 0, wins = 0, miss = 0, worst = null;
    for (var j = 0; j < ALL.length; j++) {
      var s = val(ALL[j], SIDX), c = val(ALL[j], cIdx);
      if (s === null || c === null) { miss++; continue; }
      n++;
      var g = s - c;
      if (g > 0.05) wins++;
      if (worst === null || g < worst.g) worst = { g: g, name: ALL[j].name };
    }

    var h = SS.headline || {};
    var shape = [];
    if (h.evalSupers != null) shape.push(h.evalSupers + ' super');
    if (h.evalSubs != null) shape.push(h.evalSubs + ' sub');
    if (h.evalLeaves != null) shape.push(h.evalLeaves + ' leaf');

    var deficit;
    if (worst && worst.g < -0.05) {
      deficit = '<span class="tx-dn">' + esc(worst.name) + ' ' + worst.g.toFixed(1) + '</span>';
    } else {
      deficit = 'none';
    }

    var items = [];
    // ALL.length is 12 + 26 + 52. The paper's table carries one row fewer, because
    // the Drug Operations subcategory has no published row, so say both numbers
    // rather than showing a count that does not add up against the shape beside it.
    items.push(['Categories', esc(String(ALL.length)) +
      (shape.length ? '<br><span class="tx-sm">' + esc(shape.join(', ')) + '</span>' : '')]);
    if (SS.taxonomyRowCount != null && SS.taxonomyRowCount !== ALL.length) {
      items.push(['Published rows', esc(String(SS.taxonomyRowCount)) +
        '<br><span class="tx-sm">the paper omits ' + (ALL.length - SS.taxonomyRowCount) +
        '</span>']);
    }
    items.push(['Shieldstral ahead', wins + ' of ' + n +
      '<br><span class="tx-sm">' + esc(MODELS[cIdx]) + '</span>']);
    items.push(['Largest deficit', deficit]);
    items.push(['Not comparable', miss + '<br><span class="tx-sm">score missing on one side</span>']);

    var out = '';
    for (var k = 0; k < items.length; k++) {
      out += '<div class="tx-stat"><dt>' + items[k][0] + '</dt><dd>' + items[k][1] + '</dd></div>';
    }
    return out;
  }

  function legendHtml(cIdx) {
    return '<span class="tx-key tx-key-s">' +
      '<svg viewBox="0 0 26 8" width="26" height="8" aria-hidden="true" focusable="false">' +
      '<rect class="tx-sw-s" x="0" y="1.5" width="26" height="5"/></svg>' +
      '<span class="tx-kn">' + esc(MODELS[SIDX]) + ', solid</span></span>' +
      '<span class="tx-key tx-key-c">' +
      '<svg viewBox="0 0 26 8" width="26" height="8" aria-hidden="true" focusable="false">' +
      '<rect class="tx-sw-c" x="0.5" y="2" width="25" height="4"/></svg>' +
      '<span class="tx-kn">' + esc(MODELS[cIdx]) + ', outline</span></span>';
  }

  /* ---------- wiring ---------- */
  var sel = document.getElementById(ID + '-cmp');
  var tree = document.getElementById(ID + '-tree');
  var statBox = document.getElementById(ID + '-stats');
  var legend = document.getElementById(ID + '-legend');
  var allBtn = document.getElementById(ID + '-all');
  var sortBtns = root.querySelectorAll('.tx-seg button');
  if (!sel || !tree || !statBox || !legend || !allBtn) return;

  var opts = '';
  for (i = 0; i < MODELS.length; i++) {
    if (i === SIDX) continue;
    opts += '<option value="' + i + '"' + (i === cmpIdx ? ' selected' : '') + '>' + esc(MODELS[i]) + '</option>';
  }
  sel.innerHTML = opts;

  function expandableKeys() {
    var keys = [];
    for (var j = 0; j < ALL.length; j++) if (ALL[j].kids.length) keys.push(ALL[j].key);
    return keys;
  }

  function syncAllBtn() {
    var keys = expandableKeys(), openCount = 0;
    for (var j = 0; j < keys.length; j++) if (open[keys[j]]) openCount++;
    var all = keys.length > 0 && openCount === keys.length;
    allBtn.setAttribute('aria-pressed', all ? 'true' : 'false');
    allBtn.textContent = all ? 'Collapse all' : 'Expand all';
  }

  function render() {
    var cIdx = parseInt(sel.value, 10);
    if (isNaN(cIdx)) cIdx = cmpIdx;
    cmpIdx = cIdx;
    statBox.innerHTML = statsHtml(cIdx);
    legend.innerHTML = legendHtml(cIdx);
    tree.innerHTML = listHtml(TREE, cIdx);
    syncAllBtn();
    if (pendingFocus) {
      var el = tree.querySelector('.tx-row[data-key="' + pendingFocus + '"] > button.tx-hit');
      if (el) el.focus();
      pendingFocus = null;
    }
  }

  tree.addEventListener('click', function (ev) {
    var btn = ev.target;
    while (btn && btn !== tree && !(btn.tagName && btn.tagName.toLowerCase() === 'button')) {
      btn = btn.parentNode;
    }
    if (!btn || btn === tree || !btn.parentNode) return;
    var key = btn.parentNode.getAttribute ? btn.parentNode.getAttribute('data-key') : null;
    if (!key) return;
    if (open[key]) delete open[key]; else open[key] = true;
    pendingFocus = key;
    render();
  });

  sel.addEventListener('change', render);

  for (i = 0; i < sortBtns.length; i++) {
    sortBtns[i].addEventListener('click', function (ev) {
      var next = ev.currentTarget.getAttribute('data-sort');
      mode = next;
      for (var j = 0; j < sortBtns.length; j++) {
        sortBtns[j].setAttribute('aria-pressed',
          sortBtns[j].getAttribute('data-sort') === next ? 'true' : 'false');
      }
      render();
    });
  }

  allBtn.addEventListener('click', function () {
    var keys = expandableKeys(), openCount = 0, j;
    for (j = 0; j < keys.length; j++) if (open[keys[j]]) openCount++;
    if (openCount === keys.length) {
      open = {};
    } else {
      for (j = 0; j < keys.length; j++) open[keys[j]] = true;
    }
    render();
  });

  render();
})();
