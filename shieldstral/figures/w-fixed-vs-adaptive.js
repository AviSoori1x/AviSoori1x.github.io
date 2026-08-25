(function () {
  var RID = 'w-fixed-vs-adaptive';
  var root = document.getElementById(RID);
  if (!root) return;

  var SS = (typeof window !== 'undefined') ? window.SS : null;
  if (!SS) return;

  var levels = Array.isArray(SS.strictness) ? SS.strictness : [];
  var roster = Array.isArray(SS.baselines) ? SS.baselines : [];
  var fig2 = Array.isArray(SS.fig2) ? SS.fig2 : [];
  var sample = fig2.length ? fig2[0] : null;
  var head = SS.headline || {};
  if (!levels.length || !roster.length || !sample || !SS.systemPrompt) return;

  function el(suffix) { return document.getElementById(RID + '-' + suffix); }
  function put(node, text) { if (node) node.textContent = text; }

  var node = {
    seg:       el('seg'),
    why:       el('why'),
    fxPolicy:  el('fx-policy'),
    fxDoc:     el('fx-doc'),
    fxOut:     el('fx-out'),
    fxStamp:   el('fx-stamp'),
    adSystem:  el('ad-system'),
    adInstruct:el('ad-instruct'),
    adQuery:   el('ad-query'),
    adDoc:     el('ad-doc'),
    adOut:     el('ad-out'),
    adStamp:   el('ad-stamp'),
    adRow:     el('ad-row-instruct'),
    rosterSub: el('roster-sub'),
    rosterBox: el('roster'),
    rosterFoot:el('roster-foot'),
    honest:    el('honest')
  };
  if (!node.seg) return;

  /* ---------- static fields, all read from window.SS ---------- */

  var docText = String(sample.document || '');

  /* wording tracks the report's own definition of adaptive, which is about whether the
     policy is designed to be modified at inference time, not about prompt plumbing */
  put(node.fxPolicy, 'Not designed to be modified at inference time. The category vocabulary '
    + 'was settled during training.');
  put(node.fxDoc, docText);
  put(node.adSystem, String(SS.systemPrompt));
  put(node.adQuery, String(sample.query || ''));
  put(node.adDoc, docText);

  var adaptive = roster.filter(function (b) { return b.adaptive === true; });
  var fixed = roster.filter(function (b) { return b.adaptive !== true; });

  var fixedOutputs = [];
  fixed.forEach(function (b) {
    var o = String(b.output || '');
    if (o && fixedOutputs.indexOf(o) === -1) fixedOutputs.push(o);
  });
  put(node.fxOut, fixedOutputs.length
    ? fixedOutputs.join(' or ') + ', drawn from its own category list'
    : 'a label drawn from its own category list');

  var ours = roster.filter(function (b) { return /ours/i.test(String(b.model)); })[0];
  put(node.adOut, (ours && ours.output ? String(ours.output) : 'Score')
    + ', for the one yes/no question you wrote');

  /* ---------- roster, built from the report's comparison table ---------- */

  function group(title, count, rows) {
    var box = document.createElement('div');
    box.className = 'fa-group';

    var h = document.createElement('p');
    h.className = 'fa-group-head';
    var hn = document.createElement('span');
    hn.className = 'fa-group-name';
    hn.textContent = title;
    var hc = document.createElement('span');
    hc.className = 'fa-group-count';
    hc.textContent = count + ' of ' + roster.length;
    h.appendChild(hn);
    h.appendChild(hc);
    box.appendChild(h);

    var list = document.createElement('ul');
    list.className = 'fa-chips';
    rows.forEach(function (b) {
      var isOurs = /ours/i.test(String(b.model));
      var li = document.createElement('li');
      li.className = 'fa-chip' + (isOurs ? ' is-ours' : '');
      var nm = document.createElement('span');
      nm.className = 'fa-chip-n';
      nm.textContent = String(b.model).replace(/\s*\(ours\)\s*/i, '');
      var sz = document.createElement('span');
      sz.className = 'fa-chip-s';
      /* leading space so the chip reads as "LlamaGuard-4 12B" to a screen
         reader or a copy-paste, not as "LlamaGuard-412B" */
      sz.textContent = ' ' + String(b.size || '');
      li.appendChild(nm);
      li.appendChild(sz);
      if (isOurs) {
        var tag = document.createElement('span');
        tag.className = 'fa-chip-tag';
        tag.textContent = 'this report';
        li.appendChild(tag);
      }
      list.appendChild(li);
    });
    box.appendChild(list);
    return box;
  }

  if (node.rosterBox) {
    node.rosterBox.appendChild(group('Policy set at inference time', adaptive.length, adaptive));
    node.rosterBox.appendChild(group('Policy fixed at training time', fixed.length, fixed));
  }

  /* smallest adaptive model, derived from the table rather than asserted */
  function billions(s) {
    var m = /([\d.]+)\s*B/i.exec(String(s || ''));
    return m ? parseFloat(m[1]) : Infinity;
  }
  var smallest = adaptive.slice().sort(function (a, b) { return billions(a.size) - billions(b.size); })[0];
  var sub = 'Every model the report compares, split by the report’s own Adaptive column.';
  if (adaptive.length < fixed.length) sub += ' Adaptivity is the minority property here.';
  if (smallest) {
    sub += ' The smallest model that has it is '
      + String(smallest.model).replace(/\s*\(ours\)\s*/i, '') + ' at ' + String(smallest.size) + '.';
  }
  put(node.rosterSub, sub);
  put(node.rosterFoot, '“Adaptive if policy is designed to be modified at inference time.” '
    + 'Table 4 of the report.');

  var f1 = head.adaptabilityF1;
  var best = head.adaptabilityBest;
  var honest = 'No model is called on this page. The content and the question are a training sample '
    + 'quoted from Figure 2 of the report; the <Instruct> sentence is assembled from the wording of '
    + 'the report’s strictness table rather than quoted, so read this as an illustration of the '
    + 'interface, not as a measured output. No verdict is shown, because showing one would mean '
    + 'inventing it.';
  if (typeof f1 === 'number' && typeof best === 'number') {
    honest += ' Adaptive also does not mean best: on the report’s adaptability benchmark Shieldstral '
      + 'reaches ' + f1 + ' F1, behind the strongest baseline at ' + best + '.';
  }
  put(node.honest, honest);

  /* ---------- the one control ---------- */

  var buttons = [];
  var current = 0;
  var switches = 0;

  function instructFor(lv) {
    var domains = String(lv.domains || '');
    var lead = domains ? domains.charAt(0).toLowerCase() + domains.slice(1) : 'this content';
    var rationale = String(lv.rationale || '').replace(/\.$/, '');
    return 'Evaluate this exchange for ' + lead + '. Apply a '
      + String(lv.level || '').toLowerCase() + ' standard. ' + rationale + '.';
  }

  function plural(n, one, many) { return n === 1 ? one : many; }

  function render(animate) {
    var lv = levels[current];

    /* the button already shows the domains, so this line carries only the report's rationale */
    put(node.why, 'Why the report uses this level: '
      + String(lv.rationale || '').replace(/\.$/, '').toLowerCase() + '.');
    put(node.adInstruct, instructFor(lv));

    if (switches === 0) {
      put(node.fxStamp, 'Neither field above can carry the tolerance. Switch it and see.');
      put(node.adStamp, 'One field above is about to move.');
    } else {
      put(node.fxStamp, 'Both fields still byte for byte identical after ' + switches + ' '
        + plural(switches, 'switch', 'switches') + '.');
      put(node.adStamp, 'Instruct rewritten ' + switches + ' ' + plural(switches, 'time', 'times')
        + '. Same weights, same checkpoint, no retraining.');
    }

    for (var i = 0; i < buttons.length; i++) {
      var on = (i === current);
      var b = buttons[i];
      b.classList.toggle('is-on', on);
      b.setAttribute('aria-checked', on ? 'true' : 'false');
      b.setAttribute('tabindex', on ? '0' : '-1');
    }

    if (animate && node.adRow) {
      node.adRow.classList.remove('is-flash');
      /* reading a layout property restarts the CSS animation */
      if (typeof node.adRow.offsetWidth === 'number') { void node.adRow.offsetWidth; }
      node.adRow.classList.add('is-flash');
    }
  }

  function select(i, viaUser, focus) {
    if (i < 0) i = levels.length - 1;
    if (i >= levels.length) i = 0;
    var moved = (i !== current);
    current = i;
    if (viaUser && moved) switches += 1;
    render(viaUser && moved);
    if (focus && buttons[i] && buttons[i].focus) buttons[i].focus();
  }

  levels.forEach(function (lv, i) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'fa-opt';
    b.id = RID + '-opt-' + i;
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', 'false');
    b.setAttribute('tabindex', '-1');
    b.setAttribute('aria-label', 'Deployment tolerance: ' + String(lv.level));

    var nm = document.createElement('span');
    nm.className = 'fa-opt-n';
    nm.textContent = String(lv.level);
    var dm = document.createElement('span');
    dm.className = 'fa-opt-d';
    dm.textContent = String(lv.domains || '');
    b.appendChild(nm);
    b.appendChild(dm);

    b.addEventListener('click', function () { select(i, true, false); });
    b.addEventListener('keydown', function (ev) {
      var k = ev.key;
      if (k === 'ArrowRight' || k === 'ArrowDown') { ev.preventDefault(); select(current + 1, true, true); }
      else if (k === 'ArrowLeft' || k === 'ArrowUp') { ev.preventDefault(); select(current - 1, true, true); }
      else if (k === 'Home') { ev.preventDefault(); select(0, true, true); }
      else if (k === 'End') { ev.preventDefault(); select(levels.length - 1, true, true); }
    });

    buttons.push(b);
    node.seg.appendChild(b);
  });

  select(0, false, false);
})();
