(function(){
  'use strict';

  var ID = 'w-ancestors';
  var root = document.getElementById(ID);
  if (!root) return;

  var SS = (typeof window !== 'undefined' && window.SS) ? window.SS : null;
  if (!SS || !SS.evalTaxonomy || !SS.evalTaxonomy.length) return;

  function byId(suffix){ return document.getElementById(ID + '-' + suffix); }

  var elTree      = byId('tree');
  var elRows      = byId('rows');
  var elInstruct  = byId('instruct');
  var elInstrTag  = byId('instructtag');
  var elDoc       = byId('doc');
  var elDocTag    = byId('doctag');
  var elReadout   = byId('readout');
  var elStats     = byId('stats');
  var elCaption   = byId('caption');
  var elTreeMeta  = byId('treemeta');
  var elSel       = byId('selected');
  var elHonest    = byId('honest');
  var elStatus    = byId('status');
  if (!elTree || !elRows || !elInstruct || !elDoc || !elReadout || !elStats) return;

  /* ---------- small dom helpers ---------- */

  function make(tag, cls, text){
    var el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text != null) el.textContent = text;
    return el;
  }
  function clear(el){ while (el.firstChild) el.removeChild(el.firstChild); }
  function pad2(n){ return (n < 10 ? '0' : '') + n; }

  /* One evaluation category name carries a stray space from PDF text
     extraction ("V oter Suppression"). Rejoin a lone capital with the word
     that follows it, but never for the real one-letter words A and I. */
  function tidy(s){
    return String(s == null ? '' : s).replace(/\b([B-HJ-Z]) (?=[a-z]{2,})/g, '$1');
  }

  /* ---------- values pulled from the data layer ---------- */

  var head = SS.headline || {};
  var fig3 = SS.fig3 || {};              /* paper Figure 3: taxonomy-generated TRAINING pair */
  var fig4 = SS.fig4 || {};              /* paper Figure 4: evaluation pair, fallback only */
  var fig3pos = fig3.positive || {};
  var fig4pos = fig4.positive || {};

  /* the rewriting prompt that produces one unsafe rewrite per LLM call */
  var gen = null;
  var prompts = SS.genPrompts || [];
  for (var g = 0; g < prompts.length; g++){
    if (/training/i.test(String(prompts[g].name || ''))){ gen = prompts[g]; break; }
  }
  if (!gen && prompts.length) gen = prompts[0];

  function placeholder(body, field){
    var m = String(body == null ? '' : body).match(new RegExp(field + ':\\s*\\n\\[([^\\]]+)\\]'));
    return m ? m[1].trim().replace(/\s+/g, ' ') : '';
  }
  var docSlot = gen ? placeholder(gen.body, 'REWRITTEN_TEXT') : '';
  docSlot = docSlot ? '[' + docSlot + ']' : '[rewritten unsafe text]';

  /* The worked example is the paper's Figure 3, which is the taxonomy-generated
     training sample the ancestor trick applies to. Figure 4 is an evaluation
     pair and is only a fallback if fig3 is absent from the data layer. */
  var exDoc   = fig3.document || fig4pos.document || '';
  var exQuery = fig3pos.query || fig4.query || '';
  var exLabel = fig3pos.label || fig4pos.label || '';
  var exInstr = fig3.instruct || fig4.instruct || SS.systemPrompt || '';
  var exFig   = fig3.document ? 'Figure 3' : 'Figure 4';

  elInstruct.textContent = exInstr;
  if (elInstrTag) elInstrTag.textContent = 'one sampled phrasing, from ' + exFig;

  /* Count the tree we are about to draw rather than trusting the headline
     numbers, so the caption can never disagree with what is on screen. */
  var nSuper = SS.evalTaxonomy.length, nSub = 0, nLeaf = 0;
  for (var t = 0; t < SS.evalTaxonomy.length; t++){
    var tsubs = SS.evalTaxonomy[t].subs || [];
    nSub += tsubs.length;
    for (var u = 0; u < tsubs.length; u++) nLeaf += (tsubs[u].leaves || []).length;
  }

  if (elTreeMeta){
    elTreeMeta.textContent = nSuper + ' super classes / ' + nSub +
      ' subcategories / ' + nLeaf + ' leaves';
  }

  if (elCaption){
    elCaption.textContent =
      'The argument in one line: because the taxonomy is a tree, content that violates a leaf ' +
      'category also violates every ancestor up to the super class, so the same rewrite can be relabelled at ' +
      'each level. That multiplies the positive signal per rewrite by the number of levels with no extra LLM calls. ' +
      'Two caveats. The trick belongs to the training pipeline, whose taxonomy (' + head.trainSupers +
      ' super classes, ' + head.trainLeaves + ' leaves) the paper only summarises, so the tree drawn here is the ' +
      'published evaluation taxonomy (' + nSuper + ' super classes, ' + nSub +
      ' subcategories, ' + nLeaf + ' leaves), borrowed because its category names are listed in full. ' +
      'And nothing is classified live: the leaf row reproduces ' + exFig + ' of the paper' +
      ', the ancestor rows are schematic slots, and every "' + (exLabel || 'yes') +
      '" is the construction rule, not a model prediction.';
  }

  /* ---------- build the tree ---------- */

  var nodes = [];
  var groups = [];

  function addNode(parentUl, parentNode, level, name, catId, isLeaf){
    var li = make('li');
    li.setAttribute('role', 'none');

    var el = make('div', 'anc-node anc-lvl' + level);
    el.setAttribute('role', 'treeitem');
    el.setAttribute('aria-level', String(level));
    el.setAttribute('tabindex', '-1');

    if (isLeaf){
      el.appendChild(make('span', 'anc-caret-gap'));
      el.setAttribute('aria-selected', 'false');
    } else {
      var caret = make('span', 'anc-caret');
      caret.setAttribute('aria-hidden', 'true');
      el.appendChild(caret);
      el.setAttribute('aria-expanded', 'false');
    }

    var mark = make('span', 'anc-mark');
    mark.setAttribute('aria-hidden', 'true');
    el.appendChild(mark);
    el.appendChild(make('span', 'anc-name', name));
    if (catId) el.appendChild(make('span', 'anc-id', catId));

    var tag = make('span', 'anc-tag', '');
    el.appendChild(tag);
    li.appendChild(el);

    var node = {
      el: el, tag: tag, ul: parentUl, level: level, name: name, id: catId,
      leaf: isLeaf, parent: parentNode, children: [], open: false, group: null
    };

    if (!isLeaf){
      var ul = make('ul', 'anc-group');
      ul.setAttribute('role', 'group');
      li.appendChild(ul);
      node.group = ul;
      groups.push(ul);
    }

    parentUl.appendChild(li);
    if (parentNode) parentNode.children.push(node);
    nodes.push(node);

    el.addEventListener('click', function(){ activate(node, true); });
    el.addEventListener('keydown', function(ev){ onKey(ev, node); });
    return node;
  }

  var leaves = [];
  for (var a = 0; a < SS.evalTaxonomy.length; a++){
    var sc = SS.evalTaxonomy[a];
    var scNode = addNode(elTree, null, 1, tidy(sc.name), sc.id || '', false);
    var subs = sc.subs || [];
    for (var b = 0; b < subs.length; b++){
      var sub = subs[b];
      var subNode = addNode(scNode.group, scNode, 2, tidy(sub.name), '', false);
      var lvs = sub.leaves || [];
      for (var c = 0; c < lvs.length; c++){
        leaves.push(addNode(subNode.group, subNode, 3, tidy(lvs[c].name), lvs[c].id || '', true));
      }
    }
  }
  if (!leaves.length) return;

  /* Which leaf does the paper's worked example belong to? Match the leaf whose
     name the published query actually names, longest match wins. */
  var anchor = null, anchorLen = 0;
  var qLower = exQuery.toLowerCase();
  for (var e = 0; e < leaves.length; e++){
    var nm = leaves[e].name.toLowerCase();
    if (qLower && nm.length > anchorLen && qLower.indexOf(nm) >= 0){
      anchor = leaves[e]; anchorLen = nm.length;
    }
  }

  /* ---------- open / close ---------- */

  function setOpen(node, want){
    if (node.leaf || !node.group) return;
    node.open = !!want;
    node.el.setAttribute('aria-expanded', want ? 'true' : 'false');
    if (want) node.group.className = 'anc-group is-open' + (node.group.className.indexOf('is-lit') >= 0 ? ' is-lit' : '');
    else node.group.className = 'anc-group' + (node.group.className.indexOf('is-lit') >= 0 ? ' is-lit' : '');
  }

  function toggle(node, want){
    if (node.level === 1 && want){
      for (var i = 0; i < nodes.length; i++){
        if (nodes[i].level === 1 && nodes[i] !== node) setOpen(nodes[i], false);
      }
    }
    setOpen(node, want);
    if (node.level === 1 && want){
      for (var j = 0; j < node.children.length; j++) setOpen(node.children[j], true);
    }
    if (focused && !visible(focused)) setTab(node, true);
  }

  function visible(node){
    if (!node) return false;
    var p = node.parent;
    while (p){ if (!p.open) return false; p = p.parent; }
    return true;
  }
  function visibleList(){
    var out = [];
    for (var i = 0; i < nodes.length; i++) if (visible(nodes[i])) out.push(nodes[i]);
    return out;
  }

  /* ---------- roving tabindex ---------- */

  var focused = null;
  function setTab(node, doFocus){
    if (!node) return;
    if (focused && focused !== node) focused.el.setAttribute('tabindex', '-1');
    focused = node;
    node.el.setAttribute('tabindex', '0');
    if (doFocus && node.el.focus) node.el.focus();
  }

  function activate(node, doFocus){
    if (node.leaf) select(node);
    else toggle(node, !node.open);
    setTab(node, doFocus);
  }

  function onKey(ev, node){
    var k = ev.key, vis, i;
    if (k === 'ArrowDown'){
      vis = visibleList(); i = vis.indexOf(node);
      if (i >= 0 && i < vis.length - 1) setTab(vis[i + 1], true);
    } else if (k === 'ArrowUp'){
      vis = visibleList(); i = vis.indexOf(node);
      if (i > 0) setTab(vis[i - 1], true);
    } else if (k === 'ArrowRight'){
      if (!node.leaf){
        if (!node.open) toggle(node, true);
        else if (node.children.length) setTab(node.children[0], true);
      }
    } else if (k === 'ArrowLeft'){
      if (!node.leaf && node.open) toggle(node, false);
      else if (node.parent) setTab(node.parent, true);
    } else if (k === 'Home'){
      vis = visibleList(); if (vis.length) setTab(vis[0], true);
    } else if (k === 'End'){
      vis = visibleList(); if (vis.length) setTab(vis[vis.length - 1], true);
    } else if (k === 'Enter' || k === ' ' || k === 'Spacebar'){
      activate(node, true);
    } else {
      return;
    }
    if (ev.preventDefault) ev.preventDefault();
  }

  /* ---------- selection and emitted rows ---------- */

  /* Selecting a leaf stands for one rewrite, and the paper gets a rewrite plus
     its query out of a single LLM call. */
  var CALLS_PER_REWRITE = 1;
  var llmCalls = 0;
  var posRows  = 0;
  /* The paper's claim is that the ancestor rows cost nothing: "This multiplies
     the positive signal per rewrite by the number of levels without additional
     LLM calls." Kept as a counter so the reader can watch it stay put. */
  var extraCalls = 0;

  var statValues = [];
  (function buildStats(){
    var labels = ['LLM calls in this demo', 'Positive rows produced', 'Extra LLM calls'];
    for (var i = 0; i < labels.length; i++){
      var wrap = make('div', 'anc-stat');
      wrap.appendChild(make('dt', null, labels[i]));
      var dd = make('dd', null, '0');
      wrap.appendChild(dd);
      elStats.appendChild(wrap);
      statValues.push(dd);
    }
  })();

  function levelName(level){
    if (level === 3) return 'Leaf';
    if (level === 2) return 'Subcategory';
    return 'Super class';
  }

  function clearLit(){
    for (var i = 0; i < nodes.length; i++){
      var n = nodes[i];
      n.el.className = 'anc-node anc-lvl' + n.level;
      n.tag.textContent = '';
      if (n.leaf) n.el.setAttribute('aria-selected', 'false');
    }
    for (var j = 0; j < groups.length; j++){
      groups[j].className = 'anc-group' + (groups[j].className.indexOf('is-open') >= 0 ? ' is-open' : '');
    }
  }

  function select(leaf){
    var path = [];
    var n = leaf;
    while (n){ path.push(n); n = n.parent; }   /* leaf, subcategory, super class */

    clearLit();
    for (var i = 0; i < path.length; i++){
      var node = path[i];
      node.el.className = 'anc-node anc-lvl' + node.level + ' is-lit';
      node.tag.textContent = pad2(i + 1);
      if (node.leaf) node.el.setAttribute('aria-selected', 'true');
      if (node.ul && node.ul !== elTree){
        node.ul.className = 'anc-group' + (node.ul.className.indexOf('is-open') >= 0 ? ' is-open' : '') + ' is-lit';
      }
    }

    var isExample = (anchor && leaf === anchor && exDoc);
    elDoc.className = isExample ? 'anc-fval' : 'anc-fval is-slot';
    elDoc.textContent = isExample ? exDoc : docSlot;
    if (elDocTag) elDocTag.textContent = isExample ? (exFig + ', verbatim') : 'schematic slot';
    if (elSel) elSel.textContent = 'Leaf: ' + leaf.name + (leaf.id ? ' (' + leaf.id + ')' : '');

    if (elHonest){
      elHonest.textContent = isExample
        ? ('Not a live model. The document and the leaf query are quoted from ' + exFig +
           ' of the paper. The two ancestor queries are slots: the paper says a query phrasing is sampled for ' +
           'each ancestor category but does not publish those phrasings.')
        : ('Not a live model, and the paper publishes no worked example for this leaf, so the document and all ' +
           'three queries are schematic slots. Pick ' + (anchor ? anchor.name : 'the worked example') +
           ' to read the real text from ' + exFig + '.');
    }

    clear(elRows);
    var made = [];
    for (var r = 0; r < path.length; r++){
      var p = path[r];
      var target = (r === 0);
      var li = make('li', 'anc-row');

      var top = make('div', 'anc-rowtop');
      top.appendChild(make('span', 'anc-rowno', pad2(r + 1)));
      top.appendChild(make('span', 'anc-rowlvl', levelName(p.level)));
      top.appendChild(make('span', 'anc-rowcat', p.name));
      if (exLabel) top.appendChild(make('span', 'anc-yes', exLabel));
      li.appendChild(top);

      /* Provenance differs by level and the paper is explicit about it: the
         target-category query comes out of the same LLM call, ancestor query
         phrasings are sampled per category. */
      var lab = make('span', 'anc-qlab', target
        ? 'Query written by the same LLM call'
        : 'Query phrasing sampled for this category');
      li.appendChild(lab);

      var verbatim = target && isExample && exQuery;
      var q = make('div', verbatim ? 'anc-query' : 'anc-query is-slot',
        verbatim ? exQuery : '[yes/no query about "' + p.name + '"]');
      li.appendChild(q);
      li.appendChild(make('span', 'anc-src', verbatim ? (exFig + ', verbatim') : 'schematic slot'));

      elRows.appendChild(li);
      made.push(li);
    }

    function reveal(){
      for (var m = 0; m < made.length; m++){
        made[m].style.transitionDelay = (m * 70) + 'ms';
        made[m].className = 'anc-row is-in';
      }
    }
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(reveal);
    else reveal();

    llmCalls += CALLS_PER_REWRITE;
    posRows += path.length;
    statValues[0].textContent = String(llmCalls);
    statValues[1].textContent = String(posRows);
    statValues[2].textContent = String(extraCalls);

    clear(elReadout);
    elReadout.appendChild(make('span', 'anc-big', String(CALLS_PER_REWRITE)));
    elReadout.appendChild(make('span', null, 'LLM call'));
    elReadout.appendChild(make('span', 'anc-arrow', '→'));
    elReadout.appendChild(make('span', 'anc-big', String(path.length)));
    elReadout.appendChild(make('span', null, 'positive rows, one per level, and'));
    elReadout.appendChild(make('span', 'anc-big', String(extraCalls)));
    elReadout.appendChild(make('span', null, 'extra LLM calls'));

    if (elStatus){
      elStatus.textContent = path.length + ' positive rows for ' + leaf.name +
        ', labelled ' + (exLabel || 'yes') + ' at leaf, subcategory and super class level, from one LLM call.';
    }
  }

  /* ---------- initial state ---------- */

  var start = anchor || leaves[0];
  var up = start.parent;
  while (up){
    if (up.level === 1) toggle(up, true); else setOpen(up, true);
    up = up.parent;
  }
  select(start);
  setTab(start, false);

})();
