(function(){
  var RID = 'w-policy';
  var root = document.getElementById(RID);
  if (!root) return;

  function bail(){ root.hidden = true; }

  var SS = (typeof window !== 'undefined') ? window.SS : null;
  if (!SS || !SS.fig3 || !SS.fig4 || !SS.systemPrompt) return bail();

  var f3 = SS.fig3;
  var f4 = SS.fig4;
  if (!f3.positive || !f3.negative || !f4.positive || !f4.negative) return bail();

  function el(suffix){ return document.getElementById(RID + '-' + suffix); }

  var node = {
    tabA:      el('tab-a'),
    tabB:      el('tab-b'),
    panel:     el('panel'),
    chooseLab: el('choose-lab'),
    hint:      el('hint'),
    seg:       el('seg'),
    vSystem:   el('v-system'),
    vInstruct: el('v-instruct'),
    vQuery:    el('v-query'),
    vDocument: el('v-document'),
    sInstruct: el('s-instruct'),
    sQuery:    el('s-query'),
    sDocument: el('s-document'),
    fInstruct: el('f-instruct'),
    fQuery:    el('f-query'),
    fDocument: el('f-document'),
    qsub:      el('qsub'),
    qsubT:     el('qsub-t'),
    dsub:      el('dsub'),
    dsubK:     el('dsub-k'),
    dsubT:     el('dsub-t'),
    verdict:   el('verdict'),
    tok:       el('tok'),
    glyph:     el('glyph'),
    flagTxt:   el('flagtxt'),
    becauseK:  el('because-k'),
    becauseT:  el('because-t'),
    caveat:    el('caveat')
  };
  for (var k in node){ if (!node[k]) return bail(); }

  var SVGNS = 'http://www.w3.org/2000/svg';

  // Role wording comes from the data where the paper supplies it, so the two
  // tabs stay consistent even if the wording in data.js is edited.
  var ROLE_T = f3.positive.role || 'target category';
  var ROLE_S = f3.negative.role || 'sibling category';

  var TABS = {
    a: {
      btn: node.tabA,
      live: 'query',
      chooseLab: 'swap the query',
      hint: 'One query names the category this document was written to violate. The other names a sibling category. The document is byte for byte the same in both.',
      becauseK: 'this query names',
      instruct: f3.instruct,
      doc: f3.document,
      qsubK: null,
      qsubT: null,
      dsubK: 'note',
      dsubT: 'Unsafe rewrite, identical under both options.',
      options: [
        {
          role: ROLE_T, chip: null,
          query: f3.positive.query, label: f3.positive.label,
          caveat: null
        },
        {
          role: ROLE_S, chip: null,
          query: f3.negative.query, label: f3.negative.label,
          caveat: 'The document never changed. It is still the unsafe rewrite. The answer is no only because this query asks about a sibling category.'
        }
      ]
    },
    b: {
      btn: node.tabB,
      live: 'document',
      chooseLab: 'swap the document',
      hint: 'Both documents are unsafe rewrites of the same safe sentence. One matches the category the query names, the other matches its sibling. The query is byte for byte the same in both.',
      becauseK: 'this document matches',
      instruct: f4.instruct,
      query: f4.query,
      qsubK: 'the category it names',
      qsubT: f4.category,
      dsubK: 'rewritten from',
      dsubT: f4.source,
      options: [
        {
          role: ROLE_T, chip: f4.category,
          doc: f4.positive.document, label: f4.positive.label,
          caveat: null
        },
        {
          role: ROLE_S, chip: f4.sibling,
          doc: f4.negative.document, label: f4.negative.label,
          caveat: 'Still unsafe content. It describes the sibling category, which is not the one this query asks about.'
        }
      ]
    }
  };

  var state = { tab: 'a', pick: { a: 0, b: 0 } };

  node.vSystem.textContent = SS.systemPrompt;

  function clear(n){ while (n.firstChild) n.removeChild(n.firstChild); }

  function setSub(pRow, kNode, tNode, kText, tText){
    if (tText){
      if (kNode && kText) kNode.textContent = kText;
      tNode.textContent = tText;
      pRow.hidden = false;
    } else {
      pRow.hidden = true;
    }
  }

  function buildSeg(cfg){
    clear(node.seg);
    var picked = state.pick[state.tab];
    for (var i = 0; i < cfg.options.length; i++){
      var opt = cfg.options[i];
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'wp-opt';
      b.id = RID + '-opt-' + state.tab + '-' + i;
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-checked', i === picked ? 'true' : 'false');
      b.tabIndex = i === picked ? 0 : -1;
      b.setAttribute('data-i', String(i));

      var tt = document.createElement('span');
      tt.className = 'wp-opt-t';
      tt.textContent = opt.role;
      b.appendChild(tt);

      if (opt.chip){
        var kk = document.createElement('span');
        kk.className = 'wp-opt-k';
        kk.textContent = opt.chip;
        b.appendChild(kk);
      }

      node.seg.appendChild(b);
    }
  }

  function setLive(which, yes){
    var map = { instruct: node.fInstruct, query: node.fQuery, document: node.fDocument };
    var states = { instruct: node.sInstruct, query: node.sQuery, document: node.sDocument };
    for (var key in map){
      var on = (key === which);
      map[key].className = 'wp-field' + (on ? (yes ? ' is-live is-yes' : ' is-live is-no') : '');
      states[key].textContent = on ? 'changes' : 'fixed';
    }
  }

  function setGlyph(yes){
    clear(node.glyph);
    var shape;
    if (yes){
      shape = document.createElementNS(SVGNS, 'path');
      shape.setAttribute('d', 'M8 1.5 14.5 8 8 14.5 1.5 8Z');
      shape.setAttribute('fill', 'currentColor');
    } else {
      shape = document.createElementNS(SVGNS, 'circle');
      shape.setAttribute('cx', '8');
      shape.setAttribute('cy', '8');
      shape.setAttribute('r', '5.7');
      shape.setAttribute('fill', 'none');
      shape.setAttribute('stroke', 'currentColor');
      shape.setAttribute('stroke-width', '1.7');
    }
    node.glyph.appendChild(shape);
  }

  function flash(){
    var target = (state.tab === 'a') ? node.fQuery : node.fDocument;
    target.classList.remove('wp-flash');
    void target.offsetWidth;
    target.classList.add('wp-flash');
    node.tok.classList.remove('wp-pop');
    void node.tok.offsetWidth;
    node.tok.classList.add('wp-pop');
  }

  function render(animate){
    var cfg = TABS[state.tab];
    var opt = cfg.options[state.pick[state.tab]];

    node.tabA.classList.toggle('is-on', state.tab === 'a');
    node.tabB.classList.toggle('is-on', state.tab === 'b');
    node.tabA.setAttribute('aria-selected', state.tab === 'a' ? 'true' : 'false');
    node.tabB.setAttribute('aria-selected', state.tab === 'b' ? 'true' : 'false');
    node.tabA.tabIndex = state.tab === 'a' ? 0 : -1;
    node.tabB.tabIndex = state.tab === 'b' ? 0 : -1;
    node.panel.setAttribute('aria-labelledby', RID + '-tab-' + state.tab);

    node.chooseLab.textContent = cfg.chooseLab;
    node.hint.textContent = cfg.hint;

    node.vInstruct.textContent = cfg.instruct;
    node.vQuery.textContent = (state.tab === 'a') ? opt.query : cfg.query;
    node.vDocument.textContent = (state.tab === 'a') ? cfg.doc : opt.doc;

    setSub(node.qsub, null, node.qsubT, cfg.qsubK, cfg.qsubT);
    setSub(node.dsub, node.dsubK, node.dsubT, cfg.dsubK, cfg.dsubT);

    var yes = String(opt.label).toLowerCase() === 'yes';
    setLive(cfg.live, yes);

    node.tok.textContent = opt.label;
    setGlyph(yes);
    node.flagTxt.textContent = yes ? 'meets the query' : 'does not meet the query';
    node.verdict.className = 'wp-verdict ' + (yes ? 'is-yes' : 'is-no');
    node.becauseK.textContent = cfg.becauseK;
    node.becauseT.textContent = opt.chip || opt.role;

    if (opt.caveat){
      node.caveat.textContent = opt.caveat;
      node.caveat.hidden = false;
    } else {
      node.caveat.textContent = '';
      node.caveat.hidden = true;
    }

    var checks = node.seg.querySelectorAll('.wp-opt');
    for (var i = 0; i < checks.length; i++){
      var on = (i === state.pick[state.tab]);
      checks[i].setAttribute('aria-checked', on ? 'true' : 'false');
      checks[i].tabIndex = on ? 0 : -1;
    }

    if (animate) flash();
  }

  function pick(i, focus){
    var cfg = TABS[state.tab];
    if (i < 0) i = cfg.options.length - 1;
    if (i >= cfg.options.length) i = 0;
    var changed = (i !== state.pick[state.tab]);
    state.pick[state.tab] = i;
    if (changed) render(true);
    if (focus){
      var b = document.getElementById(RID + '-opt-' + state.tab + '-' + i);
      if (b) b.focus();
    }
  }

  function selectTab(name, focus){
    if (!TABS[name]) return;
    state.tab = name;
    buildSeg(TABS[name]);
    render(false);
    if (focus) TABS[name].btn.focus();
  }

  function optIndex(target){
    var b = (target && target.closest) ? target.closest('.wp-opt') : null;
    if (!b) return -1;
    var v = parseInt(b.getAttribute('data-i'), 10);
    return isNaN(v) ? -1 : v;
  }

  node.seg.addEventListener('click', function(ev){
    var i = optIndex(ev.target);
    if (i >= 0) pick(i, false);
  });

  node.seg.addEventListener('keydown', function(ev){
    var cur = state.pick[state.tab];
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown'){ ev.preventDefault(); pick(cur + 1, true); }
    else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp'){ ev.preventDefault(); pick(cur - 1, true); }
    else if (ev.key === ' ' || ev.key === 'Enter'){
      var i = optIndex(ev.target);
      if (i >= 0){ ev.preventDefault(); pick(i, true); }
    }
  });

  function tabHandler(name){
    return function(){ selectTab(name, false); };
  }
  node.tabA.addEventListener('click', tabHandler('a'));
  node.tabB.addEventListener('click', tabHandler('b'));

  function tabKeys(ev){
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown' ||
        ev.key === 'ArrowLeft' || ev.key === 'ArrowUp'){
      ev.preventDefault(); selectTab(state.tab === 'a' ? 'b' : 'a', true);
    } else if (ev.key === 'Home'){
      ev.preventDefault(); selectTab('a', true);
    } else if (ev.key === 'End'){
      ev.preventDefault(); selectTab('b', true);
    }
  }
  node.tabA.addEventListener('keydown', tabKeys);
  node.tabB.addEventListener('keydown', tabKeys);

  selectTab('a', false);
})();
