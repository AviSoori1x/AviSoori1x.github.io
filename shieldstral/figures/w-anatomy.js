(function(){
  var RID = 'w-anatomy';
  var root = document.getElementById(RID);
  if (!root) return;

  var SS = (typeof window !== 'undefined') ? window.SS : null;
  if (!SS) return;

  var strictness = SS.strictness;
  var formats    = SS.formats;
  var tips       = SS.tips;
  var fig2       = SS.fig2;
  var fig4       = SS.fig4;
  if (!strictness || !strictness.length) return;
  if (!formats || !formats.length) return;
  if (!tips || !tips.length) return;
  if (!fig2 || !fig2.length || !fig4) return;
  if (!SS.systemPrompt || !fig4.query || !fig4.instruct) return;

  function el(suffix){ return document.getElementById(RID + '-' + suffix); }

  var node = {
    segStrict: el('seg-strict'),
    segQuery:  el('seg-query'),
    segFormat: el('seg-format'),
    sys:       el('sys'),
    req:       el('req'),
    copy:      el('copy'),
    copied:    el('copied'),
    tips:      el('tips')
  };
  for (var k in node){ if (!node[k]) return; }

  /* ---------- source material, all of it read out of SS ---------- */

  /* the report's prompt-response example, split back into its two halves
     so it can be re-delimited in any of the Table 2 families */
  var example = null;
  for (var i = 0; i < fig2.length; i++){
    var d = fig2[i] && fig2[i].document ? String(fig2[i].document) : '';
    if (d.indexOf('[User]') >= 0 && d.indexOf('[Assistant]') >= 0 && fig2[i].query){
      example = fig2[i];
      break;
    }
  }
  if (!example) return;

  var split = String(example.document).match(/\[User\]\s*([\s\S]*?)\s*\[Assistant\]\s*([\s\S]*)$/);
  var PROMPT   = split ? split[1] : String(example.document);
  var RESPONSE = split ? split[2] : '';

  function tipIndex(re){
    for (var j = 0; j < tips.length; j++){
      if (tips[j] && re.test(String(tips[j].t))) return j;
    }
    return -1;
  }

  var broadTip = tipIndex(/many policies|at once/i);

  function sentence(s){
    s = String(s).trim();
    return /[.!?]$/.test(s) ? s : s + '.';
  }

  /* The three query task types, with the report's own verbatim example queries.
     Previously the refusal option showed a category query, which was wrong. */
  var TIP_FOR = [tipIndex(/instruct/i), broadTip, tipIndex(/format/i)];
  var QUERIES = (SS.queryTypes || []).map(function (qt, qi) {
    return {
      name: qt.name,
      sub: qt.sub,
      query: qt.examples[0],
      /* Only the category-specific type adds anything to <Instruct>, and what it
         adds is the candidate-category list the model card suggests putting there.
         Descriptive text about the template taxonomy is metatext and must never
         end up inside the prompt. */
      extra: (qi === 0 && fig4.category && fig4.sibling)
        ? 'Candidate categories: ' + fig4.category + '; ' + fig4.sibling + '.'
        : '',
      tip: TIP_FOR[qi] === undefined ? -1 : TIP_FOR[qi]
    };
  });
  if (!QUERIES.length) return;

  var state = {
    strict: 0,
    query:  QUERIES.length - 1,
    format: 0,
    text:   ''
  };
  for (i = 0; i < formats.length; i++){
    if (/bracket/i.test(String(formats[i].family))) { state.format = i; break; }
  }

  /* ---------- radio rows ---------- */

  function buildSeg(container, items, key){
    container.innerHTML = '';
    for (var n = 0; n < items.length; n++){
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'wa-opt';
      b.setAttribute('role', 'radio');
      b.setAttribute('data-i', String(n));
      var t = document.createElement('span');
      t.className = 'wa-opt-t';
      t.textContent = items[n].main;
      b.appendChild(t);
      if (items[n].sub){
        var s = document.createElement('span');
        s.className = 'wa-opt-s';
        s.textContent = items[n].sub;
        b.appendChild(s);
      }
      container.appendChild(b);
    }
    container.addEventListener('click', function(ev){
      var btn = ev.target && ev.target.closest ? ev.target.closest('.wa-opt') : null;
      if (!btn) return;
      pick(key, parseInt(btn.getAttribute('data-i'), 10) || 0, false);
    });
    container.addEventListener('keydown', function(ev){
      var count = container.children.length;
      var cur = state[key];
      if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown'){
        ev.preventDefault(); pick(key, (cur + 1) % count, true);
      } else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp'){
        ev.preventDefault(); pick(key, (cur - 1 + count) % count, true);
      } else if (ev.key === 'Home'){
        ev.preventDefault(); pick(key, 0, true);
      } else if (ev.key === 'End'){
        ev.preventDefault(); pick(key, count - 1, true);
      }
    });
  }

  function paintSeg(container, sel){
    var kids = container.children;
    for (var n = 0; n < kids.length; n++){
      var on = (n === sel);
      kids[n].setAttribute('aria-checked', on ? 'true' : 'false');
      kids[n].tabIndex = on ? 0 : -1;
    }
  }

  function pick(key, idx, focus){
    if (state[key] === idx && !focus) return;
    state[key] = idx;
    render(key);
    var container = key === 'strict' ? node.segStrict : (key === 'query' ? node.segQuery : node.segFormat);
    if (focus && container.children[idx]) container.children[idx].focus();
  }

  /* ---------- assembly ---------- */

  function txt(s){ return document.createTextNode(s); }

  function span(cls, s){
    var e = document.createElement('span');
    e.className = cls;
    e.textContent = s;
    return e;
  }

  function fill(tpl, p, r){
    return String(tpl)
      .replace(/\{prompt\}/g, function(){ return p; })
      .replace(/\{response\}/g, function(){ return r; });
  }

  function render(changed){
    var tier = strictness[state.strict];
    var q    = QUERIES[state.query];
    var fmt  = formats[state.format];

    var base    = sentence(fig4.instruct);
    var phrase  = 'Apply a ' + String(tier.level).toLowerCase() + ' standard. ' + sentence(tier.rationale);
    var instruct = base + ' ' + phrase + (q.extra ? ' ' + q.extra : '');
    var document_ = fill(fmt.tpl, PROMPT, RESPONSE);

    state.text = '<Instruct>: ' + instruct + '\n' +
                 '<Query>: ' + q.query + '\n' +
                 '<Document>: ' + document_;

    var flashS = (changed === 'strict') ? ' is-flash' : '';
    var flashQ = (changed === 'query')  ? ' is-flash' : '';
    var flashF = (changed === 'format') ? ' is-flash' : '';

    var frag = document.createDocumentFragment();

    frag.appendChild(span('wa-tag', '<Instruct>:'));
    frag.appendChild(txt(' ' + base + ' '));
    frag.appendChild(span('wa-sw' + flashS, phrase));
    if (q.extra){
      frag.appendChild(txt(' '));
      frag.appendChild(span('wa-sw' + flashQ, q.extra));
    }
    frag.appendChild(txt('\n'));

    frag.appendChild(span('wa-tag', '<Query>:'));
    frag.appendChild(txt(' '));
    frag.appendChild(span('wa-sw' + flashQ, q.query));
    frag.appendChild(txt('\n'));

    frag.appendChild(span('wa-tag', '<Document>:'));
    frag.appendChild(txt(' '));
    var parts = String(fmt.tpl).split(/(\{prompt\}|\{response\})/);
    for (var n = 0; n < parts.length; n++){
      if (parts[n] === '') continue;
      if (parts[n] === '{prompt}') frag.appendChild(txt(PROMPT));
      else if (parts[n] === '{response}') frag.appendChild(txt(RESPONSE));
      else frag.appendChild(span('wa-sw wa-delim' + flashF, parts[n]));
    }

    node.req.innerHTML = '';
    node.req.appendChild(frag);

    paintSeg(node.segStrict, state.strict);
    paintSeg(node.segQuery, state.query);
    paintSeg(node.segFormat, state.format);
    paintTips(q.tip);
  }

  /* ---------- tips ---------- */

  var tipRows = [];
  function buildTips(){
    node.tips.innerHTML = '';
    for (var n = 0; n < tips.length; n++){
      var row = document.createElement('div');
      row.className = 'wa-tip';
      var dt = document.createElement('dt');
      dt.textContent = tips[n].t;
      var mark = document.createElement('span');
      mark.className = 'wa-tip-m';
      mark.textContent = 'applies here';
      mark.hidden = true;
      dt.appendChild(mark);
      var dd = document.createElement('dd');
      dd.textContent = tips[n].d;
      row.appendChild(dt);
      row.appendChild(dd);
      node.tips.appendChild(row);
      tipRows.push({ row: row, mark: mark });
    }
  }

  function paintTips(active){
    for (var n = 0; n < tipRows.length; n++){
      var on = (n === active);
      if (on) tipRows[n].row.className = 'wa-tip is-on';
      else tipRows[n].row.className = 'wa-tip';
      tipRows[n].mark.hidden = !on;
    }
  }

  /* ---------- copy ---------- */

  var copyTimer = null;
  function say(msg, fail){
    node.copied.textContent = msg;
    node.copied.className = fail ? 'wa-copied is-fail' : 'wa-copied';
    if (copyTimer) clearTimeout(copyTimer);
    copyTimer = setTimeout(function(){
      node.copied.textContent = '';
      node.copied.className = 'wa-copied';
    }, 2600);
  }

  function legacyCopy(text){
    var ok = false;
    try {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.top = '0';
      ta.style.left = '-9999px';
      root.appendChild(ta);
      ta.select();
      ok = !!document.execCommand('copy');
      root.removeChild(ta);
    } catch (e) { ok = false; }
    say(ok ? 'copied' : 'copy blocked, select the text', !ok);
  }

  node.copy.addEventListener('click', function(){
    var text = state.text;
    if (!navigator.clipboard || !navigator.clipboard.writeText){
      legacyCopy(text);
      return;
    }
    /* some contexts leave the clipboard promise pending forever, so the
       reader always gets an answer from the guard if it does not settle */
    var settled = false;
    var guard = setTimeout(function(){
      if (settled) return;
      settled = true;
      legacyCopy(text);
    }, 1200);
    navigator.clipboard.writeText(text).then(function(){
      if (settled) return;
      settled = true;
      clearTimeout(guard);
      say('copied', false);
    }, function(){
      if (settled) return;
      settled = true;
      clearTimeout(guard);
      legacyCopy(text);
    });
  });

  /* ---------- boot ---------- */

  var strictItems = [];
  for (i = 0; i < strictness.length; i++){
    strictItems.push({ main: strictness[i].level, sub: strictness[i].domains });
  }
  var queryItems = [];
  for (i = 0; i < QUERIES.length; i++){
    queryItems.push({ main: QUERIES[i].name, sub: QUERIES[i].sub });
  }
  var formatItems = [];
  for (i = 0; i < formats.length; i++){
    formatItems.push({ main: formats[i].family, sub: '' });
  }

  buildSeg(node.segStrict, strictItems, 'strict');
  buildSeg(node.segQuery, queryItems, 'query');
  buildSeg(node.segFormat, formatItems, 'format');
  buildTips();

  node.sys.textContent = SS.systemPrompt;
  render(null);
})();
