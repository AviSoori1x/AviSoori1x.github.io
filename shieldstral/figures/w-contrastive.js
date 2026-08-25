(function () {
  'use strict';

  var ID = 'w-contrastive';
  var root = document.getElementById(ID);
  if (!root) return;

  var S = (typeof window !== 'undefined') && window.SS;
  if (!S || !S.fig3) return;

  var F = S.fig3;
  var POS = F.positive || {};
  var NEG = F.negative || {};
  var H = S.headline || {};

  function el(sfx) { return document.getElementById(ID + '-' + sfx); }
  function put(sfx, txt) {
    var n = el(sfx);
    if (n && txt !== null && txt !== undefined && txt !== '') n.textContent = String(txt);
    return n;
  }
  function tagFor(key) { return '<' + key.charAt(0).toUpperCase() + key.slice(1) + '>'; }

  /* ---------- prompt template: pull placeholders and output field names ---------- */
  var P0 = (S.genPrompts && S.genPrompts[0]) || null;
  var body = (P0 && P0.body) ? P0.body : '';
  var toks = body.match(/\{[a-z_]+\}/g) || [];
  var outs = (body.match(/^[A-Z][A-Z_]{2,}:$/gm) || []).map(function (s) {
    return s.replace(/:$/, '');
  });
  function tok(name) {
    var t = '{' + name + '}';
    return toks.indexOf(t) >= 0 ? t : '';
  }
  /* Fallbacks are the prompt's own field names, never the training-format tags,
     so a label can never claim a returned field is something it is not. */
  function outName(i, fallback) {
    return outs.length > i ? outs[i] : fallback;
  }

  /* ---------- step 01: the template slots ---------- */
  put('slot-1-lab', 'safe source text');
  put('slot-1-tok', tok('text'));
  put('slot-1-gloss', 'A benign passage. The paper does not print the one used for this example.');

  var desc = tok('description');
  put('slot-2-lab', POS.role);
  put('slot-2-tok', [tok('target_category'), desc].filter(Boolean).join('  '));
  put('slot-2-gloss', 'The category the rewrite must exhibit. Its query becomes the positive row.');

  put('slot-3-lab', NEG.role);
  put('slot-3-tok', [tok('negative_category'), desc].filter(Boolean).join('  '));
  put('slot-3-gloss', 'A related category the rewrite must avoid. Its query becomes the negative row.');

  var temp = (H.genTemp !== undefined && H.genTemp !== null) ? H.genTemp : null;
  put('temp', temp);
  put('temp2', temp);

  /* ---------- step 02: what the call returns ---------- */
  var docField = outName(0, 'REWRITTEN_TEXT');
  var posField = outName(1, 'POSITIVE_QUERY');
  var negField = outName(2, 'NEGATIVE_QUERY');

  put('ret-doc-lab', docField);
  put('ret-doc-val', F.document);
  put('ret-doc-goes', 'becomes ' + tagFor('document') + ' in both rows');
  put('ret-pos-lab', posField);
  put('ret-pos-val', POS.query);
  put('ret-pos-goes', 'becomes ' + tagFor('query') + ' in the positive row');
  put('ret-neg-lab', negField);
  put('ret-neg-val', NEG.query);
  put('ret-neg-goes', 'becomes ' + tagFor('query') + ' in the negative row');

  /* ---------- step 03: the two training rows ---------- */
  put('role-pos', POS.role);
  put('role-neg', NEG.role);
  put('tag-i-p', tagFor('instruct'));
  put('tag-i-n', tagFor('instruct'));
  put('tag-q-p', tagFor('query'));
  put('tag-q-n', tagFor('query'));
  put('tag-d-p', tagFor('document'));
  put('tag-d-n', tagFor('document'));
  put('instruct-p', F.instruct);
  put('instruct-n', F.instruct);
  put('query-p', POS.query);
  put('query-n', NEG.query);
  put('doc-p-txt', F.document);
  put('doc-n-txt', F.document);
  put('ans-p', POS.label);
  put('ans-n', NEG.label);

  /* ---------- scale line, every figure read from SS ---------- */
  var scaleBits = [];
  if (H.syntheticText) scaleBits.push('about ' + H.syntheticText + 'M training samples');
  if (H.trainSupers && H.trainLeaves) {
    scaleBits.push('drawn from a generation taxonomy of ' + H.trainSupers +
      ' super classes and ' + H.trainLeaves + ' leaf categories');
  }
  if (scaleBits.length) {
    put('scale', 'The paper reports this loop produced ' + scaleBits.join(', ') + '.');
  }

  /* ---------- disclosure: the real generation prompt ---------- */
  put('pname', P0 && P0.name);
  put('sys', S.genSystem);

  var pre = el('pre');
  if (pre && body) {
    pre.textContent = '';
    body.split(/(\{[a-z_]+\})/).forEach(function (part) {
      if (!part) return;
      if (/^\{[a-z_]+\}$/.test(part)) {
        var s = document.createElement('span');
        s.className = 'wc-ph';
        s.textContent = part;
        pre.appendChild(s);
      } else {
        pre.appendChild(document.createTextNode(part));
      }
    });
  }

  var disc = el('disc');
  var signEl = el('sign');
  var sumTxt = el('sumtext');
  if (disc) {
    disc.addEventListener('toggle', function () {
      var open = disc.open;
      if (signEl) signEl.textContent = open ? '-' : '+';
      if (sumTxt) sumTxt.textContent = (open ? 'Hide' : 'Show') + ' the generation prompt';
    });
  }

  /* ---------- step spotlight ---------- */
  var chips = [el('chip-1'), el('chip-2'), el('chip-3')];
  var steps = [el('step-1'), el('step-2'), el('step-3')];
  var runBtn = el('run');
  var status = el('status');
  var live = 0;
  var timers = [];

  function say(msg) { if (status) status.textContent = msg; }

  function stop() {
    while (timers.length) clearTimeout(timers.pop());
  }

  function setLive(n, announce) {
    live = n;
    root.classList.toggle('is-focused', n > 0);
    for (var i = 0; i < 3; i++) {
      var on = (i + 1 === n);
      if (chips[i]) {
        chips[i].setAttribute('aria-pressed', on ? 'true' : 'false');
        chips[i].classList.toggle('on', on);
      }
      if (steps[i]) steps[i].classList.toggle('is-live', on);
    }
    if (announce !== false) {
      say(n ? ('Step 0' + n + ' highlighted') : 'All three steps shown');
    }
  }

  chips.forEach(function (c, i) {
    if (!c) return;
    c.addEventListener('click', function () {
      stop();
      setLive(live === i + 1 ? 0 : i + 1);
    });
  });

  if (runBtn) {
    runBtn.addEventListener('click', function () {
      stop();
      setLive(1);
      timers.push(setTimeout(function () { setLive(2); }, 950));
      timers.push(setTimeout(function () { setLive(3); }, 1900));
      runBtn.textContent = 'Replay the loop';
    });
  }

  /* ---------- cross-step trace: returned field to training cell ---------- */
  var HL = { doc: 'is-hl-doc', pos: 'is-hl-pos', neg: 'is-hl-neg' };
  var TRACE = {
    doc: 'Tracing the rewritten text into the document cell of both rows',
    pos: 'Tracing the positive query into the query cell of the positive row',
    neg: 'Tracing the negative query into the query cell of the negative row'
  };
  var pinned = null;
  var hovered = null;
  var buttons = {};

  /* At most one trace is ever lit. A hover previews and temporarily overrides
     the pin, so the reader never sees two "traced" markers at once. */
  function paint() {
    var active = hovered || pinned;
    Object.keys(HL).forEach(function (k) {
      root.classList.toggle(HL[k], active === k);
    });
    Object.keys(buttons).forEach(function (k) {
      buttons[k].setAttribute('aria-pressed', pinned === k ? 'true' : 'false');
    });
  }

  Array.prototype.forEach.call(root.querySelectorAll('[data-hl]'), function (node) {
    var key = node.getAttribute('data-hl');
    if (!HL[key]) return;
    if (node.localName === 'button') buttons[key] = node;

    node.addEventListener('mouseenter', function () { hovered = key; paint(); });
    node.addEventListener('mouseleave', function () { if (hovered === key) hovered = null; paint(); });
    node.addEventListener('focus', function () { hovered = key; paint(); });
    node.addEventListener('blur', function () { if (hovered === key) hovered = null; paint(); });

    if (node.localName === 'button') {
      node.addEventListener('click', function () {
        pinned = (pinned === key) ? null : key;
        paint();
        say(pinned ? TRACE[key] : 'Trace cleared');
      });
    }
  });

  setLive(0, false);
  say('All three steps shown');
})();
