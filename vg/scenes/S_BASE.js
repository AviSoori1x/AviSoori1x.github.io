window.SCENES = window.SCENES || {};

/* Act III, scene 22. The base checkpoint.
   One picture of the whole stack: three document shapes on top, a tokenizer
   and a Pixtral vision encoder underneath, both writing into a single token
   sequence, one 3B checkpoint, one output token that can only be yes or no.
   The claim the figure is making is that the interface never changes when the
   modality does, so the request fields stay in place and only the Document
   segment of the sequence changes colour.
   Parameter count, both F1 numbers, the declared input and output, the field
   names, the document strings and the image benchmark names are all read from
   window.SS at runtime. Nothing here is a live model call. */
window.SCENES['S_BASE'] = function (root, api) {
  var SS = api.SS || {};
  var head = SS.headline || {};
  var fig2 = SS.fig2 || [];
  var inv = SS.benchInventory || [];
  var formats = SS.formats || [];

  root.classList.add('sc-s_base');

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function commas(n) {
    return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  /* ---------- what SS says about the interface ---------- */

  /* the report's own row of the baseline table carries the declared input
     modality and output type for this checkpoint */
  var mine = null;
  (SS.baselines || []).forEach(function (b) {
    if (/shieldstral/i.test(String(b.model || ''))) mine = b;
  });

  var kindText = (fig2[0] && fig2[0].kind) || 'Text-only';
  var kindMM = (fig2[1] && fig2[1].kind) || 'Multimodal';
  var docMM = String((fig2[1] && fig2[1].document) || '[image]');

  /* the image placeholder, taken off the front of the multimodal document */
  var imgTok = (docMM.match(/^\s*(\[[^\]]+\])/) || [null, '[image]'])[1];

  /* one text document line, the bracketed family from the format table */
  var brk = null;
  formats.forEach(function (f) { if (/bracket/i.test(String(f.family || ''))) brk = f; });
  if (!brk) brk = formats[1] || formats[0] || { tpl: '[User] {prompt}\n[Assistant] {response}' };
  var docText = String(brk.tpl || '').split('\n').join('  ');

  /* the image-only splits, straight out of the benchmark inventory */
  var imgSets = [];
  inv.forEach(function (r) {
    if (String(r[1] || '').toLowerCase() === 'image') {
      imgSets.push(String(r[0]) + ' ' + commas(r[3]));
    }
  });

  /* the sentence that pins the output to two words */
  var sysP = String(SS.systemPrompt || '');
  var onlyLine = (sysP.match(/Note that[\s\S]*$/) || [''])[0] || sysP;

  /* The three request fields, named as the system prompt itself names them.
     Sentence openers are lowercased first so only the field words are left
     capitalised, then they are put back into request order. */
  var caps = sysP
    .replace(/(^|\.\s+)([A-Z])/g, function (m, a, b) { return a + b.toLowerCase(); })
    .match(/\b[A-Z][a-z]+\b/g) || [];
  function field(re, dflt) {
    for (var k = 0; k < caps.length; k++) if (re.test(caps[k])) return caps[k];
    return dflt;
  }
  var FIELDS = [
    field(/^instr/i, 'Instruction'),
    field(/^quer/i, 'Query'),
    field(/^doc/i, 'Document')
  ];

  var PARAMS = head.params != null ? String(head.params) : 'n/a';

  /* ---------- the three input shapes ---------- */
  /* Cell counts below are illustrative. The report publishes no token counts,
     so the strip fixes the request fields and varies only the document. */
  var N_INS = 12, N_QRY = 8, N_DOC = 28;

  var SHAPES = [
    {
      key: 'text', name: 'text only', img: false, txt: true,
      doc: docText,
      src: 'SS.fig2 ' + kindText + ', SS.formats ' + String(brk.family || ''),
      patch: 0, tok: N_DOC,
      say: 'a text document, no image'
    },
    {
      key: 'image', name: 'image only', img: true, txt: false,
      doc: imgTok,
      src: imgSets.length ? 'SS.benchInventory image splits, ' + imgSets.join(', ')
        : 'SS.benchInventory',
      patch: N_DOC, tok: 0,
      say: 'an image on its own, no document text'
    },
    {
      key: 'both', name: 'image + text', img: true, txt: true,
      doc: docMM,
      src: 'SS.fig2 ' + kindMM,
      patch: 16, tok: N_DOC - 16,
      say: 'an image followed by text'
    }
  ];

  /* ---------- glyphs ---------- */
  function glyph(kind) {
    var s = '<svg class="gly" viewBox="0 0 24 24" aria-hidden="true">';
    if (kind === 'text' || kind === 'both') {
      var ox = kind === 'both' ? 11 : 2;
      var w = kind === 'both' ? 11 : 20;
      s += '<rect class="ln" x="' + ox + '" y="6" width="' + w + '" height="2.4" rx="1.2"/>'
        + '<rect class="ln" x="' + ox + '" y="11" width="' + w + '" height="2.4" rx="1.2"/>'
        + '<rect class="ln" x="' + ox + '" y="16" width="' + (w * 0.6).toFixed(1)
        + '" height="2.4" rx="1.2"/>';
    }
    if (kind === 'image' || kind === 'both') {
      var iw = kind === 'both' ? 9 : 20;
      s += '<rect class="fr" x="2" y="4" width="' + iw + '" height="16" rx="2.5"/>'
        + '<circle class="dt" cx="' + (kind === 'both' ? 5 : 7.5) + '" cy="9" r="1.5"/>'
        + '<path class="pk" d="M3.4 18 L' + (kind === 'both' ? 6.5 : 10)
        + ' 12.5 L' + (iw + 1) + ' 18"/>';
    }
    return s + '</svg>';
  }

  /* One wire, routed like a bus: straight down, across, straight down again.
     The connector layers are stretched to the full width, so the corner
     rounding is given in the layer's own units and the strokes are pinned to a
     constant width. */
  function bus(x0, x1, h, lane) {
    var my = h * (lane == null ? 0.5 : lane), k = 2.4, r = h * 0.15;
    var dir = x1 > x0 ? 1 : -1;
    if (Math.abs(x1 - x0) < 0.4) return 'M ' + x0 + ' 0 L ' + x0 + ' ' + h;
    return 'M ' + x0.toFixed(2) + ' 0'
      + ' L ' + x0.toFixed(2) + ' ' + (my - r).toFixed(2)
      + ' Q ' + x0.toFixed(2) + ' ' + my + ' ' + (x0 + dir * k).toFixed(2) + ' ' + my
      + ' L ' + (x1 - dir * k).toFixed(2) + ' ' + my
      + ' Q ' + x1.toFixed(2) + ' ' + my + ' ' + x1.toFixed(2) + ' ' + (my + r).toFixed(2)
      + ' L ' + x1.toFixed(2) + ' ' + h;
  }

  /* ---------- markup ---------- */
  var cards = '';
  SHAPES.forEach(function (s, i) {
    cards += '<button type="button" class="shc" role="radio" aria-checked="false"'
      + ' tabindex="-1" id="S_BASE-shc-' + i + '">'
      + '<span class="shtop">' + glyph(s.key)
      +   '<span class="shn">' + esc(s.name) + '</span></span>'
      + '<span class="shd">' + esc(s.doc) + '</span>'
      + '<span class="shs">' + esc(s.src) + '</span>'
      + '</button>';
  });

  function stub(x) {
    return '<path class="st" d="M ' + x + ' 0 L ' + x + ' 9" '
      + 'vector-effect="non-scaling-stroke"/>';
  }

  function seg(name, n, cls, inner, id) {
    return '<div class="sg ' + cls + '" style="flex-grow:' + n + '">'
      + '<div class="cells"' + (id ? ' id="' + id + '"' : '') + '>' + inner + '</div>'
      + '<span class="sgl">&lt;' + esc(name) + '&gt;'
      + '<i class="sgn"' + (id ? ' id="' + id + '-n"' : '') + '>' + n + '</i></span>'
      + '</div>';
  }
  function cells(n, cls) {
    var s = '', i;
    for (i = 0; i < n; i++) s += '<i class="c ' + cls + '"></i>';
    return s;
  }

  var stats = '';
  if (head.textF1 != null) {
    stats += '<span class="stat"><b>' + api.num(head.textF1, 1) + '</b>'
      + '<em>text F1</em></span>';
  }
  if (head.multimodalF1 != null) {
    stats += '<span class="stat"><b>' + api.num(head.multimodalF1, 1) + '</b>'
      + '<em>multimodal F1</em></span>';
  }

  var mtags = '';
  if (mine && mine.input) {
    mtags += '<span class="mtag">input <b>' + esc(mine.input) + '</b></span>';
  }
  if (mine && mine.output) {
    mtags += '<span class="mtag">output <b>' + esc(mine.output) + '</b></span>';
  }
  mtags += '<span class="mtag">SS.headline.params</span>';

  root.appendChild(api.frag(
    '<div class="wrap">'

    + '<div class="hd">'
    +   '<span class="eyebrow">one checkpoint, three input shapes</span>'
    +   '<span class="hnote" id="S_BASE-hnote">The request fields never move. Only the '
    +   '&lt;' + esc(FIELDS[2]) + '&gt; segment of the sequence changes.</span>'
    + '</div>'

    + '<div class="shp" role="radiogroup" aria-label="input shape" id="S_BASE-shp">'
    +   cards
    + '</div>'

    + '<svg class="fan" viewBox="0 0 100 34" preserveAspectRatio="none" aria-hidden="true">'
    +   stub(16.667) + stub(50) + stub(83.333)
    +   '<path class="lk" id="S_BASE-lk-tok" d="" vector-effect="non-scaling-stroke"/>'
    +   '<path class="lk vis" id="S_BASE-lk-vis" d="" vector-effect="non-scaling-stroke"/>'
    +   '<path class="pl" id="S_BASE-pl-tok" d="" vector-effect="non-scaling-stroke"/>'
    +   '<path class="pl vis" id="S_BASE-pl-vis" d="" vector-effect="non-scaling-stroke"/>'
    + '</svg>'

    + '<div class="enc">'
    +   '<div class="ebox tok on" id="S_BASE-etok">'
    +     '<span class="ename">tokenizer and embeddings</span>'
    +     '<span class="esub">&lt;' + esc(FIELDS[0]) + '&gt; and &lt;' + esc(FIELDS[1])
    +       '&gt; are always text</span>'
    +     '<span class="etag"><i class="dot" aria-hidden="true"></i>always engaged</span>'
    +   '</div>'
    +   '<div class="ebox vis" id="S_BASE-evis">'
    +     '<span class="ename">Pixtral vision encoder</span>'
    +     '<span class="esub">image to patch tokens, in the same stream</span>'
    +     '<span class="etag"><i class="dot" aria-hidden="true"></i>'
    +       '<b id="S_BASE-vtag">idle</b></span>'
    +   '</div>'
    + '</div>'

    + '<svg class="fan2" viewBox="0 0 100 22" preserveAspectRatio="none" aria-hidden="true">'
    +   '<path class="lk" d="' + bus(25, 50, 22) + '" vector-effect="non-scaling-stroke"/>'
    +   '<path class="lk vis" id="S_BASE-lk2-vis" d="' + bus(75, 50, 22) + '"'
    +     ' vector-effect="non-scaling-stroke"/>'
    + '</svg>'

    + '<div class="seq">'
    +   '<div class="sqh">'
    +     '<span class="sqt">one token sequence</span>'
    +     '<span class="lgd">'
    +       '<span class="lg"><i class="c tk" aria-hidden="true"></i>text token</span>'
    +       '<span class="lg"><i class="c px" aria-hidden="true"></i>image patch token</span>'
    +     '</span>'
    +   '</div>'
    +   '<div class="trk" role="img" id="S_BASE-trk" aria-label="token sequence">'
    +     seg(FIELDS[0], N_INS, 'fix', cells(N_INS, 'tk dim'))
    +     seg(FIELDS[1], N_QRY, 'fix', cells(N_QRY, 'tk dim'))
    +     seg(FIELDS[2], N_DOC, 'doc', '', 'S_BASE-doc')
    +   '</div>'
    +   '<p class="sqs" id="S_BASE-say" aria-live="polite"></p>'
    + '</div>'

    + '<div class="drop"><i></i></div>'

    + '<div class="mdl">'
    +   '<b class="big">' + esc(PARAMS) + '</b>'
    +   '<span class="mside">'
    +     '<span class="mname">Ministral-3-3B-Base-2512</span>'
    +     '<span class="msub">Mistral 3 family, natively multimodal. Text and image '
    +       'moderation share one set of weights.</span>'
    +     '<span class="mtags">' + mtags + '</span>'
    +   '</span>'
    + '</div>'

    + '<div class="drop"><i></i></div>'

    + '<div class="out">'
    +   '<span class="oside">'
    +     '<span class="olab">one output token</span>'
    +     '<span class="pills">'
    +       '<span class="pill yes">yes</span>'
    +       '<span class="pill no">no</span>'
    +     '</span>'
    +     '<span class="onote">' + esc(onlyLine) + '</span>'
    +   '</span>'
    +   '<span class="ostats">'
    +     '<em class="ocap">the same checkpoint, both regimes</em>'
    +     '<span class="srow">' + stats + '</span>'
    +   '</span>'
    + '</div>'

    + '<p class="foot">A schematic, not a live model call. Cell counts in the strip are '
    +   'illustrative, the report publishes no token counts. Checkpoint and encoder names '
    +   'come from the report text, every number and field name here is read from '
    +   'window.SS. '
    +   '<span class="hint" id="S_BASE-hint">cycling on its own, click or use arrow keys to '
    +   'take over</span></p>'

    + '</div>'
  ).firstChild);

  /* ---------- handles ---------- */
  var btns = [], i;
  for (i = 0; i < SHAPES.length; i++) btns.push(root.querySelector('#S_BASE-shc-' + i));
  var box = root.querySelector('#S_BASE-shp');
  var lkTok = root.querySelector('#S_BASE-lk-tok');
  var lkVis = root.querySelector('#S_BASE-lk-vis');
  var plTok = root.querySelector('#S_BASE-pl-tok');
  var plVis = root.querySelector('#S_BASE-pl-vis');
  var lk2Vis = root.querySelector('#S_BASE-lk2-vis');
  var evis = root.querySelector('#S_BASE-evis');
  var vtag = root.querySelector('#S_BASE-vtag');
  var docCells = root.querySelector('#S_BASE-doc');
  var docN = root.querySelector('#S_BASE-doc-n');
  var trk = root.querySelector('#S_BASE-trk');
  var say = root.querySelector('#S_BASE-say');
  var hint = root.querySelector('#S_BASE-hint');

  var COLX = [16.667, 50, 83.333];
  var cur = -1;
  var auto = true;

  function setShape(idx, fromUser) {
    if (idx === cur || idx < 0 || idx >= SHAPES.length) return;
    var s = SHAPES[idx], j;

    for (j = 0; j < btns.length; j++) {
      btns[j].setAttribute('aria-checked', j === idx ? 'true' : 'false');
      btns[j].tabIndex = j === idx ? 0 : -1;
      btns[j].classList.toggle('on', j === idx);
    }

    /* two lanes so the shared horizontal run never draws one wire on top of
       the other when both encoders are fed */
    var d1 = bus(COLX[idx], 25, 34, 0.38);
    var d2 = bus(COLX[idx], 75, 34, 0.66);
    lkTok.setAttribute('d', d1);
    plTok.setAttribute('d', d1);
    lkVis.setAttribute('d', d2);
    plVis.setAttribute('d', d2);
    lkVis.classList.toggle('off', !s.img);
    plVis.classList.toggle('off', !s.img);
    lk2Vis.classList.toggle('off', !s.img);
    evis.classList.toggle('on', !!s.img);
    vtag.textContent = s.img ? 'engaged' : 'idle on this input';

    docCells.innerHTML = cells(s.patch, 'px') + cells(s.tok, 'tk');
    if (docN) docN.textContent = String(s.patch + s.tok);

    var parts = [];
    if (s.patch) parts.push(s.patch + ' image patch tokens');
    if (s.tok) parts.push(s.tok + ' text tokens');
    say.innerHTML = '&lt;' + esc(FIELDS[2]) + '&gt; carries <b>' + esc(parts.join(' then '))
      + '</b>, ' + esc(s.say) + '. The two fields in front of it are unchanged.';
    trk.setAttribute('aria-label', 'token sequence for ' + s.name + ', '
      + N_INS + ' instruction tokens, ' + N_QRY + ' query tokens, then '
      + parts.join(' then '));

    if (fromUser && auto) {
      auto = false;
      hint.textContent = 'manual, arrow keys move the focus';
    }
    cur = idx;
  }

  btns.forEach(function (b, j) {
    b.addEventListener('click', function () { setShape(j, true); });
  });
  box.addEventListener('keydown', function (e) {
    var k = e.key, nx = -1, n = SHAPES.length;
    if (k === 'ArrowLeft' || k === 'ArrowUp') nx = (cur - 1 + n) % n;
    else if (k === 'ArrowRight' || k === 'ArrowDown') nx = (cur + 1) % n;
    else if (k === 'Home') nx = 0;
    else if (k === 'End') nx = n - 1;
    if (nx < 0) return;
    e.preventDefault();
    setShape(nx, true);
    btns[nx].focus();
  });

  setShape(0, false);
  if (api.reduce) hint.textContent = 'click or use arrow keys to change the input shape';

  /* ---------- the packet running down the active wiring ---------- */
  var running = false, nextAt = null;

  function pulse(p, t) {
    if (!p || p.classList.contains('off')) return;
    var L = 0;
    try { L = p.getTotalLength(); } catch (e) { L = 0; }
    if (!L) return;
    p.style.strokeDasharray = (L * 0.18).toFixed(2) + ' ' + (L * 0.82).toFixed(2);
    p.style.strokeDashoffset = (-((t * 0.62) % 1) * L).toFixed(2);
  }

  return {
    start: function () { running = true; nextAt = null; },
    stop: function () { running = false; },
    tick: function (t) {
      if (!running || api.reduce) return;
      pulse(plTok, t);
      pulse(plVis, t);
      if (!auto) return;
      if (nextAt === null) { nextAt = t + 3.6; return; }
      if (t >= nextAt) {
        nextAt = t + 3.6;
        setShape((cur + 1) % SHAPES.length, false);
      }
    }
  };
};
