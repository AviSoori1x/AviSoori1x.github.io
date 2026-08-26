window.SCENES = window.SCENES || {};

/* S_REWRITE, act 2, beat 17. The generation loop.
   A safe source text, a target category and a sibling category go into one LLM
   call. Three fields come back, and they assemble into two training rows that
   carry the identical document and disagree on the label. The split is the
   animated moment.
   Everything on screen is read from window.SS at runtime. The document, both
   queries, both labels and the two role names come from SS.fig3. The output
   field names and the input slot names are parsed out of the real prompt body
   in SS.genPrompts[0], which also sits behind the disclosure together with
   SS.genSystem and SS.headline.genTemp. The volume line is
   SS.headline.syntheticText. Category names are matched against
   SS.evalTaxonomy by looking for a leaf name inside the query. */
window.SCENES['S_REWRITE'] = function (root, api) {
  var SS = api.SS || {};
  var F = SS.fig3 || {};
  var POS = F.positive || {};
  var NEG = F.negative || {};
  var head = SS.headline || {};
  var PROMPTS = SS.genPrompts || [];
  var P0 = PROMPTS[0] || {};
  var TAX = SS.evalTaxonomy || [];

  var ID = 'S_REWRITE';
  var DOT = '·';
  var BODY = P0.body == null ? '' : String(P0.body);
  var DOC = F.document == null ? '' : String(F.document);
  var INS = F.instruct == null ? '' : String(F.instruct);
  var TEMP = head.genTemp;
  var SYN = head.syntheticText;
  var SYSP = SS.genSystem == null ? '' : String(SS.genSystem);

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ---------- read the prompt, do not retype it ---------- */
  /* the output contract is literally the block of shouty field names */
  var FIELDS = BODY.match(/^[A-Z][A-Z_]+:$/gm) || [];
  FIELDS = FIELDS.map(function (s) { return s.replace(/:$/, ''); });
  function field(re, i) {
    for (var k = 0; k < FIELDS.length; k++) if (re.test(FIELDS[k])) return FIELDS[k];
    return FIELDS[i] || null;
  }
  var F_REW = field(/REWRIT/, 0);
  var F_POS = field(/^POSITIVE/, 1);
  var F_NEG = field(/^NEGATIVE/, 2);

  /* the input contract is the set of braced slots, minus the repeated
     description slot that just trails each category name */
  var SLOTS = {};
  (BODY.match(/\{[a-z_]+\}/g) || []).forEach(function (s) { SLOTS[s] = 1; });
  function slot(n) { return SLOTS['{' + n + '}'] ? '{' + n + '}' : null; }

  /* ---------- which taxonomy leaf is each query about ---------- */
  var LEAVES = [];
  TAX.forEach(function (sc) {
    (sc.subs || []).forEach(function (sb) {
      (sb.leaves || []).forEach(function (lf) {
        LEAVES.push({ name: String(lf.name), sub: String(sb.name), sc: String(sc.name) });
      });
    });
  });
  function matchLeaf(q) {
    var hay = String(q == null ? '' : q).toLowerCase();
    var best = null;
    for (var i = 0; i < LEAVES.length; i++) {
      var n = LEAVES[i].name.toLowerCase();
      if (n.length > 3 && hay.indexOf(n) >= 0) {
        if (!best || n.length > best.name.length) best = LEAVES[i];
      }
    }
    return best;
  }
  var LP = matchLeaf(POS.query);
  var LN = matchLeaf(NEG.query);

  /* ---------- the three inputs ---------- */
  var INPUTS = [
    {
      key: 'text',
      slot: slot('text'),
      role: 'safe source text',
      val: null,
      sub: 'a benign passage, not printed in the paper',
      empty: true
    },
    {
      key: 'target',
      slot: slot('target_category'),
      role: POS.role || 'target category',
      val: LP ? LP.name : null,
      sub: (POS.role || 'target category') + (LP ? (' ' + DOT + ' ' + LP.sc) : ''),
      empty: !LP
    },
    {
      key: 'sibling',
      slot: slot('negative_category'),
      role: NEG.role || 'sibling category',
      val: LN ? LN.name : null,
      sub: (NEG.role || 'sibling category') + (LN ? (' ' + DOT + ' ' + LN.sc) : ''),
      empty: !LN
    }
  ].filter(function (o) { return o.slot; });

  var OUTS = [
    { key: 'rew', name: F_REW, val: DOC, kind: 'the unsafe rewrite' },
    { key: 'pq', name: F_POS, val: POS.query, kind: 'question about the target' },
    { key: 'nq', name: F_NEG, val: NEG.query, kind: 'question about the sibling' }
  ].filter(function (o) { return o.name && o.val; });

  var YES = String(POS.label == null ? 'yes' : POS.label).toLowerCase();
  var NO = String(NEG.label == null ? 'no' : NEG.label).toLowerCase();

  /* ---------- glyphs, so the two labels never read by colour alone ---------- */
  function gTick() {
    return '<svg class="gl" viewBox="0 0 20 20" aria-hidden="true">'
      + '<path d="M4 10.4 L8.2 14.6 L16 6.1"></path></svg>';
  }
  function gSlash() {
    return '<svg class="gl" viewBox="0 0 20 20" aria-hidden="true">'
      + '<circle cx="10" cy="10" r="6.6"></circle><path d="M5.6 14.4 L14.4 5.6"></path></svg>';
  }

  /* ---------- markup ---------- */
  function inputHtml(o, i) {
    return '<div class="inp" id="' + ID + '-inp-' + i + '">'
      + '<span class="slot">' + esc(o.slot) + '</span>'
      + (o.empty
        ? '<span class="val ghost">' + esc(o.role) + '</span>'
        : '<span class="val">' + esc(o.val) + '</span>')
      + '<span class="sub">' + esc(o.sub) + '</span>'
      + '</div>';
  }

  function outHtml(o, i) {
    return '<div class="out" id="' + ID + '-out-' + i + '">'
      + '<span class="fname">' + esc(o.name) + '</span>'
      + '<span class="fkind">' + esc(o.kind) + '</span>'
      + '<span class="fval">' + esc(o.val) + '</span>'
      + '</div>';
  }

  function rowHtml(k, o) {
    return '<div class="row ' + o.cls + '" id="' + ID + '-row-' + k + '">'
      + '<span class="rhd">'
      +   '<span class="rmeta">'
      +     '<b>row ' + (k + 1) + '</b>'
      +     '<span class="rkind">' + esc(o.kind) + '</span>'
      +   '</span>'
      +   '<span class="rans">'
      +     '<i>label</i>'
      +     '<span class="rval">' + o.glyph + '<b>' + esc(o.label) + '</b></span>'
      +   '</span>'
      + '</span>'
      + '<span class="rfields">'
      +   (INS
        ? ('<span class="fl"><i>instruct</i><span class="ft dim">' + esc(INS) + '</span></span>')
        : '')
      +   '<span class="fl"><i>document</i><span class="ft doc1">' + esc(DOC) + '</span></span>'
      +   '<span class="fl"><i>query</i><span class="ft q">' + esc(o.q) + '</span></span>'
      + '</span>'
      + '</div>';
  }

  function slotify(s) {
    /* highlight the braced slots inside the real prompt, escaping first */
    return esc(s).replace(/\{[a-z_]+\}/g, function (m) {
      return '<b class="sl">' + m + '</b>';
    });
  }

  var funnel = '<svg class="fun" viewBox="0 0 840 26" preserveAspectRatio="xMidYMid meet"'
    + ' aria-hidden="true">'
    + '<path class="fp" d="M140 0 C140 17 420 6 420 23"></path>'
    + '<path class="fp" d="M420 0 L420 23"></path>'
    + '<path class="fp" d="M700 0 C700 17 420 6 420 23"></path>'
    + '<path class="fh" d="M414 18 L420 25 L426 18"></path>'
    + '</svg>';

  var fork = '<svg class="frk" viewBox="0 0 840 36" preserveAspectRatio="xMidYMid meet"'
    + ' aria-hidden="true">'
    + '<path class="stem" d="M420 0 L420 11"></path>'
    + '<path class="br brA" d="M420 11 C420 28 208 19 208 34"></path>'
    + '<path class="br brB" d="M420 11 C420 28 632 19 632 34"></path>'
    + '<path class="fh fhA" d="M202 29 L208 36 L214 29"></path>'
    + '<path class="fh fhB" d="M626 29 L632 36 L638 29"></path>'
    + '</svg>';

  var nPrompt = PROMPTS.length;
  var discLabel = 'the exact prompt';

  var sheet = '<div class="sheet" id="' + ID + '-sheet" hidden>'
    + '<div class="shd">'
    +   '<span class="klab hot">generation prompt</span>'
    +   '<span class="shname">' + esc(P0.name || '') + '</span>'
    +   '<button type="button" class="shx" id="' + ID + '-close"'
    +     ' aria-label="close the prompt">close</button>'
    + '</div>'
    + (SYSP
      ? ('<div class="shsys"><span class="klab">system</span>'
         + '<span class="systxt">' + esc(SYSP) + '</span></div>')
      : '')
    + '<div class="shchips">'
    +   (TEMP != null ? '<span class="pin hot">temperature ' + esc(TEMP) + '</span>' : '')
    +   (nPrompt ? '<span class="pin">prompt 1 of ' + nPrompt + ' in the paper</span>' : '')
    +   '<span class="pin">' + FIELDS.length + ' output fields</span>'
    + '</div>'
    + '<pre class="shbody">' + slotify(BODY) + '</pre>'
    + '<p class="shnote">Verbatim from the paper. The braced slots are filled per call, '
    + 'one target category and one sibling category at a time.</p>'
    + '</div>';

  var volNote = (SYN != null)
    ? ('Repeated over the corpora this stage yields about ' + SYN + 'M synthetic text samples, '
       + 'covering these target and sibling pairs plus the ancestor positives in the next scene.')
    : 'Repeated over the corpora this stage yields the synthetic slice of the training mix, '
      + 'covering target and sibling pairs plus ancestor positives.';

  root.className = 'sc-s_rewrite';
  root.appendChild(api.frag(
    '<div class="wrap">'

    + '<div class="hd">'
    +   '<span class="eyebrow">the rewrite loop</span>'
    +   '<span class="hnote">one call in, two rows out, and they disagree on purpose</span>'
    + '</div>'

    + '<div class="inps" id="' + ID + '-inps">' + INPUTS.map(inputHtml).join('') + '</div>'

    + funnel

    + '<div class="call" id="' + ID + '-call">'
    +   '<span class="cdot" aria-hidden="true"></span>'
    +   '<span class="ctxt"><b>1 LLM call</b>'
    +     '<span class="csub">' + esc(P0.name || 'category specific rewriting') + '</span>'
    +   '</span>'
    +   (TEMP != null ? '<span class="pin hot">temp ' + esc(TEMP) + '</span>' : '')
    +   '<button type="button" class="disc" id="' + ID + '-disc"'
    +     ' aria-expanded="false" aria-controls="' + ID + '-sheet">'
    +     esc(discLabel) + '</button>'
    + '</div>'

    + '<div class="outs" id="' + ID + '-outs">'
    +   '<span class="outlab">one response, ' + OUTS.length + ' fields</span>'
    +   OUTS.slice(0, 1).map(outHtml).join('')
    +   '<div class="outq">'
    +     OUTS.slice(1).map(function (o, i) { return outHtml(o, i + 1); }).join('')
    +   '</div>'
    + '</div>'

    + '<div class="splitcap"><span id="' + ID + '-cap">the same rewrite goes into both rows'
    +   '</span></div>'
    + fork

    + '<div class="rows" id="' + ID + '-rows">'
    +   rowHtml(0, {
      cls: 'pos', kind: 'positive', q: POS.query, label: YES, glyph: gTick()
    })
    +   rowHtml(1, {
      cls: 'neg', kind: 'hard negative', q: NEG.query, label: NO, glyph: gSlash()
    })
    + '</div>'

    + '<div class="tiewrap"><span class="tie" id="' + ID + '-tie">identical document, '
    +   DOC.length + ' characters, only the query moved</span></div>'

    + '<div class="sum">'
    +   '<span class="stat"><b class="n1">1</b><i>LLM call</i></span>'
    +   '<svg class="sarw" viewBox="0 0 34 20" aria-hidden="true">'
    +     '<path d="M2 10 H27"></path><path d="M21 4 L28 10 L21 16"></path></svg>'
    +   '<span class="stat big"><b class="n2" id="' + ID + '-count">2</b>'
    +     '<i>training rows,<br>opposite labels</i></span>'
    +   '<span class="sumnote">' + esc(volNote) + '</span>'
    + '</div>'

    + '<div class="foot">'
    +   '<span class="gt">Paper example plus the real generation prompt, not a live model call. '
    +   'The safe source text is not printed in the paper. Category names are matched to the '
    +   'evaluation taxonomy by the leaf name appearing in the query.</span>'
    +   '<button type="button" class="rep" id="' + ID + '-rep">replay</button>'
    + '</div>'

    + sheet
    + '</div>'
  ).firstChild);

  /* ---------- wiring ---------- */
  var inpEls = [];
  for (var i = 0; i < INPUTS.length; i++) inpEls.push(root.querySelector('#' + ID + '-inp-' + i));
  var outEls = [];
  for (var j = 0; j < OUTS.length; j++) outEls.push(root.querySelector('#' + ID + '-out-' + j));
  var callEl = root.querySelector('#' + ID + '-call');
  var rowA = root.querySelector('#' + ID + '-row-0');
  var rowB = root.querySelector('#' + ID + '-row-1');
  var brA = root.querySelector('.brA');
  var brB = root.querySelector('.brB');
  var hdA = root.querySelector('.fhA');
  var hdB = root.querySelector('.fhB');
  var stem = root.querySelector('.stem');
  var tie = root.querySelector('#' + ID + '-tie');
  var cap = root.querySelector('#' + ID + '-cap');
  var countEl = root.querySelector('#' + ID + '-count');
  var sumEl = root.querySelector('.sc-s_rewrite .sum');
  var repBtn = root.querySelector('#' + ID + '-rep');
  var discBtn = root.querySelector('#' + ID + '-disc');
  var sheetEl = root.querySelector('#' + ID + '-sheet');
  var closeBtn = root.querySelector('#' + ID + '-close');

  var NIN = inpEls.length;
  var NOUT = outEls.length;
  var P_CALL = NIN + 1;
  var P_OUT0 = P_CALL + 1;
  var P_ROWA = P_OUT0 + NOUT;
  var P_ROWB = P_ROWA + 1;
  var P_TIE = P_ROWB + 1;
  var MAXP = P_TIE;

  var CAPS = {};
  CAPS[0] = 'a safe passage and two category names go in';
  CAPS[P_CALL] = 'one call, one response';
  CAPS[P_OUT0] = 'three fields come back';
  CAPS[P_ROWA] = 'the rewrite plus the target query answers ' + YES;
  CAPS[P_ROWB] = 'the same rewrite plus the sibling query answers ' + NO;
  CAPS[P_TIE] = 'two rows, one document, opposite labels';

  var phase = MAXP;

  function setPhase(p) {
    phase = p;
    var k;
    for (k = 0; k < NIN; k++) inpEls[k].classList.toggle('lit', p >= k + 1);
    callEl.classList.toggle('lit', p >= P_CALL);
    callEl.classList.toggle('fire', p === P_CALL);
    for (k = 0; k < NOUT; k++) outEls[k].classList.toggle('lit', p >= P_OUT0 + k);
    stem.classList.toggle('lit', p >= P_ROWA);
    brA.classList.toggle('lit', p >= P_ROWA);
    hdA.classList.toggle('lit', p >= P_ROWA);
    brB.classList.toggle('lit', p >= P_ROWB);
    hdB.classList.toggle('lit', p >= P_ROWB);
    rowA.classList.toggle('lit', p >= P_ROWA);
    rowB.classList.toggle('lit', p >= P_ROWB);
    tie.classList.toggle('lit', p >= P_TIE);
    sumEl.classList.toggle('lit', p >= P_TIE);
    var n = (p >= P_ROWA ? 1 : 0) + (p >= P_ROWB ? 1 : 0);
    countEl.textContent = String(n);
    countEl.classList.toggle('zero', n === 0);
    /* find the newest caption at or below this phase */
    var best = 0;
    for (var key in CAPS) if (CAPS.hasOwnProperty(key)) {
      var kk = +key;
      if (kk <= p && kk >= best) best = kk;
    }
    cap.textContent = CAPS[best];
  }

  /* ---------- the prompt disclosure ---------- */
  var lastFocus = null;
  function openSheet() {
    lastFocus = document.activeElement;
    sheetEl.hidden = false;
    discBtn.setAttribute('aria-expanded', 'true');
    if (closeBtn) closeBtn.focus();
  }
  function closeSheet() {
    sheetEl.hidden = true;
    discBtn.setAttribute('aria-expanded', 'false');
    if (lastFocus && lastFocus.focus) lastFocus.focus();
    else discBtn.focus();
  }
  discBtn.addEventListener('click', function () {
    if (sheetEl.hidden) openSheet(); else closeSheet();
  });
  if (closeBtn) closeBtn.addEventListener('click', closeSheet);
  root.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !sheetEl.hidden) { e.preventDefault(); closeSheet(); }
  });

  /* ---------- playback ---------- */
  var running = false, nextAt = null, replayAt = 0;

  repBtn.addEventListener('click', function () {
    /* if the pump is not driving this scene, replay would strand it at phase 0 */
    if (api.reduce || !running) { setPhase(MAXP); return; }
    setPhase(0);
    nextAt = null;
    replayAt = 1;
  });

  if (api.reduce) {
    repBtn.hidden = true;
    root.querySelector('.sc-s_rewrite .foot').classList.add('still');
  }

  setPhase(MAXP);

  return {
    start: function () {
      running = true;
      nextAt = null;
      if (api.reduce) { setPhase(MAXP); return; }
      setPhase(0);
    },
    stop: function () { running = false; },
    tick: function (t) {
      if (!running || api.reduce) return;
      if (nextAt === null) { nextAt = t + (replayAt ? 0.3 : 0.6); replayAt = 0; return; }
      if (t < nextAt) return;
      if (phase >= MAXP) {
        setPhase(0);
        nextAt = t + 0.9;
        return;
      }
      setPhase(phase + 1);
      nextAt = t + (phase >= MAXP ? 3.6 : (phase === P_CALL ? 0.95 : 0.72));
    }
  };
};
