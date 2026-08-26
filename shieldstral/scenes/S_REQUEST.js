window.SCENES = window.SCENES || {};

/* Act I, scene 03. Anatomy of one request.
   A fixed system line, then three tagged fields. Two controls, the strictness
   tier and the document format family, rewrite a handful of words each and
   leave every other byte of the request exactly where it was.
   System text, tiers, format templates and the sample all come from window.SS. */
window.SCENES['S_REQUEST'] = function (root, api) {
  var SS = api.SS || {};

  var SYS = SS.systemPrompt == null ? '' : String(SS.systemPrompt);
  var TIERS = (SS.strictness || []).slice();
  var FMTS = (SS.formats || []).slice();
  var F0 = (SS.fig2 || [])[0] || {};

  if (!TIERS.length) TIERS = [{ level: 'n/a', domains: 'n/a', rationale: '' }];
  if (!FMTS.length) FMTS = [{ family: 'n/a', tpl: '{prompt}\n{response}' }];

  var QUERY = F0.query == null ? '' : String(F0.query);
  var KIND = F0.kind == null ? '' : String(F0.kind);

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ---- the document, split back into the two payload strings ----
     SS.fig2[0].document ships in the bracketed format. Pull the prompt and the
     response out of it once, then every format family is just new scaffolding
     wrapped around those same two strings. */
  function splitDoc(d) {
    var s = String(d == null ? '' : d);
    var parts = s.split(/\[Assistant\]/i);
    var p = (parts[0] || '').replace(/\[User\]/i, '').trim();
    var r = (parts[1] || '').trim();
    if (!r) { p = s.trim(); r = ''; }
    return { prompt: p, response: r };
  }
  var PAY = splitDoc(F0.document);

  /* ---- the instruct frame, taken from the paper's own sample ----
     Everything before the tier word and the run up to "reviewing" is reused
     verbatim, so the words that move are only the ones the tier supplies. */
  function instructFrame(raw, firstLevel) {
    var pre = 'You are a ';
    var mid = ' safety moderator reviewing ';
    var lvl = String(firstLevel || '').toLowerCase();
    var txt = String(raw || '');
    var i = lvl ? txt.toLowerCase().indexOf(lvl) : -1;
    if (i >= 0) {
      pre = txt.slice(0, i);
      var m = txt.slice(i + lvl.length).match(/^([\s\S]*?reviewing\s)/);
      if (m) mid = m[1];
    }
    return { pre: pre, mid: mid, glue: '. ', end: '.' };
  }
  var FR = instructFrame(F0.instruct, TIERS[0] && TIERS[0].level);

  /* the three words the tier owns, interleaved with the four frame runs, so the
     count in the pin can never drift away from what is actually rewritten */
  var ISLOT = ['lvl', 'dom', 'rat'];
  var IFRAME = [FR.pre, FR.mid, FR.glue, FR.end];
  var insHtml = '';
  IFRAME.forEach(function (f, k) {
    insHtml += '<span class="frame">' + esc(f) + '</span>';
    if (k < ISLOT.length) {
      insHtml += '<span class="mv" id="S_REQUEST-i-' + ISLOT[k] + '"></span>';
    }
  });

  function tierBtn(t, i) {
    return '<button type="button" class="seg" id="S_REQUEST-t' + i + '"'
      + ' role="radio" aria-checked="false" tabindex="-1" data-i="' + i + '">'
      + '<span class="segdot" aria-hidden="true"></span>'
      + '<span class="segtxt">' + esc(t.level || ('tier ' + (i + 1))) + '</span>'
      + '</button>';
  }
  function fmtBtn(f, i) {
    return '<button type="button" class="fmt" id="S_REQUEST-f' + i + '"'
      + ' role="radio" aria-checked="false" tabindex="-1" data-i="' + i + '">'
      + esc(f.family || ('format ' + (i + 1))) + '</button>';
  }

  root.classList.add('sc-s_request');
  root.appendChild(api.frag(
    '<div class="wrap">'

    + '<div class="hd">'
    +   '<span class="ey">anatomy of one request</span>'
    +   '<span class="note">' + TIERS.length + ' tiers &times; ' + FMTS.length
    +   ' formats, one document</span>'
    + '</div>'

    + '<div class="sys">'
    +   '<span class="syslab">system, fixed</span>'
    +   '<span class="systxt">' + esc(SYS) + '</span>'
    + '</div>'

    + '<div class="fields">'

    +   '<div class="fld">'
    +     '<div class="fhd"><span class="ftag">&lt;Instruct&gt;</span>'
    +       '<span class="frole">the framing, one per product surface</span>'
    +       '<span class="fpin mvpin">strictness rewrites ' + ISLOT.length
    +         ' spans</span></div>'
    +     '<div class="ftxt">' + insHtml + '</div>'
    +   '</div>'

    +   '<div class="fld">'
    +     '<div class="fhd"><span class="ftag">&lt;Query&gt;</span>'
    +       '<span class="frole">the actual question, one per policy</span>'
    +       '<span class="fpin">held by both controls</span></div>'
    +     '<div class="ftxt"><span class="hold">' + esc(QUERY) + '</span></div>'
    +   '</div>'

    +   '<div class="fld doc">'
    +     '<div class="fhd"><span class="ftag">&lt;Document&gt;</span>'
    +       '<span class="frole">the content under review</span>'
    +       '<span class="fpin mvpin">format rewrites the scaffolding</span></div>'
    +     '<div class="ftxt pre">'
    +       '<span class="mv" id="S_REQUEST-d-a"></span>'
    +       '<span class="hold">' + esc(PAY.prompt) + '</span>'
    +       '<span class="mv" id="S_REQUEST-d-b"></span>'
    +       '<span class="hold">' + esc(PAY.response) + '</span>'
    +       '<span class="mv" id="S_REQUEST-d-c"></span>'
    +     '</div>'
    +     '<div class="docsrc">payload bytes from the paper&#39;s sample'
    +       (KIND ? ', ' + esc(KIND.toLowerCase()) : '') + ', never edited</div>'
    +   '</div>'

    + '</div>'

    + '<div class="ctl">'
    +   '<div class="cgrp">'
    +     '<div class="chd"><span class="clab">strictness tier</span>'
    +       '<span class="cnum" id="S_REQUEST-tnum"></span></div>'
    +     '<div class="segs" role="radiogroup" aria-label="strictness tier">'
    +       TIERS.map(tierBtn).join('') + '</div>'
    +   '</div>'
    +   '<div class="cgrp">'
    +     '<div class="chd"><span class="clab">document format family</span>'
    +       '<span class="cnum" id="S_REQUEST-fnum"></span></div>'
    +     '<div class="fmts" role="radiogroup" aria-label="document format family">'
    +       FMTS.map(fmtBtn).join('') + '</div>'
    +     '<div class="tpl" id="S_REQUEST-tpl"></div>'
    +   '</div>'
    + '</div>'

    + '<div class="meter">'
    +   '<div class="mcell"><b id="S_REQUEST-chg">0</b>'
    +     '<i>characters rewritten by the last change</i></div>'
    +   '<div class="mbar"><span class="mfill" id="S_REQUEST-bar"></span></div>'
    +   '<div class="mcell right"><b id="S_REQUEST-held">0</b>'
    +     '<i>characters left exactly where they were</i></div>'
    + '</div>'

    + '<div class="foot">'
    +   '<span class="legend">dashed underline, written by a control. plain, held</span>'
    +   '<span class="hint" id="S_REQUEST-hint">cycling, click or arrow keys to take '
    +     'over</span>'
    + '</div>'
    + '<div class="gt">Illustrative assembly: the paper&#39;s sample with the tier words '
    +   'and the format scaffolding swapped in. No model is called.</div>'

    + '</div>'
  ).firstChild);

  var iLvl = root.querySelector('#S_REQUEST-i-lvl');
  var iDom = root.querySelector('#S_REQUEST-i-dom');
  var iRat = root.querySelector('#S_REQUEST-i-rat');
  var dA = root.querySelector('#S_REQUEST-d-a');
  var dB = root.querySelector('#S_REQUEST-d-b');
  var dC = root.querySelector('#S_REQUEST-d-c');
  var tplEl = root.querySelector('#S_REQUEST-tpl');
  var tNum = root.querySelector('#S_REQUEST-tnum');
  var fNum = root.querySelector('#S_REQUEST-fnum');
  var chgEl = root.querySelector('#S_REQUEST-chg');
  var heldEl = root.querySelector('#S_REQUEST-held');
  var barEl = root.querySelector('#S_REQUEST-bar');
  var hintEl = root.querySelector('#S_REQUEST-hint');
  var segs = Array.prototype.slice.call(root.querySelectorAll('.seg'));
  var fmts = Array.prototype.slice.call(root.querySelectorAll('.fmt'));

  var tIdx = -1, fIdx = -1;
  var auto = true;

  function setSpan(node, txt, animate) {
    var next = String(txt == null ? '' : txt);
    if (node.textContent === next) return 0;
    node.textContent = next;
    if (animate && !api.reduce) {
      node.classList.remove('flash');
      void node.offsetWidth;
      node.classList.add('flash');
    }
    return next.length;
  }

  /* total request size, recounted from the DOM so the held figure is real */
  function totalChars() {
    var n = SYS.length;
    Array.prototype.forEach.call(root.querySelectorAll('.ftxt'), function (f) {
      n += f.textContent.length;
    });
    return n;
  }

  function report(changed) {
    var total = totalChars();
    var held = Math.max(0, total - changed);
    chgEl.textContent = String(changed);
    heldEl.textContent = String(held);
    barEl.style.width = (total ? (100 * changed / total) : 0).toFixed(1) + '%';
  }

  function pickTier(i, animate) {
    if (i === tIdx) return 0;
    var t = TIERS[i] || {};
    tIdx = i;
    segs.forEach(function (b, k) {
      b.setAttribute('aria-checked', k === i ? 'true' : 'false');
      b.tabIndex = k === i ? 0 : -1;
      b.classList.toggle('sel', k === i);
    });
    tNum.textContent = 'tier ' + (i + 1) + ' of ' + TIERS.length;
    var n = 0;
    n += setSpan(iLvl, String(t.level || '').toLowerCase(), animate);
    n += setSpan(iDom, String(t.domains || '').toLowerCase(), animate);
    n += setSpan(iRat, String(t.rationale || ''), animate);
    return n;
  }

  function pickFmt(i, animate) {
    if (i === fIdx) return 0;
    var f = FMTS[i] || {};
    var tpl = String(f.tpl == null ? '' : f.tpl);
    fIdx = i;
    fmts.forEach(function (b, k) {
      b.setAttribute('aria-checked', k === i ? 'true' : 'false');
      b.tabIndex = k === i ? 0 : -1;
      b.classList.toggle('sel', k === i);
    });
    fNum.textContent = 'format ' + (i + 1) + ' of ' + FMTS.length;
    tplEl.textContent = 'template  ' + tpl.replace(/\n/g, '\\n');

    var iP = tpl.indexOf('{prompt}');
    var iR = tpl.indexOf('{response}');
    var a, b2, c;
    if (iP >= 0 && iR > iP) {
      a = tpl.slice(0, iP);
      b2 = tpl.slice(iP + 8, iR);
      c = tpl.slice(iR + 10);
    } else {
      a = ''; b2 = '\n'; c = '';
    }
    var n = 0;
    n += setSpan(dA, a, animate);
    n += setSpan(dB, b2, animate);
    n += setSpan(dC, c, animate);
    return n;
  }

  function takeOver() {
    if (!auto) return;
    auto = false;
    hintEl.textContent = 'manual, arrow keys move within a control';
  }

  function wireGroup(nodes, pick) {
    nodes.forEach(function (b) {
      b.addEventListener('click', function () {
        takeOver();
        report(pick(+b.getAttribute('data-i'), true));
      });
    });
    if (!nodes.length) return;
    nodes[0].parentNode.addEventListener('keydown', function (e) {
      var cur = nodes.indexOf(document.activeElement);
      if (cur < 0) return;
      var k = e.key, to = -1;
      if (k === 'ArrowLeft' || k === 'ArrowUp') to = (cur - 1 + nodes.length) % nodes.length;
      else if (k === 'ArrowRight' || k === 'ArrowDown') to = (cur + 1) % nodes.length;
      else if (k === 'Home') to = 0;
      else if (k === 'End') to = nodes.length - 1;
      if (to < 0) return;
      e.preventDefault();
      takeOver();
      report(pick(to, true));
      nodes[to].focus();
    });
  }
  wireGroup(segs, pickTier);
  wireGroup(fmts, pickFmt);

  pickTier(0, false);
  /* the sample ships in the bracketed family, so open on it if it is there */
  var startF = 0;
  FMTS.forEach(function (f, i) {
    if (/^\s*\[/.test(String(f.tpl || ''))) startF = startF || i;
  });
  pickFmt(startF, false);
  report(0);
  if (api.reduce) hintEl.textContent = 'static, use the controls to rewrite the request';

  /* auto cycle so a reader who only scrolls still sees the rewrite happen */
  var running = false, nextAt = null, step = 0;
  return {
    start: function () { running = true; nextAt = null; },
    stop: function () { running = false; },
    tick: function (t) {
      if (!running || !auto || api.reduce) return;
      if (nextAt === null) { nextAt = t + 2.0; return; }
      if (t < nextAt) return;
      nextAt = t + 2.5;
      step++;
      if (step % 3 === 0) report(pickTier((tIdx + 1) % TIERS.length, true));
      else report(pickFmt((fIdx + 1) % FMTS.length, true));
    }
  };
};
