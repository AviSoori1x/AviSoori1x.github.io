window.SCENES = window.SCENES || {};

/* Act IV, scene 33. Running it.
   One dark code card holding the scoring helper, hand tokenised and coloured
   here rather than by a library, plus the vllm one liner above it. The SYSTEM
   constant is assembled from window.SS.systemPrompt at render time, wrapped
   into Python string literals, so the code on screen can never drift away from
   the prompt the rest of the guide shows. Four buttons under the card focus the
   lines that carry the non obvious part, the logprob pull and the renormalise. */
window.SCENES['S_RUN'] = function (root, api) {
  var SS = api.SS || {};
  var SYS = SS.systemPrompt == null ? '' : String(SS.systemPrompt);

  /* ---------- the system prompt, wrapped into Python literals ----------
     Greedy wrap on word boundaries. Every segment after the first carries a
     leading space, so concatenating the literals reproduces SS.systemPrompt
     byte for byte. */
  function wrapWords(s, width) {
    var words = String(s).split(/\s+/).filter(Boolean);
    var out = [], cur = '';
    words.forEach(function (w) {
      var candidate = cur ? cur + ' ' + w : w;
      if (cur && candidate.length > width) { out.push(cur); cur = w; }
      else cur = candidate;
    });
    if (cur) out.push(cur);
    return out.map(function (seg, i) { return i ? ' ' + seg : seg; });
  }
  /* quote a segment the way Python would, picking the delimiter that needs no
     escaping, since the prompt itself carries double quotes around yes and no */
  function pyq(seg) {
    if (seg.indexOf("'") < 0) return "'" + seg + "'";
    return '"' + seg.replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
  }
  var SEGS = SYS ? wrapWords(SYS, 68) : [''];

  /* ---------- the source, one string per line ---------- */
  var LINES = [];
  LINES.push('import math');
  LINES.push('from openai import OpenAI');
  LINES.push('');
  LINES.push('SYSTEM = (');
  SEGS.forEach(function (seg) { LINES.push('    ' + pyq(seg)); });
  LINES.push(')');
  LINES.push('client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")');
  LINES.push('');
  LINES.push('def norm(t):            # strip spaces and quotes, then lower case');
  LINES.push('    return t.strip().strip(\'"\').lower()');
  LINES.push('');
  LINES.push('def best(top, want):    # largest logprob of any token normalising to want');
  LINES.push('    return max([c.logprob for c in top if norm(c.token) == want], default=-1e9)');
  LINES.push('');
  LINES.push('def score(instruct, query, document, threshold=0.5):');
  LINES.push('    user = f"<Instruct>: {instruct}\\n<Query>: {query}\\n<Document>: {document}"');
  LINES.push('    out = client.chat.completions.create(');
  LINES.push('        model="mistralai/Shieldstral-1.0-3B",');
  LINES.push('        messages=[{"role": "system", "content": SYSTEM},');
  LINES.push('                  {"role": "user", "content": user}],');
  LINES.push('        max_tokens=1, temperature=0.0,      # one token, no sampling');
  LINES.push('        logprobs=True, top_logprobs=20,     # hand back the candidate set');
  LINES.push('    )');
  LINES.push('    top = out.choices[0].logprobs.content[0].top_logprobs');
  LINES.push('    zy, zn = best(top, "yes"), best(top, "no")');
  LINES.push('    p = 1.0 / (1.0 + math.exp(zn - zy))     # exp(zy) / (exp(zy) + exp(zn))');
  LINES.push('    return p, p > threshold');

  var CODE = LINES.join('\n');
  var SERVE = 'vllm serve mistralai/Shieldstral-1.0-3B --max-model-len 32768';

  /* ---------- a very small Python tokeniser ----------
     Enough for this one listing. Strings and comments are matched before
     identifiers, left to right, so a hash inside a literal stays a literal. */
  var KW = {
    'import': 1, 'from': 1, 'def': 1, 'return': 1, 'for': 1, 'in': 1, 'if': 1,
    'else': 1, 'not': 1, 'and': 1, 'or': 1, 'None': 1, 'True': 1, 'False': 1,
    'lambda': 1, 'as': 1, 'with': 1, 'class': 1, 'while': 1, 'is': 1
  };
  var BI = { 'max': 1, 'min': 1, 'float': 1, 'int': 1, 'str': 1, 'len': 1, 'sum': 1 };

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function toks(line) {
    var re = new RegExp(
      '(#.*$)' +
      '|([frbu]{0,2}\'(?:[^\'\\\\]|\\\\.)*\')' +
      '|([frbu]{0,2}"(?:[^"\\\\]|\\\\.)*")' +
      '|(\\b\\d[\\d_]*\\.?\\d*(?:[eE][-+]?\\d+)?\\b)' +
      '|([A-Za-z_][A-Za-z_0-9]*)' +
      '|(\\s+)' +
      '|([\\s\\S])', 'g');
    var out = [], m;
    while ((m = re.exec(line)) !== null) {
      if (m[0] === '') { re.lastIndex++; continue; }
      if (m[1]) out.push(['cm', m[1]]);
      else if (m[2] || m[3]) out.push(['st', m[2] || m[3]]);
      else if (m[4]) out.push(['nu', m[4]]);
      else if (m[5]) {
        var w = m[5], after = line.slice(re.lastIndex), cls = 'id';
        if (KW[w]) cls = 'kw';
        else if (/^\s*\(/.test(after)) cls = BI[w] ? 'bi' : 'fn';
        else if (/^\s*=(?!=)/.test(after)) cls = 'ar';
        else if (/^[A-Z][A-Z_0-9]*$/.test(w)) cls = 'cn';
        out.push([cls, w]);
      } else if (m[6]) out.push(['ws', m[6]]);
      else out.push(['pu', m[7]]);
    }
    return out;
  }

  function lineHtml(line) {
    return toks(line).map(function (t) {
      return '<span class="t' + t[0] + '">' + esc(t[1]) + '</span>';
    }).join('');
  }

  /* ---------- which lines each button points at ---------- */
  function idxOf(subs) {
    var hit = [];
    LINES.forEach(function (l, i) {
      for (var k = 0; k < subs.length; k++) {
        if (l.indexOf(subs[k]) >= 0) { hit.push(i); return; }
      }
    });
    return hit;
  }
  var TAGS = [
    {
      k: 'max_tokens=1',
      lines: idxOf(['max_tokens']),
      d: 'One forward pass. The verdict is the first generated token, so nothing '
        + 'after it ever has to be produced, and temperature is irrelevant to a '
        + 'score read off the distribution.'
    },
    {
      k: 'top_logprobs=20',
      lines: idxOf(['top_logprobs']),
      d: 'The endpoint hands back the twenty highest scoring candidates at that '
        + 'one position. Ask for too few and a confident verdict can push the '
        + 'other side off the list entirely.'
    },
    {
      k: 'two logprobs',
      lines: idxOf(['def best', 'return max(', 'zy, zn =']),
      d: 'Normalise each candidate token, then take the largest logprob among '
        + 'those that read as yes and the largest among those that read as no. '
        + 'The rest of the vocabulary is discarded.'
    },
    {
      k: 'renormalise',
      lines: idxOf(['p = 1.0', 'return p,']),
      d: 'A softmax over exactly those two numbers, written in its stable form. '
        + 'The score is a probability of yes against no, not a probability over '
        + 'the vocabulary. Compare it to your threshold.'
    }
  ];

  /* ---------- build ---------- */
  root.classList.add('sc-s_run');

  var codeRows = LINES.map(function (l, i) {
    return '<div class="row" data-i="' + i + '">'
      + '<span class="ln" aria-hidden="true">' + (i + 1) + '</span>'
      + '<span class="cd">' + (l === '' ? '&nbsp;' : lineHtml(l)) + '</span>'
      + '</div>';
  }).join('');

  var tagBtns = TAGS.map(function (t, i) {
    return '<button type="button" class="tag" id="S_RUN-tag' + i + '" data-i="' + i + '"'
      + ' aria-pressed="false">' + esc(t.k) + '</button>';
  }).join('');

  root.appendChild(api.frag(
    '<div class="wrap">'

    + '<div class="hd">'
    +   '<span class="ey">running it</span>'
    +   '<span class="note">one endpoint call, two logprobs, one score</span>'
    + '</div>'

    + '<div class="serve">'
    +   '<span class="dollar" aria-hidden="true">$</span>'
    +   '<code class="scmd" id="S_RUN-serve">' + esc(SERVE) + '</code>'
    +   '<button type="button" class="cp" id="S_RUN-cpserve"'
    +     ' aria-label="Copy the vllm serve command to the clipboard">copy</button>'
    + '</div>'

    + '<div class="cardhd">'
    +   '<span class="fname">score.py</span>'
    +   '<span class="finfo">' + LINES.length + ' lines &middot; python'
    +     ' &middot; openai client</span>'
    +   '<button type="button" class="cp" id="S_RUN-cpcode"'
    +     ' aria-label="Copy the scoring helper source to the clipboard">copy</button>'
    + '</div>'

    + '<div class="code" id="S_RUN-code" tabindex="0" role="group"'
    +   ' aria-label="scoring helper source, Python">' + codeRows + '</div>'

    + '<div class="tags" id="S_RUN-tags" role="group"'
    +   ' aria-label="highlight the lines that matter">'
    +   tagBtns
    +   '<button type="button" class="tag clr" id="S_RUN-clr"'
    +     ' aria-label="Clear the line highlight">all lines</button>'
    + '</div>'
    + '<div class="desc" id="S_RUN-desc"></div>'

    + '<div class="gt">SYSTEM is built at render time from the paper&#39;s system prompt. '
    +   'The rest is deployment glue, not a paper result, and 0.5 is a placeholder '
    +   'threshold. Nothing here calls a model.</div>'

    + '</div>'
  ).firstChild);

  var rows = Array.prototype.slice.call(root.querySelectorAll('.row'));
  var btns = Array.prototype.slice.call(root.querySelectorAll('.tag[data-i]'));
  var clrBtn = root.querySelector('#S_RUN-clr');
  var descEl = root.querySelector('#S_RUN-desc');
  var codeEl = root.querySelector('#S_RUN-code');

  var sel = -1;
  var auto = true;

  function apply(i) {
    sel = i;
    var t = TAGS[i];
    var on = {};
    if (t) t.lines.forEach(function (n) { on[n] = 1; });
    rows.forEach(function (r, n) {
      var hot = !!(t && on[n]);
      r.classList.toggle('hi', hot);
      r.classList.toggle('dim', !!t && !hot);
    });
    btns.forEach(function (b, k) {
      b.setAttribute('aria-pressed', k === i ? 'true' : 'false');
      b.classList.toggle('sel', k === i);
    });
    clrBtn.classList.toggle('sel', !t);
    clrBtn.setAttribute('aria-pressed', t ? 'false' : 'true');
    descEl.textContent = t ? t.d
      : 'Four lines carry the whole trick. Pick one to bring it forward, or read '
        + 'the listing straight through.';
  }

  function takeOver() { auto = false; }

  btns.forEach(function (b) {
    b.addEventListener('click', function () {
      takeOver();
      apply(sel === +b.getAttribute('data-i') ? -1 : +b.getAttribute('data-i'));
    });
  });
  clrBtn.addEventListener('click', function () { takeOver(); apply(-1); });

  /* arrow keys walk the group, matching the rest of the guide */
  root.querySelector('#S_RUN-tags').addEventListener('keydown', function (e) {
    var all = btns.concat([clrBtn]);
    var cur = all.indexOf(document.activeElement);
    if (cur < 0) return;
    var to = -1;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') to = (cur - 1 + all.length) % all.length;
    else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') to = (cur + 1) % all.length;
    else if (e.key === 'Home') to = 0;
    else if (e.key === 'End') to = all.length - 1;
    if (to < 0) return;
    e.preventDefault();
    takeOver();
    all[to].focus();
    apply(to < btns.length ? to : -1);
  });

  /* ---------- copy ---------- */
  function flash(btn, ok) {
    var prev = btn.getAttribute('data-label') || btn.textContent;
    btn.setAttribute('data-label', prev);
    btn.textContent = ok ? 'copied' : 'press ctrl c';
    btn.classList.add('ok');
    if (btn._t) clearTimeout(btn._t);
    btn._t = setTimeout(function () {
      btn.textContent = prev;
      btn.classList.remove('ok');
    }, 1800);
  }
  function copy(txt, btn, node) {
    var settled = false;
    function done(ok) {
      if (settled) return;
      settled = true;
      /* if the clipboard is not available, at least leave the text selected so
         that the fallback instruction on the button is actionable */
      if (!ok && node && window.getSelection && document.createRange) {
        try {
          var r = document.createRange();
          r.selectNodeContents(node);
          var s = window.getSelection();
          s.removeAllRanges();
          s.addRange(r);
        } catch (err) { /* selection is a nicety, never a failure */ }
      }
      flash(btn, ok);
    }
    function legacy() {
      if (settled) return;
      var ta = document.createElement('textarea');
      ta.value = txt;
      ta.setAttribute('readonly', 'readonly');
      ta.style.cssText = 'position:absolute;left:-9999px;top:0;opacity:0';
      root.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
      root.removeChild(ta);
      done(ok);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      /* a clipboard promise that never settles would leave the button silent,
         so fall back on a timer as well as on rejection */
      var guard = setTimeout(legacy, 500);
      navigator.clipboard.writeText(txt).then(
        function () { clearTimeout(guard); done(true); },
        function () { clearTimeout(guard); legacy(); }
      );
    } else legacy();
  }
  root.querySelector('#S_RUN-cpserve').addEventListener('click', function () {
    copy(SERVE, this, root.querySelector('#S_RUN-serve'));
  });
  root.querySelector('#S_RUN-cpcode').addEventListener('click', function () {
    takeOver();
    copy(CODE, this, codeEl);
  });

  /* selecting the whole listing with the keyboard should not fight the buttons */
  codeEl.addEventListener('focus', takeOver);

  apply(-1);
  if (api.reduce) {
    descEl.textContent = 'Pick a button to bring its lines forward. '
      + 'Four lines carry the whole trick.';
  }

  /* auto walk the four callouts so a reader who only scrolls still sees them */
  var running = false, nextAt = null, step = -1;
  return {
    start: function () { running = true; nextAt = null; },
    stop: function () { running = false; },
    tick: function (t) {
      if (!running || !auto || api.reduce) return;
      if (nextAt === null) { nextAt = t + 1.6; return; }
      if (t < nextAt) return;
      nextAt = t + 3.0;
      step++;
      apply(step % TAGS.length);
    }
  };
};
