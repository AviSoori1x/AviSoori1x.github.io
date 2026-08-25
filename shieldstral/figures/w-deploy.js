(function () {
  var ID = 'w-deploy';
  var root = document.getElementById(ID);
  if (!root) return;
  var SS = window.SS;
  if (!SS) return;

  function pick(id) { return document.getElementById(ID + '-' + id); }

  function el(tag, cls, txt) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt != null) n.textContent = txt;
    return n;
  }

  /* ---- python source, built from SS so it cannot drift from the card ---- */

  function pyLit(s) {
    var t = String(s).replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/\t/g, '\\t');
    if (t.indexOf('"') === -1) return '"' + t + '"';
    if (t.indexOf("'") === -1) return "'" + t + "'";
    return '"' + t.replace(/"/g, '\\"') + '"';
  }

  function num(v, fallback) {
    return typeof v === 'number' && isFinite(v) ? String(v) : fallback;
  }

  function wrapKeep(s, width) {
    var words = String(s).split(' ');
    var out = [];
    var cur = '';
    for (var i = 0; i < words.length; i++) {
      var piece = words[i] + (i < words.length - 1 ? ' ' : '');
      if (cur && cur.length + piece.length > width) {
        out.push(cur);
        cur = piece;
      } else {
        cur += piece;
      }
    }
    if (cur) out.push(cur);
    return out;
  }

  function tokSet(word) {
    var forms = [word, word + '.', '"' + word + '"', "'" + word + "'"];
    var lits = [];
    for (var i = 0; i < forms.length; i++) lits.push(pyLit(forms[i]));
    return '{' + lits.join(', ') + '}';
  }

  var hfUrl = (SS.links && SS.links.hf) || '';
  var modelId = hfUrl.replace(/^https?:\/\/[^/]+\//, '').replace(/\/+$/, '');
  var sysLines = wrapKeep(SS.systemPrompt || '', 40).map(function (chunk) {
    return '    ' + pyLit(chunk);
  });

  var lines = [
    'import math, os',
    'from openai import OpenAI',
    '',
    'client = OpenAI(',
    '    base_url="http://localhost:8000/v1",',
    '    api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),',
    ')',
    '',
    'MODEL = ' + pyLit(modelId),
    'SYSTEM = ('
  ].concat(sysLines, [
    ')',
    '',
    'YES = ' + tokSet('yes'),
    'NO = ' + tokSet('no'),
    '',
    '',
    'def unsafe_score(instruct, query, document,',
    '                 threshold=0.5):',
    '    # Returns P(yes): the Query is answered yes',
    '    # about the Document. Phrase the Query so',
    '    # that yes is the outcome you want caught.',
    '    user = (',
    '        f"<Instruct>: {instruct}\\n\\n"',
    '        f"<Query>: {query}\\n\\n"',
    '        f"<Document>: {document}"',
    '    )',
    '    out = client.chat.completions.create(',
    '        model=MODEL,',
    '        messages=[',
    '            {"role": "system", "content": SYSTEM},',
    '            {"role": "user", "content": user},',
    '        ],',
    '        max_tokens=1,  # one forward pass',
    '        temperature=0.0,',
    '        logprobs=True,',
    '        top_logprobs=20,',
    '    )',
    '    first = out.choices[0].logprobs.content[0]',
    '    top = first.top_logprobs',
    '',
    '    def best_of(vocab):  # -10.0 floor, as in the card',
    '        z = [t.logprob for t in top',
    '             if t.token.strip().lower() in vocab]',
    '        return max(z + [-10.0])',
    '',
    '    z_yes, z_no = best_of(YES), best_of(NO)',
    '    m = max(z_yes, z_no)  # stable two-way softmax',
    '    p_yes = math.exp(z_yes - m)',
    '    score = p_yes / (p_yes + math.exp(z_no - m))',
    '    return score, score > threshold'
  ]);

  var pySource = lines.join('\n');
  var serveParts = ['vllm serve', modelId, '--max-model-len 32768'];
  var serveCmd = serveParts.join(' ');

  /* ---- tiny python highlighter, one line at a time ---- */

  var KW = /^(?:import|from|def|return|if|elif|else|for|in|not|and|or|as|with|class|lambda|while|try|except|pass|is|None|True|False)$/;
  var BI = /^(?:max|min|len|float|int|str|list|dict|set|sum|print|range|abs)$/;
  var CONST = /^[A-Z][A-Z_0-9]*$/;
  var TOK = /(#.*$)|((?:[fFrRbB]{0,2})(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'))|([A-Za-z_][A-Za-z_0-9]*)|(\d+(?:\.\d+)?(?:[eE]-?\d+)?)/g;

  function paint(line, out) {
    if (!line) {
      out.appendChild(document.createTextNode(' '));
      return;
    }
    TOK.lastIndex = 0;
    var last = 0;
    var prev = '';
    var m;
    while ((m = TOK.exec(line)) !== null) {
      if (m.index > last) out.appendChild(document.createTextNode(line.slice(last, m.index)));
      var t = m[0];
      var cls = null;
      if (m[1]) {
        cls = 'dp-t-com';
      } else if (m[2]) {
        cls = 'dp-t-str';
      } else if (m[3]) {
        if (KW.test(t)) cls = 'dp-t-kw';
        else if (prev === 'def') cls = 'dp-t-fn';
        else if (CONST.test(t)) cls = 'dp-t-const';
        else if (BI.test(t)) cls = 'dp-t-bi';
        prev = t;
      } else if (m[4]) {
        cls = 'dp-t-num';
      }
      out.appendChild(cls ? el('span', cls, t) : document.createTextNode(t));
      last = m.index + t.length;
    }
    if (last < line.length) out.appendChild(document.createTextNode(line.slice(last)));
  }

  var body = pick('src');
  if (body) {
    for (var i = 0; i < lines.length; i++) {
      var row = el('div', 'dp-cl');
      var gut = el('span', 'dp-gut', String(i + 1));
      gut.setAttribute('aria-hidden', 'true');
      var src = el('code', 'dp-src');
      paint(lines[i], src);
      row.appendChild(gut);
      row.appendChild(src);
      body.appendChild(row);
    }
  }

  var serveNode = pick('serve');
  if (serveNode) {
    for (var sp = 0; sp < serveParts.length; sp++) {
      if (sp > 0) serveNode.appendChild(document.createTextNode(' '));
      serveNode.appendChild(el('span', 'dp-nb', serveParts[sp]));
    }
  }

  /* ---- copy ---- */

  var status = pick('status');

  function legacyCopy(text) {
    try {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.top = '-2000px';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    } catch (e) {
      return false;
    }
  }

  function wire(btn, getText, label) {
    if (!btn) return;
    var timer = null;
    function settle(ok) {
      btn.textContent = ok ? 'Copied' : 'Copy failed';
      btn.classList.toggle('is-done', ok);
      if (status) status.textContent = ok ? label + ' copied to the clipboard.' : 'Copy failed. Select the text and copy it manually.';
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(function () {
        btn.textContent = 'Copy';
        btn.classList.remove('is-done');
        if (status) status.textContent = '';
      }, 1900);
    }
    btn.addEventListener('click', function () {
      var text = getText();
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          settle(true);
        }, function () {
          settle(legacyCopy(text));
        });
      } else {
        settle(legacyCopy(text));
      }
    });
  }

  wire(pick('copy-py'), function () { return pySource; }, 'Python helper');
  wire(pick('copy-sh'), function () { return serveCmd; }, 'Serving command');

  /* ---- limitations ---- */

  var lims = pick('lims');
  var limData = SS.limitations || [];
  if (lims) {
    lims.setAttribute('role', 'list');
    for (var j = 0; j < limData.length; j++) {
      var li = el('li', 'dp-lim');
      li.setAttribute('role', 'listitem');
      var n = el('span', 'dp-n', (j + 1 < 10 ? '0' : '') + (j + 1));
      n.setAttribute('aria-hidden', 'true');
      var txt = el('div', 'dp-lim-txt');
      txt.appendChild(el('div', 'dp-lim-t', limData[j].t));
      txt.appendChild(el('p', 'dp-lim-d', limData[j].d));
      li.appendChild(n);
      li.appendChild(txt);
      lims.appendChild(li);
    }
  }

  /* ---- our caveat, with every number read from SS ---- */

  var ownD = pick('own-d');
  if (ownD) {
    var H = SS.headline || {};
    var reasoner = null;
    var bl = SS.baselines || [];
    for (var b = 0; b < bl.length; b++) {
      if (/^GPT-OSS-Safeguard/.test(bl[b].model || '')) reasoner = bl[b];
    }
    ownD.appendChild(document.createTextNode(
      'The verdict is a single token, so a flag arrives with nothing attached. There is no rationale to read back, and one call returns no per category breakdown, though the model card suggests issuing one query per policy to get that. '
    ));
    if (reasoner && H.adaptabilityBest != null && H.adaptabilityF1 != null) {
      ownD.appendChild(document.createTextNode('On the adaptability benchmark the best score belongs to '));
      ownD.appendChild(el('span', 'dp-nb', reasoner.model + ' ' + reasoner.size));
      ownD.appendChild(document.createTextNode(
        ', which writes out a reasoning trace before it answers. It scores ' +
        num(H.adaptabilityBest, '') + ' F1 there against Shieldstral at ' + num(H.adaptabilityF1, '') +
        '. Shieldstral gets its number from ' + (H.params || '') +
        ' parameters and one forward pass, and the paper attributes the lower inference efficiency of GPT-OSS-Safeguard and Nemotron-3.5-Safety to their reasoning traces.'
      ));
    }
  }

  /* ---- links ---- */

  function arrow() {
    var ns = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('class', 'dp-arw');
    svg.setAttribute('viewBox', '0 0 10 10');
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    var p = document.createElementNS(ns, 'path');
    p.setAttribute('d', 'M2.2 7.8 L7.8 2.2 M3.6 2.2 L7.8 2.2 L7.8 6.4');
    svg.appendChild(p);
    return svg;
  }

  var linkBox = pick('links');
  var linkSpec = [['hf', 'Model card'], ['paper', 'Paper'], ['blog', 'Blog post']];
  if (linkBox && SS.links) {
    for (var k = 0; k < linkSpec.length; k++) {
      var url = SS.links[linkSpec[k][0]];
      if (!url) continue;
      var a = el('a', 'dp-link');
      a.href = url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      var t = el('span', 'dp-link-t', linkSpec[k][1]);
      t.appendChild(arrow());
      t.appendChild(el('span', 'dp-sr', ' (opens in a new tab)'));
      a.appendChild(t);
      a.appendChild(el('span', 'dp-link-u', url.replace(/^https?:\/\//, '').replace(/^www\./, '')));
      linkBox.appendChild(a);
    }
  }

  /* ---- core contributors ---- */

  var people = pick('people');
  var names = SS.coreContributors || [];
  if (people) {
    for (var p2 = 0; p2 < names.length; p2++) {
      if (p2 > 0) people.appendChild(document.createTextNode(', '));
      if (/Avinash/.test(names[p2])) {
        var me = el('span', 'dp-me', names[p2]);
        people.appendChild(me);
        people.appendChild(el('span', 'dp-sr', ' (author of this page)'));
      } else {
        people.appendChild(document.createTextNode(names[p2]));
      }
    }
  }
})();
