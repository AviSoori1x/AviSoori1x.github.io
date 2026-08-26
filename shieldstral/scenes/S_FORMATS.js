window.SCENES = window.SCENES || {};

/* Act II, scene 12. Seven document formats.
   One exchange, taken from SS.fig2[0].document, poured through every family in
   SS.formats. The payload never changes. The scaffolding around it does, so the
   scaffolding is the part that is lit. Every string and every count here is read
   or computed from window.SS at runtime, nothing is typed in. */
window.SCENES['S_FORMATS'] = function (root, api) {
  var SS = api.SS || {};
  var FAM = (SS.formats || []).filter(function (f) { return f && f.tpl; });
  var SAMPLE = (SS.fig2 || [])[0] || {};
  var DOT = '·';
  var RET = '↵';

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function nbsp(s) { return s.replace(/ /g, '\u00a0'); }

  /* pull the prompt and the response back out of the paper's sample document,
     which is itself written in the bracketed family */
  function parsePair(doc) {
    var s = String(doc == null ? '' : doc);
    var m = s.match(/\[\s*user\s*\]([\s\S]*?)\[\s*assistant\s*\]([\s\S]*)/i);
    if (m) return { prompt: m[1].trim(), response: m[2].trim() };
    var lines = s.split('\n').filter(function (l) { return l.trim(); });
    return { prompt: (lines[0] || '').trim(), response: lines.slice(1).join(' ').trim() };
  }

  /* the literal pieces of a template, which is exactly its delimiter set */
  function lits(tpl) {
    return String(tpl).split(/(\{prompt\}|\{response\})/).filter(function (p) {
      return p && p !== '{prompt}' && p !== '{response}';
    });
  }
  function delimCount(tpl) { return lits(tpl).join('').replace(/\n/g, '').length; }
  function signature(tpl) { return lits(tpl).join('').replace(/\n/g, RET).trim(); }

  var PAIR = parsePair(SAMPLE.document);
  var CONTENT = PAIR.prompt.length + PAIR.response.length;
  var maxC = 1;
  FAM.forEach(function (f) { maxC = Math.max(maxC, delimCount(f.tpl)); });

  if (!FAM.length) {
    root.className = 'sc-s_formats';
    root.appendChild(api.frag('<div class="wrap"><p class="empty">'
      + 'no format families found in the data</p></div>').firstChild);
    return;
  }

  function dotSvg() {
    return '<svg class="dot" viewBox="0 0 16 16" aria-hidden="true">'
      + '<circle class="ring" cx="8" cy="8" r="6.2"></circle>'
      + '<circle class="core" cx="8" cy="8" r="3.1"></circle></svg>';
  }

  function railRow(f, i) {
    var c = delimCount(f.tpl);
    var pct = Math.max(5, Math.round(c / maxC * 100));
    return '<button type="button" class="row" id="S_FORMATS-f' + i + '"'
      + ' role="radio" aria-checked="false" tabindex="-1">'
      + '<span class="rtop">' + dotSvg()
      +   '<span class="rname">' + esc(f.family || ('family ' + (i + 1))) + '</span>'
      +   '<span class="rct">' + c + '</span></span>'
      + '<span class="sig">' + esc(signature(f.tpl)) + '</span>'
      + '<span class="bar"><i style="width:' + pct + '%"></i></span>'
      + '</button>';
  }

  root.className = 'sc-s_formats';
  root.appendChild(api.frag(
    '<div class="wrap">'

    + '<div class="hd">'
    +   '<span class="eyeb">seven document formats</span>'
    +   '<span class="hnote">one exchange, seven skins, the payload never moves</span>'
    + '</div>'

    + '<div class="grid">'

    +   '<div class="rails">'
    +     '<div class="rail" role="radiogroup" aria-label="choose a document format family">'
    +       FAM.map(railRow).join('')
    +     '</div>'
    +     '<p class="rcap">number and bar = delimiter characters this family adds. '
    +     'One family is sampled per training example.</p>'
    +   '</div>'

    +   '<div class="right">'
    +     '<div class="doc" id="S_FORMATS-doc">'
    +       '<div class="docbar">'
    +         '<span class="flab">document field</span>'
    +         '<span class="fmt" id="S_FORMATS-fmt"></span>'
    +       '</div>'
    +       '<div class="render" id="S_FORMATS-render"></div>'
    +       '<div class="key">'
    +         '<span class="kk kd"><i></i>delimiter, boxed and bold</span>'
    +         '<span class="kk kp"><i></i>prompt</span>'
    +         '<span class="kk kr"><i></i>response</span>'
    +       '</div>'
    +     '</div>'

    +     '<div class="stats">'
    +       '<div class="big">'
    +         '<span class="slab">delimiter characters</span>'
    +         '<span class="bnum" id="S_FORMATS-big" aria-live="polite">0</span>'
    +         '<span class="bsub" id="S_FORMATS-bsub"></span>'
    +       '</div>'
    +       '<div class="side">'
    +         '<div class="st"><i>content characters</i><b id="S_FORMATS-content">0</b>'
    +           '<u>byte-identical in all ' + FAM.length + '</u></div>'
    +         '<div class="st"><i>format families</i><b>' + FAM.length + '</b>'
    +           '<u>trained across every one</u></div>'
    +       '</div>'
    +     '</div>'
    +   '</div>'

    + '</div>'

    + '<div class="foot">'
    +   '<span class="gt">Illustrative render. The exchange is the text sample from the paper\'s '
    +   'worked example, re-rendered here through each listed template, not a live model call.</span>'
    +   '<span class="hint" id="S_FORMATS-hint">cycling on its own, click or use arrow keys to take over</span>'
    + '</div>'

    + '</div>'
  ).firstChild);

  var rows = FAM.map(function (f, i) { return root.querySelector('#S_FORMATS-f' + i); });
  var renderEl = root.querySelector('#S_FORMATS-render');
  var fmtEl = root.querySelector('#S_FORMATS-fmt');
  var bigEl = root.querySelector('#S_FORMATS-big');
  var bsubEl = root.querySelector('#S_FORMATS-bsub');
  var hint = root.querySelector('#S_FORMATS-hint');
  root.querySelector('#S_FORMATS-content').textContent = String(CONTENT);

  var cur = -1;
  var auto = true;

  /* build the rendered document. Literal template text becomes a lit delimiter
     chip, the two placeholders become the untouched payload. */
  function paint(tpl) {
    var parts = String(tpl).split(/(\{prompt\}|\{response\})/);
    var out = '';
    var k = 0;
    var anim = !api.reduce;
    parts.forEach(function (p) {
      if (!p) return;
      if (p === '{prompt}') {
        out += '<span class="slot sp1">' + esc(PAIR.prompt) + '</span>';
        return;
      }
      if (p === '{response}') {
        out += '<span class="slot sp2">' + esc(PAIR.response) + '</span>';
        return;
      }
      var lines = p.split('\n');
      lines.forEach(function (seg, li) {
        if (li) out += '<br>';
        if (!seg) return;
        var m = seg.match(/^(\s*)([\s\S]*?)(\s*)$/);
        if (m[1]) out += '<span class="ws">' + nbsp(m[1]) + '</span>';
        if (m[2]) {
          out += '<span class="dl' + (anim ? ' in' : '') + '"'
            + (anim ? ' style="animation-delay:' + (k * 0.055).toFixed(3) + 's"' : '')
            + '>' + esc(m[2]) + '</span>';
          k++;
        }
        if (m[3]) out += '<span class="ws">' + nbsp(m[3]) + '</span>';
      });
    });
    renderEl.innerHTML = '<div class="rin">' + out + '</div>';
  }

  function select(i, fromUser) {
    if (i === cur || !FAM[i]) return;
    var f = FAM[i];
    rows.forEach(function (r, j) {
      r.setAttribute('aria-checked', j === i ? 'true' : 'false');
      r.tabIndex = j === i ? 0 : -1;
    });
    paint(f.tpl);
    fmtEl.textContent = (i + 1) + ' of ' + FAM.length + ' ' + DOT + ' ' + (f.family || '');
    var c = delimCount(f.tpl);
    bigEl.textContent = String(c);
    bsubEl.textContent = 'wrapped around ' + CONTENT + ' characters of content that never change';
    if (fromUser && auto) {
      auto = false;
      hint.textContent = 'manual, arrow keys move through the families';
    }
    cur = i;
  }

  rows.forEach(function (r, i) {
    r.addEventListener('click', function () { select(i, true); });
  });
  root.querySelector('.rail').addEventListener('keydown', function (e) {
    var k = e.key, n = FAM.length, t = -1;
    if (k === 'ArrowDown' || k === 'ArrowRight') t = (cur + 1) % n;
    else if (k === 'ArrowUp' || k === 'ArrowLeft') t = (cur - 1 + n) % n;
    else if (k === 'Home') t = 0;
    else if (k === 'End') t = n - 1;
    if (t < 0) return;
    e.preventDefault();
    select(t, true);
    rows[t].focus();
  });

  select(0, false);

  var running = false, nextAt = null;
  return {
    start: function () { running = true; nextAt = null; },
    stop: function () { running = false; },
    tick: function (t) {
      if (!running || !auto || api.reduce) return;
      if (nextAt === null) { nextAt = t + 1.6; return; }
      if (t >= nextAt) {
        nextAt = t + 2.3;
        select((cur + 1) % FAM.length, false);
      }
    }
  };
};
