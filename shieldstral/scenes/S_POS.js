window.SCENES = window.SCENES || {};

/* S_POS, act 2, beat 14. Positives at three granularities.
   One harmful document, asked three ways: coarse binary, category-specific,
   target-group-specific. All three answer yes, so a single piece of content
   becomes three positive training rows.
   Document, instruct, the category-specific query and its label are read from
   window.SS.fig3. The binary query is read from window.SS.queryTypes. The
   demographic form of the narrow rung is also read from window.SS.queryTypes.
   Counts come from window.SS.headline. Nothing is hardcoded. */
window.SCENES['S_POS'] = function (root, api) {
  var SS = api.SS || {};
  var F = SS.fig3 || {};
  var P = F.positive || {};
  var head = SS.headline || {};
  var qt = SS.queryTypes || [];

  var DOC = F.document == null ? '' : String(F.document);
  var INS = F.instruct == null ? '' : String(F.instruct);
  var leaves = head.trainLeaves;
  var supers = head.trainSupers;

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function byName(n) {
    for (var i = 0; i < qt.length; i++) if (qt[i] && qt[i].name === n) return qt[i];
    return null;
  }
  function ex(o, i) {
    return (o && o.examples && o.examples[i]) ? String(o.examples[i]) : null;
  }

  var binQ = ex(byName('Binary'), 0);
  /* the paper's own category-specific pool carries one example that names a
     target group, which is the general shape of the third rung */
  var groupForm = ex(byName('Category-specific'), 1);

  var DOT = '·';

  var RUNGS = [
    {
      id: 'binary',
      kind: 'coarse binary',
      q: binQ,
      label: 'yes',
      tag: 'query from the data file, label from the construction rule',
      scope: (leaves != null) ? ('all ' + leaves + ' leaf categories') : 'the whole policy',
      why: 'The broadest rung. It names no category, so a violation anywhere in the policy '
        + 'answers yes. This is the row that has to hold when a product ships one blunt '
        + 'safe or unsafe switch.'
    },
    {
      id: 'category',
      kind: 'category-specific',
      q: P.query,
      label: P.label,
      tag: 'query and label are the paper example'
        + (P.role ? (', role ' + DOT + ' ' + P.role) : ''),
      scope: (leaves != null) ? ('1 of ' + leaves + ' leaf categories') : 'one leaf category',
      why: 'Names the one category this document actually violates. Its sibling categories, '
        + 'asked of the very same text, answer no, which is where the hard negatives come from.'
    },
    {
      id: 'group',
      kind: 'target-group-specific',
      q: 'Does this text encourage deliberate physical harm against the specific people it '
        + 'identifies as targets?',
      label: 'yes',
      tag: 'phrasing composed for this figure',
      scope: 'one target inside that leaf',
      why: 'Emitted only where the source annotation records who the content is aimed at. '
        + (groupForm
          ? ('Where that annotation is demographic the rung takes this form: ' + groupForm)
          : 'The paper generates it from the target-group annotation on the source sample.')
    }
  ].filter(function (r) { return r.q; });

  var N = RUNGS.length;

  function pips(k) {
    var s = '', i;
    for (i = 0; i < N; i++) s += '<i' + (i <= k ? ' class="on"' : '') + '></i>';
    return '<span class="pips" aria-hidden="true">' + s + '</span>';
  }

  function tick() {
    return '<svg class="tick" viewBox="0 0 20 20" aria-hidden="true">'
      + '<path d="M4 10.4 L8.2 14.6 L16 6.1"></path></svg>';
  }

  function rowHtml(r, i) {
    var lab = String(r.label == null ? '' : r.label).toLowerCase();
    return '<button type="button" class="row" id="S_POS-row-' + i + '"'
      + ' role="radio" aria-checked="false" tabindex="-1">'
      + '<span class="gut">'
      +   '<span class="idx">' + ('0' + (i + 1)) + '</span>'
      +   pips(i)
      +   '<span class="spec">' + (i + 1) + ' / ' + N + '</span>'
      + '</span>'
      + '<span class="mid">'
      +   '<span class="kindline">'
      +     '<span class="kind">' + esc(r.kind) + '</span>'
      +     '<span class="scope">' + esc(r.scope) + '</span>'
      +   '</span>'
      +   '<span class="q">' + esc(r.q) + '</span>'
      +   '<span class="tagline">' + esc(r.tag) + '</span>'
      + '</span>'
      + '<span class="ans">'
      +   tick()
      +   '<b>' + esc(lab || 'yes') + '</b>'
      +   '<i>label</i>'
      + '</span>'
      + '</button>';
  }

  var sumnote = (leaves != null && supers != null)
    ? ('The training taxonomy carries ' + leaves + ' leaf categories under ' + supers
       + ' super classes. The coarse rung covers every one of them, the narrow rung covers '
       + 'a slice of a single leaf, and the document underneath does not change.')
    : 'The coarse rung covers the whole policy, the narrow rung covers a slice of one '
      + 'category, and the document underneath does not change.';

  root.className = 'sc-s_pos';
  root.appendChild(api.frag(
    '<div class="wrap">'

    + '<div class="hd">'
    +   '<span class="eyebrow">positives</span>'
    +   '<span class="hnote">one harmful document, three questions, every answer yes</span>'
    + '</div>'

    + (INS
      ? ('<div class="ins"><span class="klab">instruct</span>'
         + '<span class="itxt">' + esc(INS) + '</span>'
         + '<span class="pin">shared</span></div>')
      : '')

    + '<div class="doc">'
    +   '<div class="docbar">'
    +     '<span class="klab hot">document</span>'
    +     '<span class="pin">one source row ' + DOT + ' reused ' + N + ' times</span>'
    +   '</div>'
    +   '<p class="doctxt">' + esc(DOC) + '</p>'
    + '</div>'

    + '<div class="stem" aria-hidden="true"></div>'

    + '<div class="rows" id="S_POS-rows" role="radiogroup"'
    +   ' aria-label="three query granularities asked of the same document">'
    +   RUNGS.map(rowHtml).join('')
    + '</div>'

    + '<div class="note" id="S_POS-note"><span class="klab">why this rung</span>'
    +   '<span class="ntxt" id="S_POS-ntxt"></span></div>'

    + '<div class="sum">'
    +   '<span class="big" id="S_POS-count">' + N + '</span>'
    +   '<span class="biglab">positive rows<br>from one document</span>'
    +   '<span class="sumnote">' + esc(sumnote) + '</span>'
    + '</div>'

    + '<div class="foot">'
    +   '<span class="gt">Labels are the pipeline ground truth, not a live model call. The '
    +   'target-group query is written for this figure, since the paper prints no target-group '
    +   'example for this document.</span>'
    +   '<span class="hint" id="S_POS-hint">stepping on its own, click or use arrow keys</span>'
    + '</div>'

    + '</div>'
  ).firstChild);

  var rowEls = [];
  for (var i = 0; i < N; i++) rowEls.push(root.querySelector('#S_POS-row-' + i));
  var countEl = root.querySelector('#S_POS-count');
  var ntxt = root.querySelector('#S_POS-ntxt');
  var hint = root.querySelector('#S_POS-hint');

  var revealed = N;
  var cur = -1;
  var auto = true;
  var safety = 0;

  function setRevealed(n) {
    revealed = n;
    for (var k = 0; k < N; k++) rowEls[k].classList.toggle('lit', k < n);
    countEl.textContent = String(n);
    countEl.classList.toggle('zero', n === 0);
  }

  function select(k, fromUser) {
    if (k < 0 || k >= N) return;
    for (var j = 0; j < N; j++) {
      var on = j === k;
      rowEls[j].classList.toggle('sel', on);
      rowEls[j].setAttribute('aria-checked', on ? 'true' : 'false');
      rowEls[j].tabIndex = on ? 0 : -1;
    }
    ntxt.textContent = RUNGS[k].why;
    cur = k;
    if (fromUser && auto) {
      auto = false;
      clearTimeout(safety);
      setRevealed(N);
      hint.textContent = 'manual ' + DOT + ' arrow keys move between rungs';
    }
  }

  rowEls.forEach(function (b, k) {
    b.addEventListener('click', function () { select(k, true); });
  });
  root.querySelector('#S_POS-rows').addEventListener('keydown', function (e) {
    var k = e.key, n = cur;
    if (k === 'ArrowUp' || k === 'ArrowLeft') n = (cur + N - 1) % N;
    else if (k === 'ArrowDown' || k === 'ArrowRight') n = (cur + 1) % N;
    else if (k === 'Home') n = 0;
    else if (k === 'End') n = N - 1;
    else return;
    e.preventDefault();
    select(n, true);
    rowEls[n].focus();
  });

  setRevealed(N);
  select(0, false);

  /* the step-through, so a reader who only scrolls still sees the fan out.
     The first rung is always up, and a timer backstops the frame pump so the
     figure can never sit in a half-built state. */
  var running = false, nextAt = null;
  return {
    start: function () {
      running = true;
      nextAt = null;
      clearTimeout(safety);
      if (!auto) return;
      setRevealed(1);
      select(0, false);
      safety = setTimeout(function () {
        if (auto && revealed < N) { setRevealed(N); select(N - 1, false); }
      }, 5200);
    },
    stop: function () { running = false; clearTimeout(safety); },
    tick: function (t) {
      if (!running || !auto || api.reduce) return;
      if (nextAt === null) { nextAt = t + 1.1; return; }
      if (t < nextAt) return;
      if (revealed < N) {
        setRevealed(revealed + 1);
        select(revealed - 1, false);
        nextAt = t + (revealed < N ? 1.3 : 2.5);
      } else {
        nextAt = t + 2.5;
        select((cur + 1) % N, false);
      }
    }
  };
};
