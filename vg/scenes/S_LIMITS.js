/* S_LIMITS, Act IV, the closing card.
   Three limitations straight out of window.SS.limitations, verbatim, no softening.
   Then the training context length from the model card. Then one caveat that is
   ours rather than the paper's: the endpoint returns a token and a probability,
   so when it says yes and it is wrong there is nothing attached to inspect. That
   last panel is grounded in window.SS.baselines, whose output column says Score
   for us and Reason + Label for two of the comparisons. Closes with
   window.SS.links and window.SS.coreContributors. */
window.SCENES = window.SCENES || {};
window.SCENES['S_LIMITS'] = function (root, api) {
  var SS = api.SS || {};
  var el = api.el;
  var svg = api.svg;

  var lims = SS.limitations || [];
  var links = SS.links || {};
  var crew = SS.coreContributors || [];
  var bases = SS.baselines || [];

  /* The training context length is a model card fact. It is not a field in
     window.SS, so it is labelled as coming from the card wherever it appears. */
  var CTX = '32k';

  /* what the baseline table says our output is, and who returns a rationale */
  var ourOut = null;
  var reasoners = [];
  bases.forEach(function (b) {
    if (/shieldstral/i.test(b.model || '')) ourOut = b.output || null;
    else if (/reason/i.test(b.output || '')) reasoners.push(b.model + ' ' + b.size);
  });

  var wrap = el('div', 'sc-s_limits');
  root.appendChild(wrap);

  /* ---------------- header ---------------- */
  var hd = el('div', 'hd');
  hd.appendChild(el('span', 'tag', 'limits'));
  hd.appendChild(el('span', 'hdsub',
    'the model card, unsoftened, and one caveat that follows from the architecture'));
  wrap.appendChild(hd);

  /* ---------------- the three from the card ---------------- */
  var grid = el('div', 'lims');
  grid.setAttribute('role', 'list');
  grid.setAttribute('aria-label',
    lims.length + ' limitations, quoted from the model card');
  wrap.appendChild(grid);

  lims.forEach(function (L, i) {
    var c = el('article', 'lim');
    c.setAttribute('role', 'listitem');
    var top = el('div', 'ltop');
    top.appendChild(el('span', 'n', (i < 9 ? '0' : '') + (i + 1)));
    top.appendChild(el('span', 'kw', 'stated'));
    c.appendChild(top);
    c.appendChild(el('h4', 'lt', L.t || 'n/a'));
    c.appendChild(el('p', 'ld', L.d || 'n/a'));
    grid.appendChild(c);
  });

  if (!lims.length) {
    grid.appendChild(el('p', 'ld', 'window.SS.limitations is empty'));
  }

  /* ---------------- the context length bar ---------------- */
  var ctx = el('section', 'ctx');
  ctx.setAttribute('aria-label',
    'training context length, ' + CTX + ' tokens, from the model card');

  var ch = el('div', 'ctxhead');
  ch.appendChild(el('span', 'k', 'trained context'));
  ch.appendChild(el('span', 'ksrc', 'model card, not in this guide data'));
  ctx.appendChild(ch);

  var bar = el('div', 'bar');
  var fill = el('div', 'fill');
  fill.appendChild(el('span', 'fb', CTX));
  fill.appendChild(el('span', 'fu', 'tokens seen in training'));
  var edge = el('div', 'edge');
  var beyond = el('div', 'beyond');
  beyond.appendChild(el('span', 'byt', 'the architecture takes more'));
  beyond.appendChild(el('span', 'byu', 'never trained there, treat as untested'));
  bar.appendChild(fill);
  bar.appendChild(edge);
  bar.appendChild(beyond);
  ctx.appendChild(bar);

  ctx.appendChild(el('p', 'ctxnote',
    'The hard edge under limitation '
    + (lims.length >= 3 ? '03' : String(lims.length))
    + '. A long document is not merely harder, it is off the end of what the weights '
    + 'were fitted on.'));
  wrap.appendChild(ctx);

  /* ---------------- the caveat that is ours ---------------- */
  var ours = el('section', 'ours');
  ours.setAttribute('aria-label',
    'a fourth caveat, not from the paper: the single token verdict has nothing '
    + 'attached to inspect');

  var oh = el('div', 'ohead');
  oh.appendChild(el('span', 'k', 'and one the card does not list'));
  oh.appendChild(el('span', 'pill', 'ours, not the paper'));
  ours.appendChild(oh);

  var ob = el('div', 'obody');

  /* left, the whole output */
  var tok = el('div', 'tok');
  tok.appendChild(el('div', 'tl', 'the entire output'));
  var big = el('div', 'big');
  big.appendChild(el('span', 'w', 'yes'));
  var car = el('i', 'car');
  car.setAttribute('aria-hidden', 'true');
  big.appendChild(car);
  tok.appendChild(big);
  tok.appendChild(el('div', 'ts',
    'one token, and the probability behind it'
    + (ourOut ? '. Output column in the baseline table: ' + ourOut : '')));
  ob.appendChild(tok);

  /* right, what a false positive review actually needs */
  var ask = el('div', 'ask');
  ask.appendChild(el('div', 'al', 'what you reach for when it fires and it is wrong'));

  var ROWS = [
    { q: 'how sure it was', got: 1, v: 'p(yes), renormalised over the two logits' },
    { q: 'which span of the document set it off', got: 0 },
    { q: 'which part of the policy it matched', got: 0 },
    { q: 'a rationale a human reviewer can read', got: 0 },
    { q: 'why it is not the other answer', got: 0 }
  ];

  ROWS.forEach(function (r) {
    var row = el('div', 'ar' + (r.got ? ' got' : ''));

    var g = svg('svg', { viewBox: '0 0 20 20', 'aria-hidden': 'true', focusable: 'false' });
    g.setAttribute('class', 'ag');
    if (r.got) {
      g.appendChild(svg('path', {
        d: 'M4 10.6 L8.4 15 L16 5.6', fill: 'none', stroke: 'currentColor',
        'stroke-width': '2.4', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'
      }));
    } else {
      g.appendChild(svg('path', {
        d: 'M4.5 10 H15.5', fill: 'none', stroke: 'currentColor',
        'stroke-width': '2.2', 'stroke-linecap': 'round'
      }));
    }
    row.appendChild(g);

    var body = el('div', 'ab');
    body.appendChild(el('span', 'aq', r.q));
    if (r.v) body.appendChild(el('span', 'av', r.v));
    row.appendChild(body);

    row.appendChild(el('span', 'as', r.got ? 'returned' : 'not returned'));
    ask.appendChild(row);
  });
  ob.appendChild(ask);
  ours.appendChild(ob);

  var oNote = 'A scorer, not an explainer, by design. It is still what you feel first in '
    + 'production, because a flagged item with no rationale cannot be triaged, only '
    + 'overridden.';
  if (reasoners.length) {
    oNote += ' ' + reasoners.length + ' of the ' + bases.length
      + ' baseline rows do return one: ' + reasoners.join(', ') + '.';
  }
  ours.appendChild(el('p', 'onote', oNote));
  wrap.appendChild(ours);

  /* ---------------- links and credit ---------------- */
  var foot = el('section', 'foot');

  var lk = el('div', 'links');
  lk.setAttribute('aria-label', 'where the model and the report live');
  var ORDER = [
    ['paper', 'the report'],
    ['pdf', 'the pdf'],
    ['hf', 'the weights'],
    ['blog', 'the announcement']
  ];
  ORDER.forEach(function (p) {
    var url = links[p[0]];
    if (!url) return;
    var a = document.createElement('a');
    a.className = 'lnk';
    a.href = url;
    a.rel = 'noopener';
    a.target = '_blank';
    a.appendChild(el('span', 'lkk', p[1]));
    a.appendChild(el('span', 'lku', String(url).replace(/^https?:\/\//, '')));
    lk.appendChild(a);
  });
  foot.appendChild(lk);

  var crewBox = el('div', 'crew');
  var cl = el('div', 'crewlab');
  cl.appendChild(el('span', 'k', 'core contributors'));
  cl.appendChild(el('span', 'cn', String(crew.length)));
  crewBox.appendChild(cl);

  var names = el('p', 'names');
  crew.forEach(function (nm, i) {
    if (i) names.appendChild(el('span', 'sep', '·'));
    var mine = /avinash/i.test(nm);
    var s = el('span', mine ? 'nm me' : 'nm', nm);
    if (mine) s.setAttribute('aria-describedby', 'S_LIMITS-mecap');
    names.appendChild(s);
  });
  crewBox.appendChild(names);

  var cap = el('p', 'mecap', 'The underlined name wrote this guide.');
  cap.id = 'S_LIMITS-mecap';
  crewBox.appendChild(cap);
  foot.appendChild(crewBox);
  wrap.appendChild(foot);

  /* ---------------- provenance ---------------- */
  wrap.appendChild(el('p', 'honest',
    'The numbered entries, the links and the names are read from the guide data, and the '
    + 'limitations are quoted word for word. The ' + CTX + ' context is a model card fact '
    + 'that is not a field in that data, so it is typed here and labelled. The fourth '
    + 'caveat is our reading of the interface, not a claim in the report, and the five '
    + 'rows under it are illustrative of a review workflow rather than measured.'));

  /* ---------------- motion ---------------- */
  return {
    start: function () {
      wrap.classList.remove('run');
      void wrap.offsetWidth;
      wrap.classList.add('run');
    },
    stop: function () {
      wrap.classList.remove('run');
      car.style.opacity = '';
    },
    tick: function (t) {
      car.style.opacity = (Math.floor(t * 1.5) % 2) ? '.14' : '1';
    }
  };
};
