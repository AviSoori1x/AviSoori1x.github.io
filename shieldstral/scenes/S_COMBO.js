/* S_COMBO, Act 2.
   The randomisation space. The product is computed from the actual array
   lengths in SS.strictness, SS.formats and SS.queryTypes, never written down.
   Roll samples one of each, renders the training view that combination
   produces, and lights its cell in a coverage grid of the whole space. */
window.SCENES = window.SCENES || {};
window.SCENES['S_COMBO'] = function (root, api) {
  var SS = api.SS || {};
  var el = api.el;

  var STR = SS.strictness || [];
  var FMT = SS.formats || [];
  var QTY = SS.queryTypes || [];
  var nS = STR.length, nF = FMT.length, nQ = QTY.length;
  var TOTAL = nS * nF * nQ;

  /* the one exchange, held fixed so only the framing moves. Parsed out of the
     bracketed document in the paper figure rather than retyped. */
  var fig = (SS.fig2 || [])[0] || {};
  var PAIR = splitExchange(fig.document);

  function splitExchange(txt) {
    var lines = String(txt || '').split('\n');
    var ui = -1, ai = -1, i, t;
    for (i = 0; i < lines.length; i++) {
      t = lines[i].replace(/[[\]:*_#]/g, '').trim().toLowerCase();
      if (ui < 0 && (t === 'user' || t === 'human')) ui = i;
      else if (ui >= 0 && ai < 0 && (t === 'assistant' || t === 'ai' || t === 'bot')) ai = i;
    }
    if (ui < 0 || ai < 0) return { prompt: String(txt || ''), response: '' };
    return {
      prompt: lines.slice(ui + 1, ai).join(' ').trim(),
      response: lines.slice(ai + 1).join(' ').trim()
    };
  }

  /* split a filled template line into delimiter runs and content runs */
  function segs(line) {
    var marks = [], out = [], pos = 0;
    [PAIR.prompt, PAIR.response].forEach(function (s) {
      if (!s) return;
      var i = line.indexOf(s);
      if (i >= 0) marks.push({ i: i, s: s });
    });
    marks.sort(function (a, b) { return a.i - b.i; });
    marks.forEach(function (m) {
      if (m.i < pos) return;
      if (m.i > pos) out.push({ d: true, s: line.slice(pos, m.i) });
      out.push({ d: false, s: m.s });
      pos = m.i + m.s.length;
    });
    if (pos < line.length) out.push({ d: true, s: line.slice(pos) });
    if (!out.length) out.push({ d: true, s: line });
    return out;
  }

  var wrap = el('div', 'sc-s_combo');

  /* ---------------- header ---------------- */
  var hd = el('div', 'hd');
  hd.appendChild(el('span', 'eyy', 'randomisation space · one row, many views'));
  var ctr = el('div', 'ctr');

  var rollBtn = el('button', 'roll');
  rollBtn.type = 'button';
  rollBtn.appendChild(el('span', 'die', ''));
  rollBtn.appendChild(el('span', null, 'Roll a sample'));
  rollBtn.setAttribute('aria-label', 'Roll a new strictness tier, document format and query type');
  ctr.appendChild(rollBtn);

  var autoBtn = el('button', 'aut');
  autoBtn.type = 'button';
  autoBtn.setAttribute('aria-pressed', 'true');
  autoBtn.appendChild(el('span', 'dot'));
  autoBtn.appendChild(el('span', 'atx', 'auto on'));
  autoBtn.setAttribute('aria-label', 'Roll automatically every few seconds');
  ctr.appendChild(autoBtn);

  var resetBtn = el('button', 'rst', 'reset');
  resetBtn.type = 'button';
  resetBtn.setAttribute('aria-label', 'Clear the coverage grid');
  ctr.appendChild(resetBtn);

  hd.appendChild(ctr);
  wrap.appendChild(hd);

  /* ---------------- the product ---------------- */
  var prod = el('div', 'prod');
  var slots = [];

  function slot(count, dim) {
    var b = el('div', 'fac');
    b.appendChild(el('span', 'cnt', String(count)));
    b.appendChild(el('span', 'dim', dim));
    var v = el('span', 'val', '');
    var sub = el('span', 'sub', '');
    b.appendChild(v);
    b.appendChild(sub);
    prod.appendChild(b);
    slots.push({ box: b, val: v, sub: sub });
    return b;
  }

  slot(nS, 'strictness tiers');
  prod.appendChild(el('span', 'op', '×'));
  slot(nF, 'document formats');
  prod.appendChild(el('span', 'op', '×'));
  slot(nQ, 'query task types');
  prod.appendChild(el('span', 'op eqop', '='));

  var tot = el('div', 'tot');
  tot.appendChild(el('span', 'expr', nS + ' × ' + nF + ' × ' + nQ + ' ='));
  tot.appendChild(el('span', 'big', String(TOTAL)));
  tot.appendChild(el('span', 'tlab',
    'combinations of the pools in this data file, not a figure the paper reports'));
  var seenTx = el('span', 'seen', '');
  tot.appendChild(seenTx);
  prod.appendChild(tot);
  wrap.appendChild(prod);

  /* ---------------- the rendered training view ---------------- */
  var view = el('div', 'view');
  var vh = el('div', 'vh');
  vh.appendChild(el('span', 'vht', 'the training view this rolls into'));
  var rollTx = el('span', 'vhn', 'roll 0');
  vh.appendChild(rollTx);
  view.appendChild(vh);

  function field(name) {
    var f = el('div', 'fld');
    var top = el('div', 'ftop');
    top.appendChild(el('span', 'fname', name));
    var meta = el('span', 'fmeta', '');
    top.appendChild(meta);
    f.appendChild(top);
    var body = el('div', 'fbody');
    f.appendChild(body);
    view.appendChild(f);
    return { meta: meta, body: body };
  }

  var fInstr = field('instruction');
  var fQuery = field('query');
  var fDoc = field('document');
  wrap.appendChild(view);

  /* ---------------- coverage grid ---------------- */
  var cov = el('div', 'cov');
  var ch = el('div', 'ch');
  ch.appendChild(el('span', 'cht', 'coverage · every cell is one combination'));
  var covCount = el('span', 'chn', '');
  ch.appendChild(covCount);
  cov.appendChild(ch);

  var grid = el('div', 'covgrid');
  grid.setAttribute('role', 'img');
  grid.style.setProperty('--cols', String(nQ * nS));

  var groupHeads = [], strHeads = [], rowLabels = [], cells = [];

  grid.appendChild(el('span', 'corner', ''));
  QTY.forEach(function (q) {
    var g = el('span', 'gh', q.name || '');
    g.style.gridColumn = 'span ' + nS;
    g.title = (q.name || '') + ', ' + (q.sub || '');
    grid.appendChild(g);
    groupHeads.push(g);
  });

  grid.appendChild(el('span', 'corner', ''));
  QTY.forEach(function (q, qi) {
    STR.forEach(function (s, si) {
      var h = el('span', 'sh', String(s.level || '?').charAt(0));
      h.title = (s.level || '') + ' strictness, ' + (q.name || '') + ' query';
      grid.appendChild(h);
      strHeads.push({ node: h, s: si, q: qi });
    });
  });

  FMT.forEach(function (f, fi) {
    var lab = el('span', 'rl', f.family || '');
    lab.title = f.family || '';
    grid.appendChild(lab);
    rowLabels.push(lab);
    cells[fi] = [];
    QTY.forEach(function (q, qi) {
      STR.forEach(function (s, si) {
        var c = el('span', 'cell');
        c.title = (s.level || '') + ' · ' + (f.family || '') + ' · ' + (q.name || '');
        if (si === 0 && qi > 0) c.classList.add('gsep');
        grid.appendChild(c);
        cells[fi][qi * nS + si] = c;
      });
    });
  });
  cov.appendChild(grid);

  var leg = el('div', 'leg');
  STR.forEach(function (s) {
    leg.appendChild(el('span', 'lg', String(s.level || '?').charAt(0) + ' ' + (s.level || '')));
  });
  leg.appendChild(el('span', 'lg fill', 'filled cell = seen'));
  leg.appendChild(el('span', 'lg ring', 'ringed cell = this roll'));
  cov.appendChild(leg);
  wrap.appendChild(cov);

  /* ---------------- footer ---------------- */
  var foot = el('div', 'foot');
  foot.appendChild(el('b', null, 'Illustrative, not a sampled training record. '));
  foot.appendChild(document.createTextNode(
    'The grid counts only what this data file carries, ' + nS + ' tiers, ' + nF + ' formats, '
    + nQ + ' query types. The real pools are much larger: every processor holds its own '
    + 'instruction phrasings and per-category query variants, so the true space dwarfs these '
    + TOTAL + ' cells. The instruction is assembled from the tier row and the exchange is the '
    + 'one example pair from the paper figure, held fixed so only the framing moves.'));
  wrap.appendChild(foot);

  var live = el('span', 'sr');
  live.setAttribute('aria-live', 'polite');
  wrap.appendChild(live);

  root.appendChild(wrap);

  /* ---------------- state ---------------- */
  var seen = {}, seenN = 0, rolls = 0;
  var cur = { s: 0, f: 0, q: 0, qi: 0 };
  var hot = null;
  var auto = !api.reduce;
  var lastT = 0;

  function pick(n) { return Math.floor(Math.random() * n); }

  function roll() {
    if (!nS || !nF || !nQ) return;
    var s, f, q, guard = 0;
    do {
      s = pick(nS); f = pick(nF); q = pick(nQ);
      guard++;
    } while (guard < 24 && TOTAL > 1 && s === cur.s && f === cur.f && q === cur.q);
    var ex = (QTY[q].examples || []);
    cur = { s: s, f: f, q: q, qi: ex.length ? pick(ex.length) : 0 };
    rolls++;
    var key = s + '|' + f + '|' + q;
    if (!seen[key]) { seen[key] = 1; seenN++; }
    paint();
  }

  function paint() {
    var st = STR[cur.s] || {}, fm = FMT[cur.f] || {}, qt = QTY[cur.q] || {};

    /* slots */
    slots[0].val.textContent = st.level || '';
    slots[0].sub.textContent = st.domains || '';
    slots[1].val.textContent = fm.family || '';
    slots[1].sub.textContent = String(fm.tpl || '').split('\n')[0];
    slots[2].val.textContent = qt.name || '';
    slots[2].sub.textContent = qt.sub || '';

    seenTx.textContent = seenN + ' of ' + TOTAL + ' seen';
    covCount.textContent = seenN + ' / ' + TOTAL + ' · ' + rolls
      + (rolls === 1 ? ' roll' : ' rolls');
    rollTx.textContent = 'roll ' + rolls;

    /* instruction, composed from the tier row */
    fInstr.meta.textContent = 'composed from the ' + (st.level || '') + ' tier row';
    fInstr.body.textContent = '';
    fInstr.body.appendChild(el('span', 'ln',
      'Apply the ' + (st.level || '') + ' strictness tier. ' + (st.rationale || '') + '.'));
    fInstr.body.appendChild(el('span', 'ln soft',
      'Typical domains: ' + (st.domains || '') + '.'));

    /* query, drawn from that task type's pool */
    var ex = qt.examples || [];
    fQuery.meta.textContent = (qt.name || '') + ' pool, variant ' + (cur.qi + 1)
      + ' of ' + (ex.length || 0);
    fQuery.body.textContent = '';
    fQuery.body.appendChild(el('span', 'ln q', ex[cur.qi] || ''));

    /* document, the fixed exchange poured into the sampled template */
    fDoc.meta.textContent = (fm.family || '') + ' format, same exchange every roll';
    fDoc.body.textContent = '';
    var filled = String(fm.tpl || '')
      .replace(/\{prompt\}/g, PAIR.prompt)
      .replace(/\{response\}/g, PAIR.response);
    filled.split('\n').forEach(function (line) {
      var ln = el('span', 'ln');
      segs(line).forEach(function (p) {
        ln.appendChild(el('span', p.d ? 'dl' : 'ct', p.s));
      });
      fDoc.body.appendChild(ln);
    });

    /* coverage */
    if (hot) hot.classList.remove('now');
    var c = (cells[cur.f] || [])[cur.q * nS + cur.s];
    if (c) { c.classList.add('on', 'now'); hot = c; }
    rowLabels.forEach(function (n, i) { n.classList.toggle('hot', i === cur.f); });
    strHeads.forEach(function (h) {
      h.node.classList.toggle('hot', h.s === cur.s && h.q === cur.q);
    });
    groupHeads.forEach(function (g, i) { g.classList.toggle('hot', i === cur.q); });
    grid.setAttribute('aria-label', 'Coverage grid, ' + seenN + ' of ' + TOTAL
      + ' combinations seen so far.');

    live.textContent = 'Rolled ' + (st.level || '') + ' strictness, ' + (fm.family || '')
      + ' format, ' + (qt.name || '') + ' query. ' + seenN + ' of ' + TOTAL + ' seen.';
  }

  function setAuto(on) {
    auto = on;
    autoBtn.classList.toggle('off', !on);
    autoBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    autoBtn.querySelector('.atx').textContent = on ? 'auto on' : 'auto off';
  }

  rollBtn.addEventListener('click', function () { setAuto(false); roll(); });
  autoBtn.addEventListener('click', function () { setAuto(!auto); lastT = 0; });
  resetBtn.addEventListener('click', function () {
    seen = {}; seenN = 0; rolls = 0;
    cells.forEach(function (r) {
      r.forEach(function (c) { c.classList.remove('on', 'now'); });
    });
    hot = null;
    roll();
  });

  roll();
  if (api.reduce) setAuto(false);

  return {
    start: function () {},
    stop: function () {},
    tick: function (t) {
      if (!auto) return;
      if (!lastT) { lastT = t; return; }
      if (t - lastT > 3.4) { lastT = t; roll(); }
    }
  };
};
