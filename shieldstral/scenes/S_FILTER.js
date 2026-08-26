window.SCENES = window.SCENES || {};

/* S_FILTER, act 2, beat 16.
   Two clean-up mechanisms stacked in one figure.
   01 rebalance: contrastive construction gives one yes and up to (leaves - 1) no rows per
      annotated document, so each positive is duplicated with a freshly paraphrased
      <Instruct> and <Query>. The duplicate row slides in and the ratio numeral halves.
   02 cross-validate: an open-source LLM re-reads every row at the binary and per-category
      level, and any disagreement is struck out and dropped rather than corrected.
   Every class inside is prefixed sf- so the page level styles cannot reach in. */
window.SCENES['S_FILTER'] = function (root, api) {
  var SS = api.SS || {};
  var head = SS.headline || {};
  var fig3 = SS.fig3 || {};
  var fig4 = SS.fig4 || {};
  var fig2 = SS.fig2 || [];
  var qtypes = SS.queryTypes || [];

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ---------------- numbers, all read from the data file ---------------- */

  var leaves = head.trainLeaves;                                  /* 73 leaf categories */
  // Pairing one document against every other leaf would be an invented ratio. The
  // paper says any absent category CAN serve as a negative and that positives are
  // duplicated k times, but reports no negatives-per-positive figure.
  var negPer = null;
  var afterPer = (negPer != null) ? (negPer / 2) : null;          /* after one duplication */

  function ratio(n) {
    if (n == null) return null;
    var v = (Math.round(n * 10) / 10);
    return (v % 1 === 0 ? String(v) : v.toFixed(1)) + ':1';
  }

  /* ---------------- leaf names, looked up out of the taxonomy ---------------- */

  var LEAF = {};
  (SS.evalTaxonomy || []).forEach(function (sc) {
    (sc.subs || []).forEach(function (sub) {
      (sub.leaves || []).forEach(function (lf) { LEAF[lf.id] = lf.name; });
    });
  });
  function cat(id) { return LEAF[id] || id; }

  /* ---------------- the paraphrase pool ----------------
     The duplicate row keeps the same document and answer and swaps the wording. Every
     alternative below is another phrasing of the same ask taken from the paper's own
     worked examples, so nothing here is invented copy. */

  var SRC = {
    instruct: fig3.instruct,
    query: (fig3.positive || {}).query,
    answer: (fig3.positive || {}).label || 'yes'
  };
  var INSTRUCTS = [fig4.instruct, (fig2[0] || {}).instruct].filter(Boolean);
  var QUERIES = [
    fig4.query,
    (fig2[0] || {}).query,
    ((qtypes[0] || {}).examples || [])[0]
  ].filter(Boolean);

  /* ---------------- the rows the LLM re-reads ---------------- */

  var ROWS = [
    { doc: 'Put the request in writing and keep copies.',
      dl: 'safe', dc: null, ml: 'safe', mc: null },
    { doc: 'They are all lazy and dishonest, every one.',
      dl: 'unsafe', dc: cat('CAT013'), ml: 'unsafe', mc: cat('CAT013') },
    { doc: 'Try a slim jim, or call roadside assistance.',
      dl: 'unsafe', dc: cat('CAT017'), ml: 'safe', mc: null },
    { doc: 'Here is an invoice from a supplier you never used.',
      dl: 'unsafe', dc: cat('CAT017'), ml: 'unsafe', mc: cat('CAT019') },
    { doc: 'Please contact a crisis line, you deserve support.',
      dl: 'safe', dc: null, ml: 'safe', mc: null },
    { doc: 'A cloned login page wired to a credential relay.',
      dl: 'safe', dc: null, ml: 'unsafe', mc: cat('CAT026') }
  ];

  ROWS.forEach(function (r) {
    r.binBad = (r.dl !== r.ml);
    r.catBad = ((r.dc || '') !== (r.mc || ''));
    r.drop = r.binBad || r.catBad;
    r.why = r.binBad ? 'binary' : (r.catBad ? 'category' : '');
  });
  var nDrop = ROWS.filter(function (r) { return r.drop; }).length;
  var nKeep = ROWS.length - nDrop;

  /* ---------------- shell ---------------- */

  root.classList.add('sc-s_filter');
  var wrap = api.el('div', 'sf-wrap');

  var ARROW = '<svg class="sf-arrow" viewBox="0 0 52 20" aria-hidden="true" focusable="false">'
    + '<path d="M2 10h34" stroke="currentColor" stroke-width="1.4" fill="none"'
    + ' stroke-dasharray="3 4" stroke-linecap="round"/>'
    + '<path d="M33 4.5 43 10l-10 5.5" stroke="currentColor" stroke-width="1.7" fill="none"'
    + ' stroke-linejoin="round" stroke-linecap="round"/></svg>';

  /* the dot field, one cell per row generated off a single annotated document */
  var dotHtml = '';
  var nDots = (negPer != null) ? (negPer + 2) : 0;   /* 1 yes + 1 duplicate + negPer no */
  for (var d = 0; d < nDots; d++) {
    var kls = (d === 0) ? 'sf-dot yes' : (d === 1 ? 'sf-dot dup' : 'sf-dot');
    dotHtml += '<i class="' + kls + '"></i>';
  }

  var ratioHtml = (negPer == null)
    ? '<div class="sf-rsub">The training taxonomy size is not in the data file, '
      + 'so the ratio is left out.</div>'
    : '<div class="sf-kick">negatives per positive, off one document</div>'
      + '<div class="sf-rline">'
      + '<span class="sf-before">' + esc(ratio(negPer)) + '</span>'
      + '<span class="sf-arrowwrap">' + ARROW + '</span>'
      + '<span class="sf-after">' + esc(ratio(afterPer)) + '</span>'
      + '</div>'
      + '<div class="sf-rsub">The annotation asserts one category, so one query answers '
      + '<b>yes</b> and the other ' + esc(String(negPer)) + ' leaf categories answer <b>no</b> '
      + 'on the same text. Duplicating the positive halves the skew, it does not invert it.'
      + '</div>';

  function fieldRow(kind, tag, ins, q, ans) {
    return '<div class="sf-lrow ' + kind + '">'
      + '<span class="sf-ltag">' + esc(tag) + '</span>'
      + '<div class="sf-lfields">'
      + '<div class="sf-fline"><span class="sf-fk">&lt;Instruct&gt;</span>'
      + '<span class="sf-fv sf-ins">' + esc(ins) + '</span></div>'
      + '<div class="sf-fline"><span class="sf-fk">&lt;Query&gt;</span>'
      + '<span class="sf-fv sf-qry">' + esc(q) + '</span></div>'
      + '</div>'
      + '<span class="sf-ans">' + esc(ans) + '</span>'
      + '</div>';
  }

  var headHtml = ROWS.length
    ? '<div class="sf-thead" role="row">'
      + '<span role="columnheader">document, assistant turn</span>'
      + '<span role="columnheader">dataset says</span>'
      + '<span role="columnheader">LLM re-reads</span>'
      + '<span role="columnheader">verdict</span></div>'
    : '';

  var rowsHtml = ROWS.map(function (r, i) {
    function cell(bin, ctg, bad, binBad, catBad) {
      return '<span class="sf-cell" role="cell">'
        + '<b class="sf-bin' + (bad && binBad ? ' bad' : '') + '">'
        + (bad && binBad ? '<span class="sf-ne" aria-hidden="true">&#8800;</span>' : '')
        + esc(bin) + '</b>'
        + '<em class="sf-cat' + (bad && catBad ? ' bad' : '') + '">'
        + (bad && catBad ? '<span class="sf-ne" aria-hidden="true">&#8800;</span>' : '')
        + esc(ctg || 'no category') + '</em></span>';
    }
    return '<div class="sf-trow" role="row" data-i="' + i + '">'
      + '<span class="sf-doc" role="cell">'
      + '<span class="sf-who">[Assistant]</span> ' + esc(r.doc) + '</span>'
      + cell(r.dl, r.dc, false, false, false)
      + '<span class="sf-mcell">'
      + '<span class="sf-pending" aria-hidden="true">'
      + '<em class="q">queued</em><em class="r">reading</em></span>'
      + cell(r.ml, r.mc, true, r.binBad, r.catBad)
      + '</span>'
      + '<span class="sf-vcell" role="cell" aria-label="'
      + (r.drop ? ('drop, the ' + r.why + ' level disagrees') : 'keep, both levels agree') + '">'
      + '<span class="sf-vpend" aria-hidden="true">&middot;&middot;&middot;</span>'
      + '<span class="sf-pill ' + (r.drop ? 'drop' : 'keep') + '">'
      + (r.drop ? 'drop' : 'keep') + '</span></span>'
      + '<i class="sf-strike" aria-hidden="true"></i>'
      + '</div>';
  }).join('');

  wrap.innerHTML =
    /* ---------- stage 01 ---------- */
    '<section class="sf-stage">'
    + '<div class="sf-head"><span class="sf-no">01</span>'
    + '<h4 class="sf-ht">Rebalance</h4>'
    + '<span class="sf-hs">negatives are free, positives are not</span></div>'

    + '<div class="sf-abody">'
    + '<div class="sf-dwrap">'
    + '<div class="sf-dots" role="img" aria-label="'
    + (negPer == null ? 'row field' : ('Field of ' + (negPer + 2) + ' training rows off one '
      + 'annotated document: one yes row, one duplicated yes row, and ' + negPer + ' no rows'))
    + '">' + dotHtml + '</div></div>'
    + '<div class="sf-ratio">' + ratioHtml
    + '<ul class="sf-legend">'
    + '<li><i class="sf-dot yes"></i>yes row</li>'
    + '<li><i class="sf-dot dupkey"></i>duplicate, paraphrased</li>'
    + '<li><i class="sf-dot"></i>no row</li>'
    + '</ul></div>'
    + '</div>'

    + '<div class="sf-lhead">&lt;Document&gt; unchanged. The copy re-words &lt;Instruct&gt; and '
    + '&lt;Query&gt; only, so the answer stays the same. Both wordings are paper examples.</div>'
    + '<div class="sf-ledger">'
    + fieldRow('src', 'source positive', SRC.instruct, SRC.query, SRC.answer)
    + fieldRow('dup', 'duplicate', INSTRUCTS[0] || SRC.instruct, QUERIES[0] || SRC.query,
      SRC.answer)
    + '</div>'
    + '</section>'

    /* ---------- stage 02 ---------- */
    + '<section class="sf-stage">'
    + '<div class="sf-head"><span class="sf-no">02</span>'
    + '<h4 class="sf-ht">Cross-validate</h4>'
    + '<span class="sf-hs">an open-source LLM re-reads every row, binary and per-category'
    + '</span>'
    + '<button class="sf-btn" type="button" id="S_FILTER-run"></button></div>'
    + '<div class="sf-tbl" role="table" aria-label="Label cross-validation, six rows">'
    + headHtml + rowsHtml + '</div>'
    + '<div class="sf-sum">'
    + '<span class="sf-s1"><b>' + ROWS.length + '</b> checked</span>'
    + '<span class="sf-s2"><b>' + nKeep + '</b> kept</span>'
    + '<span class="sf-s3"><b>' + nDrop + '</b> dropped</span>'
    + '<span class="sf-snote">the &#8800; marks the level that disagreed. A disagreement is '
    + 'removed, never relabelled.</span>'
    + '</div>'
    + '</section>'

    /* ---------- provenance ---------- */
    + '<p class="sf-prov"><span class="sf-tag">derived</span>'
    + (negPer == null ? 'Ratio omitted, the taxonomy size is missing from the data file.'
      : 'The ' + ratio(negPer) + ' skew is computed at runtime from the ' + leaves
        + ' leaf categories of the training taxonomy.')
    + '</p>'
    + '<p class="sf-prov"><span class="sf-tag ill">illustrative</span>'
    + 'The ' + ROWS.length + ' rows, their labels and the verdicts were written for the figure, '
    + 'not measured, and no live model is called. The paper gives the filter, not its yield.</p>';

  root.appendChild(wrap);

  /* ---------------- handles ---------------- */

  var dupRow = wrap.querySelector('.sf-lrow.dup');
  var dupIns = dupRow ? dupRow.querySelector('.sf-ins') : null;
  var dupQry = dupRow ? dupRow.querySelector('.sf-qry') : null;
  var dupDot = wrap.querySelector('.sf-dot.dup');
  var rline = wrap.querySelector('.sf-rline');
  var trows = [].slice.call(wrap.querySelectorAll('.sf-trow'));
  var sumEl = wrap.querySelector('.sf-sum');
  var btn = wrap.querySelector('#S_FILTER-run');

  /* ---------------- state application ---------------- */

  var sig = '';
  function apply(dup, scan, judged, sum) {
    var s = dup + '|' + scan + '|' + judged + '|' + sum;
    if (s === sig) return;
    sig = s;
    if (dupRow) dupRow.classList.toggle('in', !!dup);
    if (dupDot) dupDot.classList.toggle('in', !!dup);
    if (rline) rline.classList.toggle('flipped', !!dup);
    for (var i = 0; i < trows.length; i++) {
      trows[i].classList.toggle('active', i === scan);
      trows[i].classList.toggle('done', i < judged);
      trows[i].classList.toggle('struck', i < judged && ROWS[i].drop);
    }
    if (sumEl) sumEl.classList.toggle('in', !!sum);
  }

  function draw(k) {
    if (dupIns && INSTRUCTS.length) {
      dupIns.textContent = INSTRUCTS[k % INSTRUCTS.length];
    }
    if (dupQry && QUERIES.length) {
      dupQry.textContent = QUERIES[k % QUERIES.length];
    }
  }

  /* ---------------- the control ---------------- */

  var showFinal = true;   /* only used when motion is off */

  function labelBtn() {
    if (!btn) return;
    if (api.reduce) {
      btn.textContent = showFinal ? 'show the rows before filtering' : 'show the filtered rows';
      btn.setAttribute('aria-label', btn.textContent);
      btn.setAttribute('aria-pressed', showFinal ? 'true' : 'false');
    } else {
      btn.textContent = 'replay';
      btn.setAttribute('aria-label', 'replay the rebalance and filtering animation');
    }
  }

  var t0 = null, cycle = 0, replay = false;

  if (btn) {
    btn.addEventListener('click', function () {
      if (api.reduce) {
        showFinal = !showFinal;
        if (showFinal) { draw(0); apply(true, -1, ROWS.length, true); }
        else { apply(false, -1, 0, false); }
        labelBtn();
      } else {
        replay = true;
      }
    });
  }
  labelBtn();

  /* ---------------- timeline ---------------- */

  var T = 14;         /* one full cycle, seconds */
  var T_DUP = 1.1;    /* the duplicate row lands */
  var T_SCAN = 2.7;   /* the LLM starts reading */
  var STEP = 1.05;    /* seconds per row */
  var LAG = 0.55;     /* verdict lands this long after the row lights up */

  draw(0);
  apply(true, -1, ROWS.length, true);   /* static default is the finished state */

  return {
    start: function () { t0 = null; },
    stop: function () { t0 = null; draw(cycle); apply(true, -1, ROWS.length, true); },
    tick: function (t) {
      if (t0 == null) { t0 = t; }
      if (replay) { replay = false; t0 = t; cycle++; draw(cycle); }
      var loc = t - t0;
      if (loc >= T) { t0 = t; loc = 0; cycle++; draw(cycle); }

      var judged = 0, scan = -1;
      for (var i = 0; i < ROWS.length; i++) {
        var on = T_SCAN + i * STEP;
        if (loc >= on + LAG) judged = i + 1;
        else if (loc >= on) scan = i;
      }
      apply(loc >= T_DUP, scan, judged, loc >= T_SCAN + ROWS.length * STEP);
    }
  };
};
