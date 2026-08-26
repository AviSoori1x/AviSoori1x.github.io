window.SCENES = window.SCENES || {};

/* Act IV, scene 31. The adaptability benchmark, the one Shieldstral loses.
   The scoreboard is the whole point and it stays on screen in every view:
   the paper's reported adaptability F1 for the winner against Shieldstral's,
   the signed gap in rose, and one flat sentence saying the loss is real. Three
   views hang off that. "the trade" runs the same worked request through both
   output shapes, a reasoning trace typed token by token against a single token
   that has already landed, which is the thing the 2.8 points buys. "the field"
   ranks all ten baselines by a macro recomputed live from the paper's own per
   category table, and prints the disagreement between that recomputation and
   the headline rather than hiding it. "by super class" is the twelve row
   breakdown of where the gap actually sits. Every number is read from
   window.SS at runtime. The trace prose is written for this figure and is
   labelled as such, as is the yes/no label, which is the paper's ground truth
   for that example and not a live call to either model. */
window.SCENES['S_ADAPT'] = function (root, api) {
  var SS = api.SS || {};
  var H = SS.headline || {};
  var BASE = SS.baselines || [];
  var TAXM = SS.taxonomyModels || [];
  var TAX = SS.evalTaxonomy || [];

  root.classList.add('sc-s_adapt');

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function norm(s) {
    return String(s == null ? '' : s).toLowerCase()
      .replace(/\(ours\)/g, '').replace(/[^a-z0-9.]/g, '');
  }
  function splitName(full) {
    var m = /^(.*?)[-\s](\d+(?:\.\d+)?B)$/i.exec(String(full == null ? '' : full));
    return m ? { base: m[1], size: m[2] } : { base: String(full == null ? '' : full), size: null };
  }
  function parseB(sz) {
    var m = /(\d+(?:\.\d+)?)\s*B/i.exec(String(sz == null ? '' : sz));
    return m ? parseFloat(m[1]) : null;
  }
  function signed(v, dp) {
    return (v >= 0 ? '+' : '-') + api.num(Math.abs(v), dp == null ? 1 : dp);
  }

  var ourF1 = (typeof H.adaptabilityF1 === 'number') ? H.adaptabilityF1 : null;
  var bestF1 = (typeof H.adaptabilityBest === 'number') ? H.adaptabilityBest : null;

  if (ourF1 == null || bestF1 == null || !TAXM.length || !TAX.length) {
    root.appendChild(api.frag('<div class="wrap"><p class="miss">'
      + 'SS.headline.adaptabilityF1, SS.headline.adaptabilityBest or the per category table '
      + 'is not present in the data file, so this comparison cannot be drawn.</p></div>'));
    return null;
  }

  /* ---------------- model identity, from the paper's baseline table ------- */

  function describe(full) {
    var sp = splitName(full);
    var key = norm(sp.base), hit = null, i;
    for (i = 0; i < BASE.length; i++) {
      if (norm(BASE[i].model) === key) { hit = BASE[i]; break; }
    }
    var size = hit ? hit.size : sp.size;
    return {
      full: full, base: sp.base, size: size || null, b: parseB(size), matched: !!hit,
      input: hit ? hit.input : null,
      output: hit ? hit.output : null,
      adaptive: hit ? !!hit.adaptive : null
    };
  }

  var META = TAXM.map(describe);
  var matched = META.filter(function (m) { return m.matched; }).length;

  /* ---------------- every row of the paper's per category table ----------- */

  var ROWS = [];                 /* super class, subcategory and leaf rows alike */
  var SUPERS = [];               /* the twelve super class rows on their own */
  TAX.forEach(function (sc) {
    if (sc.f1) { ROWS.push(sc.f1); SUPERS.push({ id: sc.id, name: sc.name, f1: sc.f1 }); }
    (sc.subs || []).forEach(function (u) {
      if (u.f1) ROWS.push(u.f1);
      (u.leaves || []).forEach(function (l) { if (l.f1) ROWS.push(l.f1); });
    });
  });

  function macro(ix) {
    var s = 0, n = 0;
    ROWS.forEach(function (r) {
      var v = r[ix];
      if (typeof v === 'number') { s += v; n += 1; }
    });
    return n ? s / n : null;
  }

  var usIx = -1;
  META.forEach(function (m, i) { if (norm(m.base) === 'shieldstral') usIx = i; });
  if (usIx < 0) usIx = META.length - 1;

  var FIELD = META.map(function (m, i) {
    return { ix: i, m: m, mu: macro(i) };
  }).filter(function (f) { return f.mu != null; });
  FIELD.sort(function (a, b) { return b.mu - a.mu; });

  var winIx = -1, winMu = -Infinity;
  FIELD.forEach(function (f) {
    if (f.ix !== usIx && f.mu > winMu) { winMu = f.mu; winIx = f.ix; }
  });
  var WIN = META[winIx] || { base: 'the leading baseline', size: null, b: null, output: null };
  var US = META[usIx];

  var usRank = 1, i2;
  for (i2 = 0; i2 < FIELD.length; i2++) if (FIELD[i2].ix === usIx) usRank = i2 + 1;

  var reUs = macro(usIx), reWin = macro(winIx);
  var agreeUs = (reUs != null) && Math.abs(reUs - ourF1) < 0.05;
  var agreeWin = (reWin != null) && Math.abs(reWin - bestF1) < 0.05;

  /* nearest model to Shieldstral on the recomputation, either side */
  var near = null;
  FIELD.forEach(function (f) {
    if (f.ix === usIx) return;
    if (!near || Math.abs(f.mu - reUs) < Math.abs(near.mu - reUs)) near = f;
  });

  var GAP = ourF1 - bestF1;
  var ratio = (US.b && WIN.b) ? (WIN.b / US.b) : null;

  /* the data file carries only the top score and ours for this benchmark, so the
     placement on the scoreboard is the paper's reported one and says so */
  var reportedRank = (ourF1 < bestF1) ? 2 : 1;

  var ORD = ['', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'];
  function ord(n) { return ORD[n] || (n + 'th'); }

  /* ---------------- the twelve super class rows, worst gap first ---------- */

  var SC = SUPERS.map(function (s) {
    var a = s.f1[usIx], b = s.f1[winIx];
    return {
      id: s.id, name: s.name, us: a, win: b,
      d: (typeof a === 'number' && typeof b === 'number') ? (a - b) : null
    };
  }).filter(function (s) { return s.d != null; });
  SC.sort(function (a, b) { return a.d - b.d; });

  var scAhead = SC.filter(function (s) { return s.d > 0; }).length;
  var scBehind = SC.filter(function (s) { return s.d < 0; }).length;

  var DSPAN = 0;
  SC.forEach(function (s) { DSPAN = Math.max(DSPAN, Math.abs(s.d)); });
  DSPAN = Math.max(1, Math.ceil(DSPAN));

  /* ---------------- the worked request, from the paper's figure 2 --------- */

  var FIG = (SS.fig2 && SS.fig2[0]) || null;
  var Q = FIG ? FIG.query : null;
  var DOC = FIG ? FIG.document : null;
  var LABEL = FIG ? String(FIG.label || '').toLowerCase() : null;

  /* Written for this figure. The paper publishes output types, not traces. */
  var TRACE = 'The query asks about the response, not the request. The user asks how to build '
    + 'a weapon at home, which would meet the policy if the request were the text under review. '
    + 'The assistant declines and gives no method, no materials, no steps. A refusal is not '
    + 'promotion of violence, so the response does not meet the policy. Answer: ';
  var TOK = [];
  (function () {
    var parts = TRACE.split(' '), i;
    for (i = 0; i < parts.length; i++) {
      if (parts[i] === '') continue;
      TOK.push(parts[i] + (i < parts.length - 1 ? ' ' : ''));
    }
  })();
  var NTOK = TOK.length;

  /* ---------------- shell ------------------------------------------------- */

  var VIEWS = ['the trade', 'the field', 'by super class'];

  root.appendChild(api.frag(
    '<div class="wrap" id="S_ADAPT-wrap">'

    + '<div class="hd">'
    +   '<span class="eyeb">adaptability benchmark<i>F1 on the held out '
    +     esc(H.evalLeaves != null ? H.evalLeaves : SC.length) + ' category taxonomy</i></span>'
    +   '<span class="seg" role="group" aria-label="which side of the result to look at">'
    +     VIEWS.map(function (v, i) {
          return '<button type="button" class="sg' + (i === 0 ? ' sel' : '') + '"'
            + ' id="S_ADAPT-view-' + i + '" aria-pressed="' + (i === 0) + '">'
            + esc(v) + '</button>';
        }).join('')
    +   '</span>'
    + '</div>'

    /* the scoreboard, on screen in every view */
    + '<div class="board">'
    +   '<div class="cell win">'
    +     '<span class="pos">' + esc(ord(1)) + '<i>as reported</i></span>'
    +     '<span class="fig">' + api.num(bestF1, 1) + '</span>'
    +     '<span class="who">' + esc(WIN.base) + '</span>'
    +     '<span class="spec"><b>' + esc(WIN.size || 'size n/a') + '</b>'
    +       '<em>' + esc(WIN.output || 'output n/a') + '</em>'
    +       '<em>' + esc(WIN.input || 'input n/a') + '</em></span>'
    +   '</div>'
    +   '<div class="cell mid">'
    +     '<span class="gapfig">' + signed(GAP, 1) + '</span>'
    +     '<span class="gaplab">F1 points<br>behind</span>'
    +   '</div>'
    +   '<div class="cell us">'
    +     '<span class="pos">' + esc(ord(reportedRank)) + '<i>as reported</i></span>'
    +     '<span class="fig">' + api.num(ourF1, 1) + '</span>'
    +     '<span class="who">' + esc(US.base) + '</span>'
    +     '<span class="spec"><b>' + esc(US.size || 'size n/a') + '</b>'
    +       '<em>' + esc(US.output || 'output n/a') + '</em>'
    +       '<em>' + esc(US.input || 'input n/a') + '</em></span>'
    +   '</div>'
    + '</div>'

    + '<p class="straight">This is a loss. ' + esc(US.base) + ' is <b>' + api.num(-GAP, 1)
    +   ' F1 points</b> behind ' + esc(WIN.base) + ' on the benchmark that tests the one thing it '
    +   'is sold on, adapting to a policy it never saw in training. Nothing below cancels that.</p>'

    + '<div class="body" id="S_ADAPT-body"></div>'

    + '<p class="foot" id="S_ADAPT-foot"></p>'

    + '</div>'
  ));

  var bodyEl = root.querySelector('#S_ADAPT-body');
  var footEl = root.querySelector('#S_ADAPT-foot');

  footEl.innerHTML =
    'The two headline scores are the figures the paper reports, read from the data file at load '
    + 'time. They are not a live model call. Parameter counts, input modality and output type '
    + 'are matched by name against the paper’s baseline table, ' + matched + ' of ' + META.length
    + ' resolved. The per category rows below are the paper’s own, ' + ROWS.length
    + ' rows across ' + SUPERS.length + ' super classes.';

  /* ---------------- view 0, the trade ------------------------------------- */

  var dots = [], traceEl = null, cntEl = null, labelEl = null, replayEl = null;

  function buildTrade() {
    var html = '<div class="trade">';

    if (Q && DOC) {
      html += '<div class="input">'
        + '<span class="ik">one request, both models</span>'
        + '<div class="ifields">'
        +   '<div class="ifield"><span class="fk">query</span>'
        +     '<span class="fv">' + esc(Q) + '</span></div>'
        +   '<div class="ifield"><span class="fk">document</span>'
        +     '<span class="fv doc">' + esc(DOC) + '</span></div>'
        + '</div>'
        + '</div>';
    }

    html += '<div class="cols">';

    /* the winner */
    html += '<div class="pane rival">'
      + '<div class="ph"><span class="pn">' + esc(WIN.full) + '</span>'
      +   '<span class="pt">' + esc(WIN.output || 'output n/a') + '</span></div>'
      + '<div class="out"><span class="trc" id="S_ADAPT-trace"></span>'
      +   '<span class="lbl" id="S_ADAPT-lbl">' + (LABEL ? esc(LABEL) : '') + '</span></div>'
      + '<div class="meter"><div class="dots" id="S_ADAPT-dots" aria-hidden="true"></div>'
      +   '<span class="cnt" id="S_ADAPT-cnt"></span></div>'
      + '</div>';

    /* Shieldstral */
    var cands = ['yes', 'no'];
    html += '<div class="pane mine">'
      + '<div class="ph"><span class="pn">' + esc(US.full) + '</span>'
      +   '<span class="pt">' + esc(US.output || 'output n/a') + '</span></div>'
      + '<div class="out one">'
      +   '<div class="slot">'
      +     cands.map(function (c) {
            var on = (LABEL === c);
            return '<span class="cand' + (on ? ' on' : '') + '">' + esc(c)
              + (on ? '<i>emitted</i>' : '<i>not emitted</i>') + '</span>';
          }).join('')
      +   '</div>'
      +   '<span class="slotk">one token position, two candidates. The score is the probability '
      +     'mass on <b>yes</b>, read off the logprobs.</span>'
      + '</div>'
      + '<div class="meter"><div class="dots"><span class="dot on"></span></div>'
      +   '<span class="cnt fin"><b>1</b> token, always</span></div>'
      + '</div>';

    html += '</div>';

    html += '<div class="tradefoot">'
      + '<p class="tf1">' + esc(WIN.base) + ' is <b>'
      + (ratio ? api.num(ratio, 1) + 'x' : 'more than') + '</b> the parameters, '
      + esc(WIN.size || 'size n/a') + ' against ' + esc(US.size || 'size n/a')
      + ', and it decodes a trace before it commits. ' + esc(US.base) + ' decodes one token. '
      + 'Both read the same input first. If the guardrail runs on every request, that difference '
      + 'is the bill.</p>'
      + '<p class="tf2"><span class="ill">illustrative</span> The trace prose is written for this '
      + 'figure. The paper publishes output <em>types</em>, ' + esc(WIN.output || 'n/a')
      + ' against ' + esc(US.output || 'n/a') + ', not lengths, so the meter counts words and '
      + 'understates the real token count. The <b>' + esc(LABEL || 'yes/no')
      + '</b> is the paper’s ground truth label for this example, not a live call to either model.'
      + (api.reduce ? '' : ' <button type="button" class="rp" id="S_ADAPT-replay">replay the '
          + 'trace</button>')
      + '</p>'
      + '</div>';

    html += '</div>';
    bodyEl.innerHTML = html;

    traceEl = root.querySelector('#S_ADAPT-trace');
    cntEl = root.querySelector('#S_ADAPT-cnt');
    labelEl = root.querySelector('#S_ADAPT-lbl');
    replayEl = root.querySelector('#S_ADAPT-replay');

    var host = root.querySelector('#S_ADAPT-dots');
    dots = [];
    var i, s;
    for (i = 0; i < NTOK; i++) {
      s = document.createElement('span');
      s.className = 'dot';
      host.appendChild(s);
      dots.push(s);
    }

    if (replayEl) {
      replayEl.addEventListener('click', function () { t0 = null; shown = -1; });
    }

    /* reduced motion never gets a tick, so it gets the finished state */
    paintTrace(api.reduce ? NTOK : 0);
  }

  /* the untyped remainder stays in the flow but hidden, so the pane is the same
     height at the first token as at the last and nothing jumps while it runs */
  function paintTrace(n) {
    if (!traceEl) return;
    var done = (n >= NTOK);
    traceEl.innerHTML =
      '<span class="tp' + (done ? '' : ' typing') + '">' + esc(TOK.slice(0, n).join('')) + '</span>'
      + '<span class="tq">' + esc(TOK.slice(n).join('')) + '</span>';
    if (labelEl) labelEl.className = 'lbl' + (done ? ' on' : '');
    var i, on;
    for (i = 0; i < dots.length; i++) {
      on = (i < n);
      if (dots[i]._on !== on) { dots[i]._on = on; dots[i].className = 'dot' + (on ? ' on' : ''); }
    }
    if (cntEl) {
      cntEl.className = 'cnt' + (done ? ' fin' : '');
      cntEl.innerHTML = '<b>' + n + '</b> of ' + NTOK + ' tokens'
        + (done ? ', then the label' : ' and still going');
    }
  }

  /* ---------------- view 1, the field ------------------------------------- */

  function buildField() {
    var top = FIELD[0] ? FIELD[0].mu : 100;
    var html = '<div class="field">'
      + '<div class="fhd"><span class="fk2">every baseline the paper scores on this benchmark, '
      + 'ranked by a macro recomputed here from the ' + ROWS.length + ' rows of its per category '
      + 'table</span></div>'
      + '<ol class="frows">';

    FIELD.forEach(function (f, i) {
      var isUs = (f.ix === usIx), isWin = (f.ix === winIx);
      html += '<li class="fr' + (isUs ? ' us' : '') + (isWin ? ' top' : '') + '">'
        + '<span class="rk">' + esc(ord(i + 1)) + '</span>'
        + '<span class="fnm">' + esc(f.m.base) + '</span>'
        + '<span class="fsz">' + esc(f.m.size || 'n/a') + '</span>'
        + '<span class="ftr" aria-hidden="true"><i style="width:'
        +   ((f.mu / 100) * 100).toFixed(2) + '%"></i></span>'
        + '<span class="fvl">' + api.num(f.mu, 2) + '</span>'
        + '<span class="fop">' + esc(f.m.output || 'output n/a') + '</span>'
        + '</li>';
    });

    html += '</ol>'
      + '<p class="fnote">Bars run the full 0 to 100 axis, and the number is printed either way. '
      + 'Rank is by the recomputed macro, so it is a check on the headline rather than a copy of '
      + 'it.</p>'
      + '<p class="fnote warn"><span class="ill rose">read this before quoting the rank</span> '
      + 'Averaging all ' + ROWS.length + ' rows of the paper’s table gives <b>'
      + api.num(reUs, 2) + '</b> for ' + esc(US.base) + ', which '
      + (agreeUs ? 'matches' : 'does not match') + ' its headline ' + api.num(ourF1, 1)
      + ', and <b>' + api.num(reWin, 2) + '</b> for ' + esc(WIN.base) + ', which '
      + (agreeWin ? 'matches' : 'does not match') + ' its headline ' + api.num(bestF1, 1) + '. '
      + (agreeWin
          ? 'The two aggregations agree.'
          : 'So the headline uses an aggregation this data file does not reproduce exactly, and '
            + 'the recomputed column is close but not identical to the reported one.')
      + (near
          ? ' On that same recomputation ' + esc(near.m.base) + ' at '
            + esc(near.m.size || 'size n/a') + ' lands at <b>' + api.num(near.mu, 2)
            + '</b>, ' + api.num(Math.abs(near.mu - reUs), 2) + ' from ' + esc(US.base)
            + ', which puts ' + esc(US.base) + ' ' + esc(ord(usRank))
            + ' here rather than second. Second place is the paper’s reported ordering, not '
            + 'something this table settles.'
          : '')
      + '</p>'
      + '</div>';

    bodyEl.innerHTML = html;
  }

  /* ---------------- view 2, by super class -------------------------------- */

  function buildSC() {
    var html = '<div class="scv">'
      + '<div class="fhd"><span class="fk2">F1 per super class, ' + esc(US.base) + ' minus '
      + esc(WIN.base) + ', worst first. Behind on <b>' + scBehind + '</b> of ' + SC.length
      + ', ahead on <b>' + scAhead + '</b>.</span></div>'
      + '<div class="schdr" aria-hidden="true">'
      +   '<span class="sid"></span><span class="snm">super class</span>'
      +   '<span class="sv rival">' + esc(WIN.base.split('-').slice(0, 2).join('-')) + '</span>'
      +   '<span class="sv mine">ours</span>'
      +   '<span class="sbar"><i class="zero"></i></span>'
      +   '<span class="sd">gap</span>'
      + '</div>'
      + '<ol class="scrows">';

    SC.forEach(function (s) {
      var w = (Math.abs(s.d) / DSPAN) * 50;
      html += '<li class="sr' + (s.d >= 0 ? ' ahead' : ' behind') + '">'
        + '<span class="sid">' + esc(s.id) + '</span>'
        + '<span class="snm">' + esc(s.name) + '</span>'
        + '<span class="sv rival">' + api.num(s.win, 1) + '</span>'
        + '<span class="sv mine">' + api.num(s.us, 1) + '</span>'
        + '<span class="sbar" aria-hidden="true">'
        +   '<i class="zero"></i>'
        +   '<i class="fill" style="' + (s.d >= 0
              ? 'left:50%;width:' + w.toFixed(2) + '%'
              : 'right:50%;width:' + w.toFixed(2) + '%') + '"></i>'
        + '</span>'
        + '<span class="sd">' + signed(s.d, 1) + '</span>'
        + '<span class="vh">' + esc(s.name) + ', ' + esc(WIN.base) + ' ' + api.num(s.win, 1)
        +   ', ' + esc(US.base) + ' ' + api.num(s.us, 1) + ', '
        +   (s.d >= 0 ? 'ahead by ' : 'behind by ') + api.num(Math.abs(s.d), 1) + '.</span>'
        + '</li>';
    });

    html += '</ol>'
      + '<p class="fnote">Bars are centred on zero and share one scale, plus or minus '
      + DSPAN + ' F1 points. Left of the line is a loss for ' + esc(US.base)
      + ', right of it is a win, and the sign is printed so the direction never rests on colour. '
      + 'These are the paper’s super class rows, not a live evaluation.</p>'
      + '</div>';

    bodyEl.innerHTML = html;
  }

  /* ---------------- view switching ---------------------------------------- */

  var view = 0;
  var t0 = null, shown = -1;
  var RATE = 15;          /* words per second */
  var LEAD = 0.45;        /* a beat before the trace starts */

  function build() {
    dots = []; traceEl = null; cntEl = null; labelEl = null; replayEl = null;
    if (view === 0) buildTrade();
    else if (view === 1) buildField();
    else buildSC();
  }

  VIEWS.forEach(function (v, i) {
    var b = root.querySelector('#S_ADAPT-view-' + i);
    b.addEventListener('click', function () {
      if (view === i) return;
      view = i;
      VIEWS.forEach(function (x, j) {
        var o = root.querySelector('#S_ADAPT-view-' + j);
        o.classList.toggle('sel', j === i);
        o.setAttribute('aria-pressed', String(j === i));
      });
      t0 = null; shown = -1;
      build();
    });
  });

  build();

  return {
    start: function () { t0 = null; shown = -1; },
    stop: function () {},
    tick: function (t) {
      if (view !== 0 || !traceEl) return;
      if (t0 === null) t0 = t;
      var e = (t - t0) - LEAD;
      var n = Math.max(0, Math.min(NTOK, Math.floor(e * RATE)));
      if (n !== shown) { shown = n; paintTrace(n); }
    }
  };
};
