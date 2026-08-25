window.SCENES = window.SCENES || {};

/* S_NEG, act 2, beat 15.
   Three negative construction strategies. The point of the figure is the first two:
   the document really is unsafe and the correct label is still "no".
   Every class inside is prefixed sn- so the page level styles cannot reach in. */
window.SCENES['S_NEG'] = function (root, api) {
  var SS = api.SS || {};
  var fig3 = SS.fig3 || {};
  var pos = fig3.positive || {};
  var neg = fig3.negative || {};
  var head = SS.headline || {};
  var qt = SS.queryTypes || [];

  function ex(i, j) {
    var o = qt[i];
    return (o && o.examples && o.examples[j]) ? o.examples[j] : null;
  }
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ---------------- the three strategies ---------------- */

  var leaves = head.trainLeaves;

  var STRATS = [
    {
      name: 'Category-based',
      hint: 'violates A, ask about B',
      unsafe: true,
      stamp: 'unsafe',
      stampWhy: 'it violates one leaf category of the training taxonomy',
      conn: 'keep the text, swap the query',
      ansKick: 'the hard negative label',
      ansWhy: 'The query names a category this document does not violate.',
      naive: true,
      doc: fig3.document,
      rows: [
        { role: pos.role || 'target category', q: pos.query, a: pos.label },
        { role: neg.role || 'sibling category', q: neg.query, a: neg.label }
      ],
      why: 'One violating document, one query per category. The category it actually violates '
        + 'answers yes. Every other category in the taxonomy answers no on the same text.',
      count: (leaves != null)
        ? (leaves + ' leaf categories in the training taxonomy, so a single document can source '
           + 'one yes and up to ' + (leaves - 1) + ' hard no rows')
        : null,
      tag: 'from the paper',
      tagKind: 'src',
      prov: 'Document, queries and both labels are the paper example, read from the data file at '
        + 'runtime. Labels are the pipeline ground truth, not a live model call.'
    },
    {
      name: 'Demographic-based',
      hint: 'targets one group, ask about another',
      unsafe: true,
      stamp: 'unsafe',
      stampWhy: 'it attacks a protected group',
      conn: 'keep the text, swap the group',
      ansKick: 'the hard negative label',
      ansWhy: 'The query names a group this document does not target.',
      naive: true,
      doc: '[User] What do you make of the new families on our street?\n'
        + '[Assistant] That whole religious community is a blight on the neighbourhood. '
        + 'People like them should be pushed out before they settle in.',
      rows: [
        { role: 'targeted group', a: 'yes',
          q: 'Does this text demean or attack a group on the basis of their religion?' },
        { role: 'unrelated group', a: 'no',
          q: 'Does this text demean or attack a group on the basis of their sexual orientation?' },
        { role: 'unrelated group', a: 'no',
          q: 'Does this text demean or attack a group on the basis of their disability?' }
      ],
      why: 'Target-group annotations become group-specific queries. Hate aimed at one group is '
        + 'not hate aimed at every group, and the model has to hold that line.',
      count: null,
      tag: 'illustrative',
      tagKind: 'ill',
      prov: 'The strategy is from the paper. This document and its three queries were written '
        + 'for the figure, and the labels are the construction rule, not a live model call.'
    },
    {
      name: 'Safe-content',
      hint: 'safe text, ordinary harm queries',
      unsafe: false,
      stamp: 'safe',
      stampWhy: 'nothing in it violates anything',
      conn: 'ask the ordinary questions',
      ansKick: 'every label here',
      ansWhy: 'Both halves agree. This is the easy case, and the high volume case.',
      naive: false,
      doc: '[User] What is the safest way to store kitchen knives with a toddler in the house?\n'
        + '[Assistant] Keep them in a locked drawer, or on a magnetic strip mounted well above '
        + 'a child’s reach.',
      rows: [
        { role: 'category-specific query', a: 'no', q: ex(0, 0) },
        { role: 'category-specific query', a: 'no', q: ex(0, 1) },
        { role: 'binary query', a: 'no', q: ex(1, 0) }
      ],
      why: 'Benign text paired with the same harm queries the positives use. It teaches the '
        + 'floor: absent a violation the answer is no, however the question is phrased.',
      count: null,
      tag: 'illustrative document',
      tagKind: 'ill',
      prov: 'The three queries are read from the data file at runtime, the document was written '
        + 'for the figure, and the labels are the construction rule, not a live model call.'
    }
  ];

  STRATS.forEach(function (s) {
    s.rows = s.rows.filter(function (r) { return r.q; });
    s.nYes = s.rows.filter(function (r) { return String(r.a).toLowerCase() === 'yes'; }).length;
    s.nNo = s.rows.length - s.nYes;
  });

  /* ---------------- shell ---------------- */

  root.classList.add('sc-s_neg');

  var box = api.el('div', 'sn-wrap');
  var tabhtml = STRATS.map(function (s, i) {
    return '<button class="sn-tab" type="button" role="tab" id="S_NEG-tab-' + i + '"'
      + ' aria-controls="S_NEG-panel" aria-selected="' + (i === 0) + '"'
      + ' tabindex="' + (i === 0 ? '0' : '-1') + '">'
      + '<span class="sn-tn">0' + (i + 1) + '</span>'
      + '<span class="sn-tl">' + esc(s.name) + '</span>'
      + '<span class="sn-ts">' + esc(s.hint) + '</span>'
      + '<i class="sn-prog" aria-hidden="true"></i></button>';
  }).join('');

  box.innerHTML =
    '<div class="sn-tabs" role="tablist" aria-label="Negative construction strategies">'
    + tabhtml + '</div>'
    + '<div class="sn-panel" id="S_NEG-panel" role="tabpanel" aria-labelledby="S_NEG-tab-0"></div>';
  root.appendChild(box);

  var tabEls = [].slice.call(box.querySelectorAll('.sn-tab'));
  var panel = box.querySelector('.sn-panel');

  var CHEV = '<svg class="sn-chev" viewBox="0 0 64 24" aria-hidden="true" focusable="false">'
    + '<path d="M4 12h44" stroke="currentColor" stroke-width="1.5" fill="none"'
    + ' stroke-dasharray="3 5" stroke-linecap="round"/>'
    + '<path d="M44 5.5 55 12l-11 6.5" stroke="currentColor" stroke-width="1.8" fill="none"'
    + ' stroke-linejoin="round" stroke-linecap="round"/></svg>';

  function render(s) {
    var rowhtml = s.rows.map(function (r, i) {
      var yes = String(r.a).toLowerCase() === 'yes';
      var delay = api.reduce ? '' : ' style="animation-delay:' + (0.05 + i * 0.07) + 's"';
      return '<div class="sn-row ' + (yes ? 'yes' : 'no') + '"' + delay + '>'
        + '<div><span class="sn-role">' + esc(r.role) + '</span>'
        + '<span class="sn-q">' + esc(r.q) + '</span></div>'
        + '<span class="sn-pill ' + (yes ? 'y' : 'n') + '">'
        + '<i class="sn-dot" aria-hidden="true"></i>' + esc(r.a) + '</span>'
        + '</div>';
    }).join('');

    panel.innerHTML =
      '<div class="sn-hero">'
      + '<div class="sn-hcell">'
      + '<div class="sn-kick">the document is</div>'
      + '<div class="sn-stamp ' + (s.unsafe ? 'bad' : 'ok') + '">' + esc(s.stamp) + '</div>'
      + '<div class="sn-ksub">' + esc(s.stampWhy) + '</div></div>'
      + '<div class="sn-conn">' + CHEV + '<span>' + esc(s.conn) + '</span></div>'
      + '<div class="sn-hcell sn-right">'
      + '<div class="sn-kick">' + esc(s.ansKick) + '</div>'
      + '<div class="sn-big">no</div>'
      + '<div class="sn-ksub">' + esc(s.ansWhy)
      + (s.naive ? ' <span class="sn-wrong">a plain harm detector fires here, wrongly</span>' : '')
      + '</div></div></div>'

      + '<div class="sn-docbox"><div class="sn-flab">&lt;Document&gt;</div>'
      + '<div class="sn-doctext">' + esc(s.doc) + '</div></div>'

      + '<div class="sn-flab sn-qlab">&lt;Query&gt; &nbsp;&middot;&nbsp; ' + s.rows.length
      + ' training rows off this one document &nbsp;&middot;&nbsp; '
      + s.nYes + ' yes, ' + s.nNo + ' no</div>'
      + '<div class="sn-rows">' + rowhtml + '</div>'

      + '<div class="sn-foot"><p class="sn-why">' + esc(s.why) + '</p>'
      + (s.count ? '<p class="sn-cnt">' + esc(s.count) + '</p>' : '')
      + '<p class="sn-prov"><span class="sn-tag ' + esc(s.tagKind) + '">' + esc(s.tag)
      + '</span>' + esc(s.prov) + '</p></div>';
  }

  /* ---------------- stepping ---------------- */

  var idx = 0;
  var auto = !api.reduce;
  var t0 = null;
  var PERIOD = 9;

  function setProg(p) {
    for (var i = 0; i < tabEls.length; i++) {
      var bar = tabEls[i].querySelector('.sn-prog');
      if (bar) {
        bar.style.transform = 'scaleX(' + (i === idx ? Math.max(0, Math.min(1, p)) : 0) + ')';
      }
    }
  }

  function select(i, byUser) {
    idx = (i + STRATS.length) % STRATS.length;
    tabEls.forEach(function (t, k) {
      var on = k === idx;
      t.classList.toggle('on', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
    });
    panel.setAttribute('aria-labelledby', 'S_NEG-tab-' + idx);
    render(STRATS[idx]);
    if (byUser) { auto = false; t0 = null; setProg(0); }
  }

  tabEls.forEach(function (t, i) {
    t.addEventListener('click', function () { select(i, true); });
    t.addEventListener('keydown', function (e) {
      var k = e.key, n = null;
      if (k === 'ArrowRight' || k === 'ArrowDown') n = idx + 1;
      else if (k === 'ArrowLeft' || k === 'ArrowUp') n = idx - 1;
      else if (k === 'Home') n = 0;
      else if (k === 'End') n = STRATS.length - 1;
      if (n == null) return;
      e.preventDefault();
      select(n, true);
      tabEls[idx].focus();
    });
  });

  select(0, false);

  return {
    start: function () { t0 = null; },
    stop: function () { t0 = null; setProg(0); },
    tick: function (t) {
      if (!auto) return;
      if (t0 == null) { t0 = t; }
      var p = (t - t0) / PERIOD;
      if (p >= 1) { t0 = t; p = 0; select(idx + 1, false); }
      setProg(p);
    }
  };
};
