window.SCENES = window.SCENES || {};

/* S_IMGSCARCE, act 2, beat 19, section 3.4.
   Two parallel construction lanes. Text: a safe document is rewritten by an LLM into an
   unsafe variant, which yields a positive row. Image: the same move is blocked, because an
   unsafe image cannot simply be generated the way unsafe text can be rewritten. Then the
   workaround, general purpose classification and detection corpora donating naturally safe
   pictures as negatives.

   Nothing unsafe is ever depicted. Every picture on screen is a plain geometric glyph.
   All numerals are read from window.SS at build time. Class prefix si- throughout. */
window.SCENES['S_IMGSCARCE'] = function (root, api) {
  var SS = api.SS || {};
  var head = SS.headline || {};
  var fig4 = SS.fig4 || {};
  var inv = SS.benchInventory || [];
  var mmRows = ((SS.benchmarks || {}).multimodal || {}).rows || [];

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function fmt(n) {
    if (n == null || isNaN(Number(n))) return null;
    return String(Number(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  /* ---------------- numbers, all from SS ---------------- */

  var synthText = head.syntheticText;        /* millions of rewritten text rows */
  var mmSamples = head.multimodalSamples;    /* millions of multimodal rows */
  var openText = head.openSourceText;        /* millions of public text rows */
  var totalM = head.totalSamples;

  /* how thin the image side of the paper's own evaluation inventory is.
     Measured here by matching the multimodal benchmark names against the inventory. */
  var mmNames = {};
  mmRows.forEach(function (r) { if (r && r.name) mmNames[String(r.name).toLowerCase()] = 1; });
  var splitsAll = inv.length;   // benchmarks in the eval inventory, not splits
  var sampAll = 0, splitsImg = 0, sampImg = 0;
  inv.forEach(function (r) {
    var n = Number(r[3]) || 0;
    sampAll += n;
    if (mmNames[String(r[0]).toLowerCase()]) { splitsImg += 1; sampImg += n; }
  });

  /* the training mix, three parts that sum to the total */
  var mix = [
    { k: 'open source text', v: openText, tone: 'dim' },
    { k: 'synthetic text', v: synthText, tone: 'dim' },
    { k: 'multimodal', v: mmSamples, tone: 'hot' }
  ].filter(function (m) { return m.v != null; });
  var mixSum = mix.reduce(function (a, m) { return a + Number(m.v); }, 0);

  /* ---------------- glyph vocabulary, abstract shapes only ---------------- */

  var GS = 'stroke="currentColor" fill="none" stroke-width="2.4"'
    + ' stroke-linecap="round" stroke-linejoin="round"';

  var GLYPHS = [
    '<circle cx="40" cy="27" r="13" ' + GS + '/><path d="M9 45h62" ' + GS + '/>',
    '<path d="M40 13 62 46H18Z" ' + GS + '/>',
    '<rect x="16" y="14" width="27" height="27" rx="4" ' + GS + '/>'
      + '<circle cx="57" cy="38" r="9" ' + GS + '/>',
    '<circle cx="40" cy="30" r="6" ' + GS + '/><circle cx="40" cy="30" r="12" ' + GS + '/>'
      + '<circle cx="40" cy="30" r="18" ' + GS + '/>',
    '<path d="M22 44V22M40 44V12M58 44V28" ' + GS + '/>',
    '<path d="M40 11 60 22v22L40 55 20 44V22Z" ' + GS + '/>',
    '<path d="M40 14v32M24 30h32" ' + GS + '/><rect x="20" y="10" width="40" height="40" rx="7" '
      + GS + '/>',
    '<path d="M14 42q26-32 52 0" ' + GS + '/><circle cx="26" cy="18" r="3.4" ' + GS + '/>'
      + '<circle cx="40" cy="13" r="3.4" ' + GS + '/><circle cx="54" cy="18" r="3.4" ' + GS + '/>'
  ];

  function glyph(i) {
    return '<svg class="si-g" viewBox="0 0 80 60" aria-hidden="true" focusable="false">'
      + GLYPHS[i % GLYPHS.length] + '</svg>';
  }

  /* the dashed detection frame, drawn only on the detection sourced tiles */
  var BBOX = '<svg class="si-bb" viewBox="0 0 80 60" aria-hidden="true" focusable="false">'
    + '<rect x="12" y="8" width="56" height="44" rx="2" fill="none" stroke="currentColor"'
    + ' stroke-width="1.6" stroke-dasharray="5 4"/>'
    + '<path d="M12 16V8h8M60 8h8v8M68 44v8h-8M20 52h-8v-8" fill="none" stroke="currentColor"'
    + ' stroke-width="2.4" stroke-linecap="round"/></svg>';

  var TILES = [
    { g: 0, kind: 'cls', tag: 'class: outdoor' },
    { g: 1, kind: 'det', tag: 'box: 1 object' },
    { g: 2, kind: 'cls', tag: 'class: object' },
    { g: 3, kind: 'det', tag: 'box: 1 object' },
    { g: 4, kind: 'cls', tag: 'class: chart' },
    { g: 5, kind: 'det', tag: 'box: 1 object' },
    { g: 6, kind: 'cls', tag: 'class: symbol' },
    { g: 7, kind: 'det', tag: 'box: 3 objects' }
  ];

  /* ---------------- text lane copy, from the paper example ---------------- */

  var srcText = fig4.source || null;
  var outText = (fig4.positive || {}).document || null;
  var outLabel = (fig4.positive || {}).label || 'yes';

  /* ---------------- shell ---------------- */

  root.classList.add('sc-s_imgscarce');

  var wrap = api.el('div', 'si-wrap');
  if (api.reduce) wrap.classList.add('si-still');

  var CHEVRON = '<svg class="si-cv" viewBox="0 0 24 40" aria-hidden="true" focusable="false">'
    + '<path d="M12 3v27" stroke="currentColor" stroke-width="1.6" fill="none"'
    + ' stroke-dasharray="3 5" stroke-linecap="round"/>'
    + '<path d="M5.5 24 12 33l6.5-9" stroke="currentColor" stroke-width="2" fill="none"'
    + ' stroke-linejoin="round" stroke-linecap="round"/></svg>';

  var BAR = '<svg class="si-barsvg" viewBox="0 0 240 22" preserveAspectRatio="none"'
    + ' aria-hidden="true" focusable="false">'
    + '<defs><pattern id="S_IMGSCARCE-hatch" width="9" height="9" patternUnits="userSpaceOnUse"'
    + ' patternTransform="rotate(45)">'
    + '<line x1="0" y1="0" x2="0" y2="9" stroke="currentColor" stroke-width="3.4"'
    + ' opacity=".5"/></pattern></defs>'
    + '<rect x="0" y="0" width="240" height="22" rx="4" fill="url(#S_IMGSCARCE-hatch)"'
    + ' stroke="currentColor" stroke-width="1.4" vector-effect="non-scaling-stroke"/></svg>';

  var STOP = '<svg class="si-stop" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
    + '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/>'
    + '<path d="M6 18 18 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';

  var tilesHtml = TILES.map(function (t, i) {
    return '<figure class="si-tile" data-i="' + i + '">'
      + '<div class="si-frame">' + glyph(t.g) + (t.kind === 'det' ? BBOX : '')
      + '<span class="si-pill n"><i class="si-dot" aria-hidden="true"></i>no</span>'
      + '</div>'
      + '<figcaption class="si-cap"><span class="si-tag ' + t.kind + '">' + esc(t.tag)
      + '</span></figcaption></figure>';
  }).join('');

  var mixBar = mix.map(function (m) {
    var pct = mixSum > 0 ? (Number(m.v) / mixSum * 100) : 0;
    return '<i class="si-seg ' + m.tone + '" style="flex:' + pct.toFixed(3) + ' 1 0"'
      + ' aria-hidden="true"></i>';
  }).join('');
  var mixKey = mix.map(function (m) {
    var pct = mixSum > 0 ? (Number(m.v) / mixSum * 100) : 0;
    return '<li class="si-key ' + m.tone + '"><i class="si-sw" aria-hidden="true"></i>'
      + '<span class="si-kk">' + esc(m.k) + '</span>'
      + '<b class="si-kv">' + esc(api.num(m.v, 1)) + 'M</b>'
      + '<span class="si-kp">' + pct.toFixed(1) + '%</span></li>';
  }).join('');

  var invLine = (splitsImg && splitsAll)
    ? ('paper evaluation inventory, not a count of training sources: ' + splitsImg + ' of ' + splitsAll + ' benchmarks are image, '
       + fmt(sampImg) + ' of ' + fmt(sampAll) + ' samples')
    : null;

  wrap.innerHTML =

    '<div class="si-head">'
    + '<p class="si-kick">the rewrite trick, run twice</p>'
    + '<h4 class="si-title">You cannot rewrite a picture</h4>'
    + '</div>'

    + '<div class="si-lanes">'

    /* ---- lane 1, text, the trick lands ---- */
    + '<section class="si-lane ok" aria-label="Text lane, the rewrite works">'
    + '<header class="si-lh"><span class="si-ln">01</span>'
    + '<span class="si-lt">text</span>'
    + '<span class="si-verd good si-rev" data-r="tv">rewrite lands</span></header>'

    + '<div class="si-cell si-rev" data-r="tsrc"><span class="si-clab">safe source</span>'
    + (srcText
        ? '<p class="si-mono">' + esc(srcText) + '</p>'
        : '<p class="si-mono si-miss">source text not in the data file</p>')
    + '</div>'

    + '<div class="si-track si-rev" data-r="tarr">'
    + '<span class="si-op">' + CHEVRON + '<b>LLM rewrite</b>'
    + '<i>target a category, keep the topic</i></span>'
    + '<i class="si-pulse" data-p="text" aria-hidden="true"></i></div>'

    + '<div class="si-cell hot si-rev" data-r="tout">'
    + '<div class="si-chd"><span class="si-clab">unsafe variant</span>'
    + '<span class="si-rlab">training row</span>'
    + '<span class="si-pill y"><i class="si-dot" aria-hidden="true"></i>' + esc(outLabel)
    + '</span></div>'
    + (outText
        ? '<p class="si-mono">' + esc(outText) + '</p>'
        : '<p class="si-mono si-miss">rewritten text not in the data file</p>')
    + '</div>'

    + '<p class="si-lfoot">'
    + (synthText != null
        ? '<b>' + esc(api.num(synthText, 1)) + 'M</b> synthetic text samples come out of this lane'
        : 'synthetic text volume not in the data file')
    + '</p>'
    + '</section>'

    /* ---- lane 2, image, the trick is blocked ---- */
    + '<section class="si-lane no" aria-label="Image lane, the rewrite is blocked">'
    + '<header class="si-lh"><span class="si-ln">02</span>'
    + '<span class="si-lt">image</span>'
    + '<span class="si-verd bad si-rev" data-r="iv">' + STOP + 'blocked</span></header>'

    + '<div class="si-cell si-rev" data-r="isrc"><span class="si-clab">safe source</span>'
    + '<div class="si-shot"><div class="si-frame big">' + glyph(3) + '</div>'
    + '<p class="si-mono si-thin">an ordinary picture, nothing to moderate</p></div></div>'

    + '<div class="si-track si-rev" data-r="iarr">'
    + '<span class="si-op muted">' + CHEVRON + '<b>same move?</b>'
    + '<i>generate the unsafe counterpart</i></span>'
    + '<i class="si-pulse" data-p="img" aria-hidden="true"></i>'
    + '<div class="si-bar si-rev" data-r="bar">' + BAR
    + '<span class="si-barlab">' + STOP + 'no generator will do this</span></div></div>'

    + '<div class="si-cell dead si-rev" data-r="iout">'
    + '<div class="si-chd"><span class="si-clab">unsafe variant</span>'
    + '<span class="si-rlab">training row</span>'
    + '<span class="si-pill x">none</span></div>'
    + '<div class="si-shot"><div class="si-frame big empty">'
    + '<span class="si-qm">?</span></div>'
    + '<p class="si-mono si-thin">unsafe images cannot simply be generated the way unsafe text '
    + 'can be rewritten</p></div></div>'

    + '<p class="si-lfoot">positives can only come from existing image moderation sets, which '
    + 'are fewer, smaller and narrower'
    + (invLine ? '<span class="si-inv">' + esc(invLine) + '</span>' : '')
    + '</p>'
    + '</section>'

    + '</div>'

    /* ---- the workaround ---- */
    + '<section class="si-borrow si-rev" data-r="bor" aria-label="The workaround, borrowed negatives">'
    + '<div class="si-bhead">'
    + '<span class="si-bnum">03</span>'
    + '<div><b>So the pipeline borrows.</b> General purpose classification and object detection '
    + 'corpora hand over an enormous pool of naturally safe images.</div>'
    + '<div class="si-chips">'
    + '<span class="si-src cls">image classification</span>'
    + '<span class="si-src det">object detection</span>'
    + '</div></div>'
    + '<div class="si-tiles">' + tilesHtml + '</div>'
    + '<p class="si-bfoot">Each one is a high quality negative. Same three fields, a visual '
    + 'moderation query, and the answer is <b>no</b>.</p>'
    + '</section>'

    /* ---- footer, the mix and the honesty line ---- */
    + '<div class="si-mixwrap si-rev" data-r="mix">'
    + '<div class="si-mixlab">training mix'
    + '<ul class="si-keys">' + mixKey + '</ul>'
    + '<span class="si-tot">'
    + (totalM != null ? esc(String(totalM)) + 'M' : 'total not in the data file')
    + '</span></div>'
    + '<div class="si-mix">' + mixBar + '</div>'
    + '</div>'

    + '<div class="si-foot">'
    + '<p class="si-note">Lane structure is the paper method, section 3.4. Labels are the '
    + 'pipeline ground truth, not a live model call. Every picture here is an abstract glyph '
    + 'standing in for a photograph.</p>'
    + '<button class="si-btn" type="button" id="S_IMGSCARCE-replay">Replay</button></div>';

  root.appendChild(wrap);

  /* ---------------- reveal machinery ---------------- */

  var revs = {};
  [].slice.call(wrap.querySelectorAll('.si-rev')).forEach(function (n) {
    revs[n.getAttribute('data-r')] = n;
  });
  var tileEls = [].slice.call(wrap.querySelectorAll('.si-tile'));
  var pulseT = wrap.querySelector('.si-pulse[data-p="text"]');
  var pulseI = wrap.querySelector('.si-pulse[data-p="img"]');
  var btn = wrap.querySelector('#S_IMGSCARCE-replay');

  var CUE = [
    ['tsrc', 0.02], ['tarr', 0.10], ['tout', 0.26], ['tv', 0.30],
    ['isrc', 0.05], ['iarr', 0.14], ['bar', 0.36], ['iv', 0.39], ['iout', 0.44],
    ['bor', 0.56], ['mix', 0.90]
  ];
  var TILE0 = 0.60, TILEGAP = 0.028;
  var BARSTOP = 0.63;   /* fraction of the track the blocked pulse reaches */
  var PERIOD = 13;

  function setAll(on) {
    for (var k in revs) if (revs.hasOwnProperty(k)) revs[k].classList.toggle('on', on);
    tileEls.forEach(function (t) { t.classList.toggle('on', on); });
    if (pulseT) pulseT.style.opacity = '0';
    if (pulseI) pulseI.style.opacity = '0';
  }

  function frame(p) {
    CUE.forEach(function (c) {
      var n = revs[c[0]];
      if (n) n.classList.toggle('on', p >= c[1]);
    });
    tileEls.forEach(function (t, i) {
      t.classList.toggle('on', p >= TILE0 + i * TILEGAP);
    });

    /* text pulse, 0.10 to 0.26 */
    if (pulseT) {
      var a = (p - 0.10) / 0.16;
      if (a >= 0 && a <= 1.06) {
        pulseT.style.opacity = '1';
        pulseT.style.top = (Math.min(1, a) * 100).toFixed(2) + '%';
      } else {
        pulseT.style.opacity = '0';
      }
    }
    /* image pulse, 0.14 to 0.36, stops dead at the barrier */
    if (pulseI) {
      var b = (p - 0.14) / 0.22;
      if (b >= 0 && b <= 1.55) {
        pulseI.style.opacity = b > 1 ? String(Math.max(0, 1 - (b - 1) / 0.55)) : '1';
        pulseI.style.top = (Math.min(1, b) * BARSTOP * 100).toFixed(2) + '%';
      } else {
        pulseI.style.opacity = '0';
      }
    }
    var bar = revs.bar;
    if (bar) bar.classList.toggle('hit', p >= 0.36 && p < 0.46);
  }

  var t0 = null;
  var running = !api.reduce;

  if (api.reduce) {
    setAll(true);
    if (btn) btn.parentNode.removeChild(btn);
  } else {
    setAll(false);
    frame(0);
    btn.addEventListener('click', function () { t0 = null; setAll(false); frame(0); });
  }

  return {
    start: function () { if (running) { t0 = null; } },
    stop: function () { t0 = null; },
    tick: function (t) {
      if (!running) return;
      if (t0 == null) t0 = t;
      var p = (t - t0) / PERIOD;
      if (p >= 1) { t0 = t; p = 0; }
      frame(p);
    }
  };
};
