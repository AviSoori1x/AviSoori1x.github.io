window.SCENES = window.SCENES || {};

/* Act II, scene 10. Strictness tiers.
   SS.strictness holds three tiers. Each is one detent on a switch, and each
   detent drags a decision boundary along a severity axis, so the very same
   cloud of content gets flagged more often or less often.
   Every tier name, domain list and rationale is read from window.SS at runtime.
   The boundary position itself is drawn, the paper does not publish one. */
window.SCENES['S_STRICT'] = function (root, api) {
  var SS = api.SS || {};
  var TIERS = (SS.strictness || []).filter(function (t) { return t && t.level; });
  var N = TIERS.length;
  var DOT = '·';

  root.className = 'sc-s_strict';

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  if (!N) {
    root.appendChild(api.frag('<div class="wrap"><div class="hd">'
      + '<span class="eyebrow">strictness tiers</span>'
      + '<span class="hnote">SS.strictness is empty, nothing to draw</span>'
      + '</div></div>').firstChild);
    return;
  }

  /* Where each tier's line sits. Illustrative, spread evenly across the middle
     of the axis so the leftmost tier flags the most and the rightmost the least. */
  function boundary(i) {
    return N < 2 ? 0.5 : 0.30 + (i / (N - 1)) * 0.44;
  }

  /* A deterministic cloud, so the picture is identical on every load and the
     only thing that ever moves between tiers is the line. */
  var seed = 1710243101;
  function rnd() {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  }
  var SAMPLES = [];
  var i, k, s, y;
  for (i = 0; i < 46; i++) {
    s = 0.035 + rnd() * 0.93;
    for (k = 0; k < N; k++) {
      if (Math.abs(s - boundary(k)) < 0.016) s += (s < boundary(k) ? -0.024 : 0.024);
    }
    /* the band leaves the top clear for the boundary caption and the bottom
       clear for the two zone labels, at both stage sizes */
    y = 26 + rnd() * 53;
    SAMPLES.push({ s: Math.min(0.975, Math.max(0.025, s)), y: y });
  }

  /* ---------- markup ---------- */
  var btns = '', leads = '', ghosts = '', marks = '', dots = '';
  for (i = 0; i < N; i++) {
    var lv = String(TIERS[i].level);
    var bx = boundary(i) * 100;
    btns += '<button type="button" class="tb" role="radio" aria-checked="false" tabindex="-1"'
      + ' id="S_STRICT-tb-' + i + '"><span class="tbn">' + esc(lv) + '</span>'
      + '<span class="tbd" aria-hidden="true"></span></button>';
    leads += '<path class="ld ld-' + i + '" d="M' + ((i + 0.5) * (100 / N)).toFixed(2) + ' 0 C'
      + ((i + 0.5) * (100 / N)).toFixed(2) + ' 24,' + bx.toFixed(2) + ' 15,' + bx.toFixed(2)
      + ' 40" vector-effect="non-scaling-stroke"></path>';
    ghosts += '<span class="ghost gh-' + i + '" style="left:' + bx.toFixed(2) + '%"></span>';
    marks += '<span class="mk mk-' + i + '" style="left:' + bx.toFixed(2) + '%">'
      + '<i aria-hidden="true"></i><b>' + esc(lv) + '</b></span>';
  }
  for (i = 0; i < SAMPLES.length; i++) {
    dots += '<span class="dt" style="left:' + (SAMPLES[i].s * 100).toFixed(2) + '%;top:'
      + SAMPLES[i].y.toFixed(2) + '%"></span>';
  }

  root.appendChild(api.frag(
    '<div class="wrap">'

    + '<div class="hd">'
    +   '<span class="eyebrow">strictness tiers</span>'
    +   '<span class="hnote">the tier a dataset is given decides how much has to be there '
    +   'before the answer is yes</span>'
    + '</div>'

    + '<div class="track" role="radiogroup" aria-label="strictness tier" id="S_STRICT-track">'
    +   '<span class="knob" id="S_STRICT-knob" aria-hidden="true"></span>' + btns
    + '</div>'

    + '<svg class="lead" viewBox="0 0 100 40" preserveAspectRatio="none" aria-hidden="true">'
    +   leads
    + '</svg>'

    + '<div class="axis">'
    +   '<div class="arow">'
    +     '<span class="alab">content severity</span>'
    +     '<span class="cnt" id="S_STRICT-cnt" aria-live="polite"></span>'
    +   '</div>'
    +   '<div class="lane">'
    +     '<span class="zone" id="S_STRICT-zone"></span>'
    +     ghosts
    +     '<span class="dots">' + dots + '</span>'
    +     '<span class="bline" id="S_STRICT-bline">'
    +       '<span class="bcap"><b>decision boundary</b><i>illustrative</i></span>'
    +     '</span>'
    +     '<span class="zlab zl">not flagged ' + DOT + ' no</span>'
    +     '<span class="zlab zr">flagged ' + DOT + ' yes</span>'
    +   '</div>'
    +   '<span class="ticks" aria-hidden="true"></span>'
    +   '<div class="marks">' + marks + '</div>'
    +   '<div class="ends"><span>less severe</span><span>more severe</span></div>'
    + '</div>'

    + '<div class="detail">'
    +   '<div class="dtop">'
    +     '<div class="dname">'
    +       '<span class="dlab">selected tier</span>'
    +       '<b class="dbig" id="S_STRICT-name"></b>'
    +       '<span class="dpos" id="S_STRICT-pos"></span>'
    +     '</div>'
    +     '<div class="drat">'
    +       '<span class="dlab">rationale, the paper\'s words</span>'
    +       '<p class="drtx" id="S_STRICT-rat"></p>'
    +     '</div>'
    +   '</div>'
    +   '<div class="ddom">'
    +     '<span class="dlab">example domains</span>'
    +     '<div class="chips" id="S_STRICT-chips"></div>'
    +   '</div>'
    + '</div>'

    + '<div class="foot">'
    +   '<span class="gt">Tier names, domains and rationales are the paper\'s. The boundary '
    +   'position and the sample cloud are drawn to show the idea, the paper publishes neither.</span>'
    +   '<span class="hint" id="S_STRICT-hint">cycling on its own, click or use arrow keys to '
    +   'take over</span>'
    + '</div>'

    + '</div>'
  ).firstChild);

  /* ---------- handles ---------- */
  var track = root.querySelector('#S_STRICT-track');
  var knob = root.querySelector('#S_STRICT-knob');
  var zone = root.querySelector('#S_STRICT-zone');
  var bline = root.querySelector('#S_STRICT-bline');
  var cnt = root.querySelector('#S_STRICT-cnt');
  var nameEl = root.querySelector('#S_STRICT-name');
  var posEl = root.querySelector('#S_STRICT-pos');
  var ratEl = root.querySelector('#S_STRICT-rat');
  var chips = root.querySelector('#S_STRICT-chips');
  var hint = root.querySelector('#S_STRICT-hint');
  var lead = root.querySelector('.lead');
  var tbs = [], ghostEls = [], markEls = [];
  for (i = 0; i < N; i++) {
    tbs.push(root.querySelector('#S_STRICT-tb-' + i));
    ghostEls.push(root.querySelector('.ghost.gh-' + i));
    markEls.push(root.querySelector('.mk.mk-' + i));
  }
  var dotEls = Array.prototype.slice.call(root.querySelectorAll('.dt'));

  track.style.gridTemplateColumns = 'repeat(' + N + ',1fr)';
  knob.style.width = 'calc((100% - 10px) / ' + N + ')';

  var cur = -1;
  var auto = true;

  function setTier(idx, fromUser) {
    if (idx === cur || idx < 0 || idx >= N) return;
    var t = TIERS[idx];
    var pos = boundary(idx);
    var j, hits = 0;

    for (j = 0; j < N; j++) {
      tbs[j].setAttribute('aria-checked', j === idx ? 'true' : 'false');
      tbs[j].tabIndex = j === idx ? 0 : -1;
      ghostEls[j].classList.toggle('on', j === idx);
      markEls[j].classList.toggle('on', j === idx);
      lead.querySelector('.ld-' + j).classList.toggle('on', j === idx);
    }
    knob.style.transform = 'translateX(' + (idx * 100) + '%)';

    zone.style.width = ((1 - pos) * 100).toFixed(2) + '%';
    bline.style.left = (pos * 100).toFixed(2) + '%';
    bline.classList.toggle('flip', pos > 0.62);

    for (j = 0; j < dotEls.length; j++) {
      var on = SAMPLES[j].s >= pos;
      if (on) hits++;
      dotEls[j].style.transitionDelay = api.reduce
        ? '0s' : (Math.abs(SAMPLES[j].s - pos) * 0.3).toFixed(3) + 's';
      dotEls[j].classList.toggle('hit', on);
    }

    cnt.innerHTML = '<b>' + hits + '</b> of ' + dotEls.length
      + ' illustrative samples flagged';

    nameEl.textContent = String(t.level);
    posEl.textContent = 'tier ' + (idx + 1) + ' of ' + N;
    ratEl.textContent = String(t.rationale == null ? '' : t.rationale);

    chips.innerHTML = '';
    String(t.domains == null ? '' : t.domains).split(/,\s*/).forEach(function (d) {
      if (!d) return;
      chips.appendChild(api.el('span', 'chp', d));
    });

    if (fromUser && auto) {
      auto = false;
      hint.textContent = 'manual, arrow keys move the switch';
    }
    cur = idx;
  }

  tbs.forEach(function (b, j) {
    b.addEventListener('click', function () { setTier(j, true); });
  });
  track.addEventListener('keydown', function (e) {
    var k2 = e.key, nx = -1;
    if (k2 === 'ArrowLeft' || k2 === 'ArrowUp') nx = (cur - 1 + N) % N;
    else if (k2 === 'ArrowRight' || k2 === 'ArrowDown') nx = (cur + 1) % N;
    else if (k2 === 'Home') nx = 0;
    else if (k2 === 'End') nx = N - 1;
    if (nx < 0) return;
    e.preventDefault();
    setTier(nx, true);
    tbs[nx].focus();
  });

  setTier(0, false);

  /* auto cycle, so a reader who only scrolls still sees the line move */
  var running = false, nextAt = null;
  return {
    start: function () { running = true; nextAt = null; },
    stop: function () { running = false; },
    tick: function (t) {
      if (!running || !auto || api.reduce) return;
      if (nextAt === null) { nextAt = t + 2.2; return; }
      if (t >= nextAt) {
        nextAt = t + 3.4;
        setTier((cur + 1) % N, false);
      }
    }
  };
};
