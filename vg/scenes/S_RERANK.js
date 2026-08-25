window.SCENES = window.SCENES || {};

/* S_RERANK, act 2, beat 21. Asymmetric filtering.

   A vision-language reranker gives every image and query pair one agreement
   score. The pipeline then cuts the two pools at two different places: lenient
   on the rare violations so they survive, strict on the abundant negatives.

   The reader drags each cut independently on a shared score axis and watches
   the survivor counts move. The switch snaps both cuts onto the strict bar,
   which is the symmetric counterfactual, and the rare violations that only the
   lenient cut was keeping turn into a loss.

   The paper publishes no threshold values and no per-pool sample counts, so the
   two cuts, the pool sizes and every dot position here are drawn, not measured.
   The one measured number on screen is the multimodal share of the training
   mix, read from window.SS at runtime. */
window.SCENES['S_RERANK'] = function (root, api) {
  var SS = api.SS || {};
  var head = SS.headline || {};

  root.classList.add('sc-s_rerank');

  /* ---------------- illustrative pools ----------------
     One deterministic pseudo random stream so the picture is identical on
     every load and the only thing that ever moves is a cut. */
  var seed = 20250321;
  function rnd() {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  }
  function mk(n, lo, hi) {
    var out = [], i;
    for (i = 0; i < n; i++) {
      out.push({ s: lo + rnd() * (hi - lo), y: 7 + rnd() * 86, st: '' });
    }
    return out;
  }

  /* Violations are rare. Most of them the reranker scores high, but a real tail
     of them score low, and that tail is where a strict bar would do its damage. */
  var VI = mk(58, 0.60, 0.97).concat(mk(32, 0.09, 0.68));
  /* Negatives are abundant, mostly clean, with a doubtful tail that costs
     nothing to throw away. */
  var NE = mk(340, 0.74, 0.99).concat(mk(130, 0.08, 0.80));

  function shuffle(a) {
    var i, j, t;
    for (i = a.length - 1; i > 0; i--) {
      j = Math.floor(rnd() * (i + 1));
      t = a[i]; a[i] = a[j]; a[j] = t;
    }
  }
  shuffle(VI);
  shuffle(NE);

  var NVI = VI.length, NNE = NE.length;

  var POS0 = 0.30, NEG0 = 0.72;
  var pos = POS0, neg = NEG0;      /* the two cut values, 0 to 1 */
  var dispPos = POS0;              /* what the lenient handle is drawn at */
  var symOn = false;
  var touched = false;

  var mmShare = head.multimodalSamples;
  var mmTxt = (mmShare == null) ? null : api.num(mmShare, 1) + 'M';

  /* ---------------- markup ---------------- */
  function dotHtml(list) {
    var s = '', i;
    for (i = 0; i < list.length; i++) {
      s += '<span class="rk-dt" style="left:' + (list[i].s * 100).toFixed(2)
        + '%;top:' + list[i].y.toFixed(2) + '%"></span>';
    }
    return s;
  }

  function cutHtml(key, kind, label) {
    return '<span class="rk-cut rk-' + key + '" id="S_RERANK-cut-' + key + '"'
      + ' role="slider" tabindex="0" aria-orientation="horizontal"'
      + ' aria-label="' + label + '"'
      + ' aria-valuemin="0" aria-valuemax="1" aria-valuenow="0" aria-valuetext="0.00">'
      + '<i class="rk-hit" aria-hidden="true"></i>'
      + '<i class="rk-bar" aria-hidden="true"></i>'
      + '<i class="rk-grip" aria-hidden="true"></i>'
      + '<span class="rk-tag"><b>' + kind + '</b>'
      + '<em id="S_RERANK-val-' + key + '">0.00</em></span>'
      + '</span>';
  }

  root.appendChild(api.frag(
    '<div class="rk-wrap">'

    + '<div class="rk-hd">'
    +   '<div class="rk-hl">'
    +     '<span class="rk-eyebrow">vision language reranker</span>'
    +     '<span class="rk-hnote">one agreement score per image and query pair, then two '
    +     'different bars'
    +     (mmTxt ? ', on the way to the <b>' + mmTxt + '</b> multimodal samples in the mix' : '')
    +     '</span>'
    +   '</div>'
    +   '<button type="button" class="rk-sw" id="S_RERANK-sw" role="switch" aria-checked="false">'
    +     '<span class="rk-swt" aria-hidden="true"><i></i></span>'
    +     '<span class="rk-swl"><b>symmetric cut</b>'
    +     '<em id="S_RERANK-swsub">off, the two pools are cut apart</em></span>'
    +   '</button>'
    + '</div>'

    + '<div class="rk-board">'

    +   '<div class="rk-lrow">'
    +     '<span class="rk-pool"><b>violation pool</b>'
    +     '<i class="rk-rare">rare</i>'
    +     '<u>n = ' + NVI + '</u></span>'
    +     '<span class="rk-lnote">lenient, so the few real ones survive</span>'
    +   '</div>'

    +   '<div class="rk-lane rk-lpos" id="S_RERANK-lane-pos">'
    +     '<span class="rk-keep" id="S_RERANK-keep-pos" aria-hidden="true"></span>'
    +     '<span class="rk-band" id="S_RERANK-band" aria-hidden="true"></span>'
    +     '<span class="rk-ghost" id="S_RERANK-ghost" aria-hidden="true"></span>'
    +     '<span class="rk-dots" id="S_RERANK-dots-pos">' + dotHtml(VI) + '</span>'
    +     '<span class="rk-zl">dropped</span><span class="rk-zr">kept</span>'
    +     cutHtml('pos', 'lenient', 'Lenient cut for the rare violation pool')
    +   '</div>'

    +   '<div class="rk-gapstrip">'
    +     '<span class="rk-brk" id="S_RERANK-brk">'
    +       '<i aria-hidden="true"></i>'
    +       '<b id="S_RERANK-brktx"></b>'
    +     '</span>'
    +   '</div>'

    +   '<div class="rk-lane rk-lneg" id="S_RERANK-lane-neg">'
    +     '<span class="rk-keep" id="S_RERANK-keep-neg" aria-hidden="true"></span>'
    +     '<span class="rk-dots" id="S_RERANK-dots-neg">' + dotHtml(NE) + '</span>'
    +     '<span class="rk-zl">dropped</span><span class="rk-zr">kept</span>'
    +     cutHtml('neg', 'strict', 'Strict cut for the abundant negative pool')
    +   '</div>'

    +   '<div class="rk-lrow rk-lrowb">'
    +     '<span class="rk-pool"><b>negative pool</b>'
    +     '<i class="rk-abu">abundant</i>'
    +     '<u>n = ' + NNE + '</u></span>'
    +     '<span class="rk-lnote">strict, because a doubtful one costs nothing to bin</span>'
    +   '</div>'

    +   '<span class="rk-ticks" aria-hidden="true"></span>'
    +   '<div class="rk-ends">'
    +     '<span>0.00 &nbsp;reranker disagrees with the label</span>'
    +     '<span>0.50</span>'
    +     '<span>reranker agrees&nbsp; 1.00</span>'
    +   '</div>'

    + '</div>'

    + '<div class="rk-reads">'
    +   '<div class="rk-cell">'
    +     '<span class="rk-clab">violations kept</span>'
    +     '<b class="rk-big rk-lime" id="S_RERANK-kv">0</b>'
    +     '<span class="rk-csub" id="S_RERANK-sv"></span>'
    +   '</div>'
    +   '<div class="rk-cell">'
    +     '<span class="rk-clab">negatives kept</span>'
    +     '<b class="rk-big rk-lime" id="S_RERANK-kn">0</b>'
    +     '<span class="rk-csub" id="S_RERANK-sn"></span>'
    +   '</div>'
    +   '<div class="rk-cell rk-punch" id="S_RERANK-punch">'
    +     '<span class="rk-clab" id="S_RERANK-plab"></span>'
    +     '<b class="rk-big rk-amber" id="S_RERANK-kr">0</b>'
    +     '<span class="rk-csub" id="S_RERANK-sr"></span>'
    +   '</div>'
    + '</div>'

    + '<p class="rk-status" id="S_RERANK-status" role="status"></p>'

    + '<div class="rk-foot">'
    +   '<span class="rk-gt">The asymmetry is the paper\'s. Both cut values, both pool sizes and '
    +   'every dot are illustrative, the paper publishes no thresholds. Only the multimodal '
    +   'share is read from the data file.</span>'
    +   '<span class="rk-hint">drag either cut, or focus one and use the arrow keys</span>'
    + '</div>'

    + '</div>'
  ).firstChild);

  /* ---------------- handles ---------------- */
  var $ = function (id) { return root.querySelector('#S_RERANK-' + id); };
  var lanePos = $('lane-pos'), laneNeg = $('lane-neg');
  var cutPos = $('cut-pos'), cutNeg = $('cut-neg');
  var valPos = $('val-pos'), valNeg = $('val-neg');
  var keepPos = $('keep-pos'), keepNeg = $('keep-neg');
  var band = $('band'), ghost = $('ghost'), brk = $('brk'), brktx = $('brktx');
  var kv = $('kv'), kn = $('kn'), kr = $('kr');
  var sv = $('sv'), sn = $('sn'), sr = $('sr');
  var plab = $('plab'), punch = $('punch');
  var sw = $('sw'), swsub = $('swsub'), statusEl = $('status');

  var dotsVI = [].slice.call($('dots-pos').children);
  var dotsNE = [].slice.call($('dots-neg').children);

  function pct(a, b) { return b ? Math.round((a / b) * 100) : 0; }
  function fx(v) { return v.toFixed(2); }

  /* the same breakpoint the stylesheet uses, so the bracket caption can drop to
     a short form rather than wrapping over the lane below it */
  var narrow = matchMedia('(max-width:70rem)');
  if (narrow.addEventListener) {
    narrow.addEventListener('change', function () { apply(); });
  }

  /* ---------------- render ---------------- */
  function apply() {
    var effPos = dispPos;
    var lo = Math.min(symOn ? pos : dispPos, neg);
    var i, d, st, inBand, keptV = 0, keptN = 0, risk = 0;

    for (i = 0; i < NVI; i++) {
      d = VI[i];
      inBand = (d.s >= lo && d.s < neg);
      if (inBand) risk++;
      if (d.s >= effPos) { keptV++; st = inBand ? 'rk-dt rk-edge' : 'rk-dt rk-keepd'; }
      else { st = inBand ? 'rk-dt rk-lost' : 'rk-dt'; }
      if (d.st !== st) { d.st = st; dotsVI[i].className = st; }
    }
    for (i = 0; i < NNE; i++) {
      d = NE[i];
      if (d.s >= neg) { keptN++; st = 'rk-dt rk-keepd'; } else { st = 'rk-dt'; }
      if (d.st !== st) { d.st = st; dotsNE[i].className = st; }
    }

    keepPos.style.left = (effPos * 100).toFixed(2) + '%';
    keepNeg.style.left = (neg * 100).toFixed(2) + '%';

    cutPos.style.left = (effPos * 100).toFixed(2) + '%';
    cutNeg.style.left = (neg * 100).toFixed(2) + '%';
    cutPos.classList.toggle('rk-flip', effPos > 0.6);
    cutNeg.classList.toggle('rk-flip', neg > 0.6);
    cutPos.classList.toggle('rk-locked', symOn);
    cutPos.setAttribute('aria-valuenow', fx(symOn ? neg : pos));
    cutNeg.setAttribute('aria-valuenow', fx(neg));
    cutPos.setAttribute('aria-valuetext',
      fx(symOn ? neg : pos) + (symOn ? ', locked to the strict cut' : ''));
    cutNeg.setAttribute('aria-valuetext', fx(neg));
    cutPos.setAttribute('aria-disabled', symOn ? 'true' : 'false');
    cutPos.tabIndex = symOn ? -1 : 0;
    valPos.textContent = fx(effPos);
    valNeg.textContent = fx(neg);

    var w = Math.max(0, neg - lo);
    band.style.left = (lo * 100).toFixed(2) + '%';
    band.style.width = (w * 100).toFixed(2) + '%';
    band.classList.toggle('rk-cost', symOn);
    ghost.style.left = (dispPos * 100).toFixed(2) + '%';
    ghost.classList.toggle('rk-on', symOn);

    brk.style.left = (lo * 100).toFixed(2) + '%';
    brk.style.width = (w * 100).toFixed(2) + '%';
    brk.classList.toggle('rk-cost', symOn);
    /* the caption is centred on the bracket, unless that would push it off the
       board, in which case it hangs off whichever end has room */
    var mid = lo + w / 2;
    brk.classList.toggle('rk-atl', mid < 0.2);
    brk.classList.toggle('rk-atr', mid > 0.8);
    lanePos.classList.toggle('rk-off', symOn);
    brktx.textContent = narrow.matches
      ? (symOn ? 'gap 0.00 · ' + risk + ' lost' : 'gap ' + fx(w) + ' · ' + risk + ' rare')
      : (symOn
        ? 'gap 0.00, and ' + risk + ' rare violations go with it'
        : 'gap ' + fx(w) + ', ' + risk + ' rare violations live in here');

    kv.textContent = keptV;
    kn.textContent = keptN = countKept(NE, neg);
    sv.textContent = pct(keptV, NVI) + '% of ' + NVI + ', ' + (NVI - keptV) + ' dropped';
    sn.textContent = pct(keptN, NNE) + '% of ' + NNE + ', ' + (NNE - keptN) + ' dropped';

    kr.textContent = risk;
    kr.className = 'rk-big ' + (symOn ? 'rk-rose' : 'rk-amber');
    punch.classList.toggle('rk-on', symOn);
    plab.textContent = symOn
      ? 'rare violations the symmetric cut throws away'
      : 'rare violations only the lenient cut keeps';
    sr.textContent = symOn
      ? 'gone, and there are no more where they came from'
      : 'under the strict bar, over the lenient one';

    swsub.textContent = symOn
      ? 'on, one bar at ' + fx(neg) + ' for both pools'
      : 'off, the two pools are cut apart';
  }

  function countKept(list, cut) {
    var i, k = 0;
    for (i = 0; i < list.length; i++) if (list[i].s >= cut) k++;
    return k;
  }

  function announce() {
    var effPos = symOn ? neg : pos;
    statusEl.textContent = 'Lenient cut ' + fx(effPos) + ', strict cut ' + fx(neg) + '. '
      + countKept(VI, effPos) + ' of ' + NVI + ' violations kept, '
      + countKept(NE, neg) + ' of ' + NNE + ' negatives kept.';
  }

  /* ---------------- interaction ---------------- */
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function setPos(v) {
    if (symOn) return;            /* locked to the strict cut, nothing to move */
    pos = clamp(v, 0.02, neg);
    dispPos = pos;
    touched = true;
    easing = false;
    apply();
  }
  function setNeg(v) {
    neg = clamp(v, pos, 0.98);
    if (symOn) { dispPos = neg; }
    touched = true;
    easing = false;
    apply();
  }

  function fromEvent(lane, e) {
    var r = lane.getBoundingClientRect();
    return r.width ? clamp((e.clientX - r.left) / r.width, 0, 1) : 0;
  }

  function wireLane(lane, which) {
    lane.addEventListener('pointerdown', function (e) {
      if (which === 'pos' && symOn) return;
      e.preventDefault();
      try { lane.setPointerCapture(e.pointerId); } catch (err) { /* no live pointer */ }
      lane._drag = true;
      if (which === 'pos') { setPos(fromEvent(lane, e)); cutPos.focus(); }
      else { setNeg(fromEvent(lane, e)); cutNeg.focus(); }
    });
    lane.addEventListener('pointermove', function (e) {
      if (!lane._drag) return;
      if (which === 'pos') setPos(fromEvent(lane, e));
      else setNeg(fromEvent(lane, e));
    });
    function end(e) {
      if (!lane._drag) return;
      lane._drag = false;
      try {
        if (lane.hasPointerCapture && e.pointerId != null
          && lane.hasPointerCapture(e.pointerId)) {
          lane.releasePointerCapture(e.pointerId);
        }
      } catch (err) { /* nothing to release */ }
      announce();
    }
    lane.addEventListener('pointerup', end);
    lane.addEventListener('pointercancel', end);
  }
  wireLane(lanePos, 'pos');
  wireLane(laneNeg, 'neg');

  function keyOn(elem, which) {
    elem.addEventListener('keydown', function (e) {
      if (which === 'pos' && symOn) return;
      var cur = (which === 'pos') ? pos : neg;
      var step = (e.shiftKey ? 0.1 : 0.01);
      var v = null;
      var k = e.key;
      if (k === 'ArrowRight' || k === 'ArrowUp') v = cur + step;
      else if (k === 'ArrowLeft' || k === 'ArrowDown') v = cur - step;
      else if (k === 'PageUp') v = cur + 0.1;
      else if (k === 'PageDown') v = cur - 0.1;
      else if (k === 'Home') v = 0.02;
      else if (k === 'End') v = 0.98;
      if (v === null) return;
      e.preventDefault();
      if (which === 'pos') setPos(v); else setNeg(v);
      announce();
    });
  }
  keyOn(cutPos, 'pos');
  keyOn(cutNeg, 'neg');

  sw.addEventListener('click', function () {
    symOn = !symOn;
    touched = true;
    sw.setAttribute('aria-checked', symOn ? 'true' : 'false');
    if (api.reduce) { dispPos = symOn ? neg : pos; easing = false; }
    else { easing = true; }
    apply();
    announce();
  });

  /* ---------------- the opening reveal ----------------
     The scene arrives with both cuts on the strict bar, which is the wrong
     answer, then the lenient cut glides out and the tail of rare violations
     lights up. It runs once, and only if the reader has not already grabbed
     something. */
  var easing = false;
  var last = null, introAt = null, introTimer = null;

  dispPos = pos;
  apply();
  announce();

  /* belt and braces: if the animation pump is throttled or never runs, the
     reveal still has to land on the real asymmetric state rather than sit on
     the opening frame */
  function settle() {
    introTimer = null;
    if (touched) return;
    easing = false;
    dispPos = symOn ? neg : pos;
    apply();
  }

  return {
    start: function () {
      last = null; introAt = null;
      if (introTimer) { clearTimeout(introTimer); introTimer = null; }
      if (api.reduce || touched) return;
      dispPos = neg;
      easing = false;
      apply();
      introTimer = setTimeout(settle, 1900);
    },
    stop: function () {
      easing = false;
      introAt = null;
      if (introTimer) { clearTimeout(introTimer); introTimer = null; }
      if (!touched) { dispPos = pos; apply(); }
    },
    tick: function (t) {
      var dt = (last == null) ? 0 : Math.min(0.05, t - last);
      last = t;
      if (!touched) {
        if (introAt == null) { introAt = t; }
        else if (!easing && t - introAt > 0.75 && Math.abs(dispPos - pos) > 0.001) {
          easing = true;
        }
      }
      if (!easing) return;
      var tgt = symOn ? neg : pos;
      var d = tgt - dispPos;
      if (Math.abs(d) < 0.0015) { dispPos = tgt; easing = false; }
      else { dispPos += d * Math.min(1, dt * 7.5); }
      apply();
    }
  };
};
