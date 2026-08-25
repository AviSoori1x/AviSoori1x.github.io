(function () {
  var ID = 'w-softmax';
  var root = document.getElementById(ID);
  if (!root) return;

  var SS = (typeof window !== 'undefined' && window.SS) ? window.SS : null;

  function el(suffix) { return document.getElementById(ID + '-' + suffix); }

  var zYes = el('zyes');
  var zNo = el('zno');
  var tauIn = el('tau');
  if (!zYes || !zNo || !tauIn) return;

  var valYes = el('val-yes');
  var valNo = el('val-no');
  var valTau = el('val-tau');
  var barYes = el('bar-yes');
  var barNo = el('bar-no');
  var scoreOut = el('score');
  var verdictWrap = el('verdict-wrap');
  var verdictOut = el('verdict');
  var ruleOut = el('rule');
  var dirNote = el('dir');
  var segSafe = el('seg-safe');
  var segFlag = el('seg-flag');
  var tauTick = el('tau-tick');
  var tauLab = el('tau-lab');
  var tauNote = el('tau-note');
  var marker = el('marker');
  var markerLab = el('marker-lab');
  var mathBox = el('math');
  var identBox = el('ident');
  var presetsBox = el('presets');
  var quoteBox = el('quote');
  var frameBox = el('frame');
  var illusBox = el('illus');
  var costBox = el('cost');
  var costNote = el('cost-note');
  var costCounter = el('cost-counter');

  var LIM = 8;

  // Reported operating point. The paper thresholds the score at 0.5 for every
  // Shieldstral result it reports. Read from SS if the data layer ever carries
  // it, so this never drifts away from the source of truth.
  var TAU_PAPER = (SS && typeof SS.tau === 'number') ? SS.tau
    : (SS && SS.headline && typeof SS.headline.tau === 'number') ? SS.headline.tau
    : 0.5;

  var reduce = (typeof window.matchMedia === 'function') &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Illustrative logit pairs. Nothing here is a measured Shieldstral output;
  // they only place the sliders at three readable points on the curve. The
  // visible note under the presets says so.
  var PRESETS = [
    { name: 'strongly no', yes: -2.4, no: 2.6 },
    { name: 'near the boundary', yes: 0.5, no: 0.3 },
    { name: 'strongly yes', yes: 4.1, no: -1.4 }
  ];

  function pad(str, n) {
    var s = String(str);
    while (s.length < n) s = ' ' + s;
    return s;
  }

  function fmtExp(v) {
    if (v >= 1000) return v.toFixed(1);
    if (v >= 100) return v.toFixed(2);
    return v.toFixed(3);
  }

  function setText(node, txt) {
    if (node) node.textContent = txt;
  }

  function num(v, dp) {
    return (typeof v === 'number' && isFinite(v)) ? v.toFixed(dp === undefined ? 1 : dp) : null;
  }

  /* ---------- static content pulled from SS ---------- */

  var hasPrompt = !!(SS && typeof SS.systemPrompt === 'string' && SS.systemPrompt);

  if (quoteBox) {
    if (hasPrompt) {
      var q = document.createElement('span');
      q.className = 'ss-q';
      q.textContent = SS.systemPrompt;
      quoteBox.textContent = '';
      quoteBox.appendChild(q);
    } else {
      quoteBox.textContent = 'System prompt unavailable.';
    }
  }

  setText(frameBox,
    (hasPrompt ? 'That system prompt' : 'The fixed system prompt') +
    ' pins the answer to a single token. At that one output position the model ' +
    'unembeds towards just two token ids, the one for "yes" and the one for "no", and the rest of ' +
    'the vocabulary is thrown away. Those two numbers are the entire decision.');

  setText(illusBox,
    'Nothing here calls a model. You set both logits yourself, and the three presets are ' +
    'illustrative values picked to show the shape of the function, not measured Shieldstral ' +
    'outputs. The arithmetic underneath is exact.');

  setText(dirNote,
    's is the share of probability left on "yes" once yes and no are renormalised against each ' +
    'other, not a full vocabulary probability. Queries are written so that yes means the document ' +
    'matches the policy being asked about, so a high s is a flag and not a clean bill of health.');

  /* ---------- what each guardrail emits, grouped from SS.baselines ---------- */

  if (costBox) {
    var baselines = (SS && Array.isArray(SS.baselines)) ? SS.baselines : [];
    var total = baselines.length;

    var GROUPS = [
      {
        key: 'score',
        test: function (b) { return /^score$/i.test(String(b.output).trim()); },
        cost: 'a number comes back, so the threshold is yours to move'
      },
      {
        key: 'label',
        test: function (b) { return /^label$/i.test(String(b.output).trim()); },
        cost: 'a verdict comes back, already decided at whatever cut the implementation bakes in'
      },
      {
        key: 'reason',
        test: function (b) { return /reason/i.test(String(b.output)); },
        cost: 'a reasoning trace is decoded first, and only then the label'
      }
    ];

    if (!total) {
      costBox.textContent = 'Baseline table unavailable.';
    }

    if (total) {
      GROUPS.forEach(function (grp) {
        var members = baselines.filter(grp.test);
        if (!members.length) return;

        var label = members[0].output;
        var hasOurs = members.some(function (b) { return /ours/i.test(String(b.model)); });

        var row = document.createElement('div');
        row.className = 'ss-crow ss-crow-' + grp.key + (hasOurs ? ' is-ours' : '');

        var head = document.createElement('span');
        head.className = 'ss-cotype';
        head.textContent = 'emits ' + label;

        var count = document.createElement('span');
        count.className = 'ss-ccount';
        count.textContent = members.length + ' of ' + total;

        var who = document.createElement('span');
        who.className = 'ss-cwho';
        who.textContent = members.map(function (b) {
          return String(b.model).replace(/\s*\(ours\)\s*/i, '') + ' ' + b.size;
        }).join(', ');

        var cost = document.createElement('span');
        cost.className = 'ss-ccost';
        cost.textContent = grp.cost;

        row.appendChild(head);
        row.appendChild(count);
        row.appendChild(who);
        row.appendChild(cost);
        costBox.appendChild(row);
      });

      var oursRow = baselines.filter(function (b) { return /ours/i.test(String(b.model)); })[0];
      setText(costNote,
        'Shieldstral sits in the first group at ' + ((oursRow && oursRow.size) || 'its reported size') +
        '. The paper describes its verdict as a single forward pass with single-token output, which ' +
        'is where the efficiency argument comes from. It is not the only model in the table that ' +
        'hands back a score.');

      // Counterweight. Cheap output is not the same as best accuracy, and the
      // paper is explicit that a reasoning baseline leads the adaptability run.
      // Only name the leader when the data picks out exactly one candidate, so
      // a future edit to SS.baselines cannot silently attach the wrong name.
      var hl = (SS && SS.headline) ? SS.headline : {};
      var mine = num(hl.adaptabilityF1);
      var best = num(hl.adaptabilityBest);
      var leaders = baselines.filter(function (b) {
        return /reason/i.test(String(b.output)) && b.adaptive;
      });
      var leader = (leaders.length === 1) ? leaders[0] : null;

      if (mine && best && parseFloat(best) > parseFloat(mine)) {
        setText(costCounter,
          'A single token is cheaper, not better. On the adaptability benchmark the top F1 is ' + best +
          (leader ? ', held by ' + leader.model + ' ' + leader.size + ' with its reasoning trace,' : ',') +
          ' against Shieldstral at ' + mine + '. The paper argues efficiency at close accuracy, ' +
          'not a clean sweep.');
      } else if (mine && best) {
        setText(costCounter,
          'A single token is cheaper, not better. Shieldstral scores ' + mine +
          ' F1 on the adaptability benchmark against a best-in-table ' + best + '.');
      }
    }
  }

  /* ---------- presets ---------- */

  var buttons = [];
  if (presetsBox) {
    PRESETS.forEach(function (p, idx) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'ss-preset';
      b.textContent = p.name;
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', function () { runPreset(idx); });
      presetsBox.appendChild(b);
      buttons.push(b);
    });
  }

  function markPreset(idx) {
    buttons.forEach(function (b, i) {
      b.setAttribute('aria-pressed', i === idx ? 'true' : 'false');
    });
  }

  var anim = null;

  function stopAnim() {
    if (anim !== null && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(anim);
    anim = null;
  }

  function runPreset(idx) {
    var p = PRESETS[idx];
    if (!p) return;
    markPreset(idx);
    stopAnim();

    if (reduce || typeof requestAnimationFrame !== 'function') {
      zYes.value = String(p.yes);
      zNo.value = String(p.no);
      update();
      return;
    }

    // frame counted rather than clock driven, so a frozen or coarse clock
    // can never strand the sliders part way to the preset
    var y0 = parseFloat(zYes.value);
    var n0 = parseFloat(zNo.value);
    var frames = 26;
    var f = 0;

    function step() {
      f += 1;
      var k = Math.min(1, f / frames);
      var e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
      if (k >= 1) {
        zYes.value = String(p.yes);
        zNo.value = String(p.no);
      } else {
        zYes.value = (y0 + (p.yes - y0) * e).toFixed(1);
        zNo.value = (n0 + (p.no - n0) * e).toFixed(1);
      }
      update();
      anim = (k < 1) ? requestAnimationFrame(step) : null;
    }
    anim = requestAnimationFrame(step);
  }

  /* ---------- live maths ---------- */

  function update() {
    var zy = parseFloat(zYes.value);
    var zn = parseFloat(zNo.value);
    var tau = parseFloat(tauIn.value);
    if (!isFinite(zy)) zy = 0;
    if (!isFinite(zn)) zn = 0;
    if (!isFinite(tau)) tau = TAU_PAPER;

    // numerically stable softmax over exactly two logits
    var m = Math.max(zy, zn);
    var sy = Math.exp(zy - m);
    var sn = Math.exp(zn - m);
    var s = sy / (sy + sn);

    var ey = Math.exp(zy);
    var en = Math.exp(zn);
    var sum = ey + en;
    var flagged = s > tau;

    setText(valYes, (zy >= 0 ? '+' : '') + zy.toFixed(2));
    setText(valNo, (zn >= 0 ? '+' : '') + zn.toFixed(2));
    setText(valTau, tau.toFixed(2));

    // carry the outcome in the slider value text, so the figure is usable
    // without seeing the big number
    var spoken = ', score ' + s.toFixed(3) + ', ' + (flagged ? 'flagged' : 'not flagged');
    zYes.setAttribute('aria-valuetext', 'yes logit ' + zy.toFixed(1) + spoken);
    zNo.setAttribute('aria-valuetext', 'no logit ' + zn.toFixed(1) + spoken);
    tauIn.setAttribute('aria-valuetext', 'threshold ' + tau.toFixed(2) + spoken);

    // bars, drawn from the zero line at x = 100 in a 0..200 viewBox
    setBar(barYes, zy);
    setBar(barNo, zn);

    setText(scoreOut, s.toFixed(3));
    setText(verdictOut, flagged ? 'flagged' : 'not flagged');
    setText(ruleOut, flagged ? 's is above tau' : 's is at or below tau');
    if (verdictWrap) {
      if (flagged) { verdictWrap.classList.add('is-flagged'); }
      else { verdictWrap.classList.remove('is-flagged'); }
    }

    if (tauNote) {
      var moved = Math.abs(tau - TAU_PAPER) > 0.004;
      tauNote.textContent = moved
        ? 'Moved off the reported setting. The paper thresholds the score at tau = ' +
          TAU_PAPER.toFixed(2) + '.'
        : 'This is the reported setting. The paper thresholds the score at tau = ' +
          TAU_PAPER.toFixed(2) + '.';
      if (moved) { tauNote.classList.add('is-moved'); }
      else { tauNote.classList.remove('is-moved'); }
    }

    // 0..1 track
    var tauPct = tau * 100;
    var sPct = s * 100;
    if (segSafe) segSafe.style.width = tauPct + '%';
    if (segFlag) segFlag.style.width = (100 - tauPct) + '%';
    if (tauTick) tauTick.style.left = tauPct + '%';
    setText(tauLab, 'tau ' + tau.toFixed(2));
    edgeClamp(tauLab, tau);
    if (marker) {
      marker.style.left = sPct + '%';
      if (flagged) { marker.classList.add('is-flagged'); }
      else { marker.classList.remove('is-flagged'); }
    }
    setText(markerLab, 's ' + s.toFixed(3));
    edgeClamp(markerLab, s);

    // written-out arithmetic with the current numbers substituted in
    var zys = pad((zy >= 0 ? '+' : '') + zy.toFixed(2), 5);
    var zns = pad((zn >= 0 ? '+' : '') + zn.toFixed(2), 5);
    var eys = fmtExp(ey);
    var ens = fmtExp(en);
    var sums = fmtExp(sum);
    var w = Math.max(eys.length, sums.length);

    var lines = [
      's = exp(z_yes) / (exp(z_yes) + exp(z_no))',
      '  = exp(' + zys + ') / (exp(' + zys + ') + exp(' + zns + '))',
      '  = ' + pad(eys, w) + ' / (' + eys + ' + ' + ens + ')',
      '  = ' + pad(eys, w) + ' / ' + pad(sums, w),
      '  = ' + s.toFixed(3)
    ];

    if (mathBox) {
      mathBox.textContent = '';
      lines.forEach(function (line, i) {
        var d = document.createElement('div');
        d.className = 'ss-mline ss-mline-' + i + (i === lines.length - 1 ? ' ss-mline-last' : '');
        d.textContent = line;
        mathBox.appendChild(d);
      });
    }

    setText(identBox,
      'equivalently s = sigmoid(z_yes - z_no), gap = ' +
      ((zy - zn) >= 0 ? '+' : '') + (zy - zn).toFixed(2));
  }

  // keep the floating labels inside the figure at the two ends of the track
  function edgeClamp(node, frac) {
    if (!node) return;
    if (frac < 0.1) { node.style.transform = 'translateX(-3px)'; }
    else if (frac > 0.9) { node.style.transform = 'translateX(-100%) translateX(3px)'; }
    else { node.style.transform = 'translateX(-50%)'; }
  }

  function setBar(rect, z) {
    if (!rect) return;
    var half = 98;
    var w = Math.abs(z) / LIM * half;
    rect.setAttribute('x', String(z >= 0 ? 100 : 100 - w));
    rect.setAttribute('width', String(w));
  }

  function onManual() {
    markPreset(-1);
    stopAnim();
    update();
  }

  zYes.addEventListener('input', onManual);
  zNo.addEventListener('input', onManual);
  tauIn.addEventListener('input', function () { update(); });

  // open at the reported threshold, on the near boundary case, so the reader
  // lands mid decision
  tauIn.value = String(TAU_PAPER);
  zYes.value = String(PRESETS[1].yes);
  zNo.value = String(PRESETS[1].no);
  markPreset(1);
  update();
})();
