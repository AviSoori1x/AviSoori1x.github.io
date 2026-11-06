/* S_ADV: group-relative advantages (Equation 1, §4.4.4) and the groups that are
   dropped before the trainer because every reward in them is equal (§4.4.1).
   Group sizes are the paper's. The rewards, and the rest of the batch, are
   schematic: seeded values, so the figure draws the same way on every visit. */
window.SCENES.S_ADV = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var SIZES = [F['rl.group_min'], F['rl.group_default'], F['rl.group_code']];
  var LO = F['a4b.reward_lo'], HI = F['a4b.reward_hi'];
  var svg = K.board(host, { alt: 'A group of rollouts: rewards, the group mean, and the advantages they become' });
  K.head(svg, 'Advantage is relative to the group', '§4.4.1, §4.4.4 · Equation 1 · groups of ' + SIZES[0] +
    ' to ' + SIZES[2] + ' · rewards are schematic');

  var q20 = function (v) { return Math.round(v * 20) / 20; };
  function seeded(nn, s) {
    var rnd = CH.rng(s), out = [];
    for (var i = 0; i < nn; i++) {
      var u = rnd(), v = rnd();
      out.push(u < 0.3 ? LO : u < 0.55 ? HI : q20(LO + (HI - LO) * v));
    }
    if (out.every(function (v) { return v === out[0]; })) out[0] = out[0] ? LO : HI;
    return out;
  }
  function centred(rs) {
    var m = rs.reduce(function (a, b) { return a + b; }, 0) / rs.length;
    return { m: m, c: rs.map(function (v) { return v - m; }) };
  }
  var flatOf = function (rs) { return rs.every(function (v) { return Math.abs(v - rs[0]) < 1e-9; }); };

  // the rest of the batch, schematic: groups of eight; two of them are flat and dropped
  var REST = [];
  for (var g = 0; g < 7; g++) REST.push(g === 2 ? [HI, HI, HI, HI, HI, HI, HI, HI] :
    g === 5 ? [LO, LO, LO, LO, LO, LO, LO, LO] : seeded(8, 101 + g * 17));
  var restSS = 0, restN = 0;
  REST.forEach(function (rs) {
    if (flatOf(rs)) return;
    centred(rs).c.forEach(function (c) { restSS += c * c; restN++; });
  });

  var X0 = 44, PW = 496, RY = 100, AY = 228, PH = 86, AMAX = 2.8;
  var rewards = [], size = SIZES[1], sel = -1;
  var body = K.n('g', {}), out, HITS = [];   // attached after the controls

  function draw() {
    CH.clear(body);
    var nn = rewards.length, cs = centred(rewards), flat = flatOf(rewards);
    // Equation 1: centred rewards over their standard deviation across the kept batch
    var ss = restSS, cnt = restN;
    if (!flat) cs.c.forEach(function (c) { ss += c * c; cnt++; });
    var sd = Math.sqrt(ss / cnt);
    var A = cs.c.map(function (c) { return flat ? 0 : c / sd; });
    var slot = PW / nn, bw = Math.min(slot * 0.64, 46);
    var cx = function (i) { return X0 + slot * (i + 0.5); };
    var colOf = function (i) { return flat ? CH.GREY : A[i] > 1e-9 ? C.amber : A[i] < -1e-9 ? C.blue : CH.GREY; };

    // rewards
    var ra = CH.axes(body, X0, RY, PW, PH, { x0: 0, x1: 1, y0: LO, y1: HI, yt: [LO, (LO + HI) / 2, HI],
      yf: function (v) { return v === LO || v === HI ? String(v) : CH.fmt(v, 1); },
      yl: 'reward, one bar per rollout', noY: false });
    rewards.forEach(function (r, i) {
      var y = ra.sy(r);
      body.appendChild(K.n('rect', { x: cx(i) - bw / 2, y: Math.min(y, RY + PH - 2), width: bw,
        height: Math.max(2, RY + PH - y), rx: Math.min(4, bw / 3), fill: colOf(i),
        opacity: sel === i || sel < 0 ? 1 : 0.55 }));
    });
    var my = ra.sy(cs.m);
    body.appendChild(K.n('path', { d: 'M' + X0 + ' ' + my + ' H' + (X0 + PW), stroke: C.ink,
      'stroke-width': 1.6, 'stroke-dasharray': '5 4' }));
    K.mono(body, X0 + PW + 10, my + 4.5, 'mean ' + CH.fmt(cs.m, 2), { size: 12.5, color: C.ink, weight: 600 });

    // advantages
    var aa = CH.axes(body, X0, AY, PW, PH, { x0: 0, x1: 1, y0: -AMAX, y1: AMAX, yt: [-2, 0, 2],
      yf: function (v) { return v > 0 ? '+' + v : String(v); },
      yl: 'advantage: centered reward over the batch spread' });
    var zy = aa.sy(0);
    body.appendChild(K.n('path', { d: 'M' + X0 + ' ' + zy + ' H' + (X0 + PW), stroke: C.ink3, 'stroke-width': 1.2 }));
    A.forEach(function (a, i) {
      var v = Math.max(-AMAX, Math.min(AMAX, a)), y = aa.sy(v);
      if (Math.abs(y - zy) < 0.5) return;
      body.appendChild(K.n('rect', { x: cx(i) - bw / 2, y: Math.min(y, zy), width: bw,
        height: Math.abs(y - zy), rx: Math.min(3, bw / 3), fill: colOf(i),
        opacity: sel === i || sel < 0 ? 1 : 0.55 }));
    });
    if (flat) {
      K.mono(body, X0 + PW / 2, zy - 16, 'every advantage is zero', { size: 13.5, anchor: 'middle',
        color: C.ink, weight: 600 });
    }
    K.mono(body, X0 + PW + 10, aa.sy(1.6) + 4.5, 'more likely', { size: 12.5, color: C.ink2 });
    K.mono(body, X0 + PW + 10, aa.sy(-1.6) + 4.5, 'less likely', { size: 12.5, color: C.ink2 });

    // one hit column per rollout, over the reward panel
    rewards.forEach(function (r, i) {
      var h = K.n('rect', { x: X0 + slot * i, y: RY - 8, width: slot, height: PH + 12, fill: 'transparent',
        tabindex: 0, role: 'button', 'aria-label': 'rollout ' + (i + 1) + ', set its reward',
        style: 'cursor:ns-resize;outline:none' });
      h.addEventListener('click', function (e) {
        var p = CH.toLocal(h, e);
        rewards[i] = Math.max(LO, Math.min(HI, q20(LO + (RY + PH - p.y) / PH * (HI - LO))));
        sel = i; draw();
      });
      h.addEventListener('keydown', function (e) {
        var d = e.key === 'ArrowUp' ? 0.05 : e.key === 'ArrowDown' ? -0.05 : 0;
        if (!d) return;
        e.preventDefault();
        rewards[i] = Math.max(LO, Math.min(HI, q20(rewards[i] + d)));
        sel = i; draw();
        if (HITS[i]) HITS[i].focus();
      });
      body.appendChild(h);
    });
    HITS = body.querySelectorAll('rect[role=button]');

    out = CH.readout(body, 0, 338, { size: 13 });
    out.set(sel >= 0 ? 'Rollout ' + (sel + 1) + ': reward ' + CH.fmt(rewards[sel], 2) + ', group mean ' +
      CH.fmt(cs.m, 2) + ', advantage ' + (A[sel] > 0 ? '+' : '') + CH.fmt(A[sel], 2) :
      'Click a rollout\'s column to set its reward.', sel >= 0 ? C.ink : C.ink2);

    // the batch strip, schematic: this group outlined, dropped groups faded
    K.label(body, 0, 360, 'the batch, schematic');
    var groups = [rewards].concat(REST), BW = 40, BH = 44, BY = 370;
    groups.forEach(function (rs, j) {
      var bx = j * (BW + 7), dropped = flatOf(rs), mine = j === 0;
      var gg = K.n('g', { opacity: dropped ? 0.38 : 1 });
      body.appendChild(gg);
      gg.appendChild(K.n('rect', { x: bx, y: BY, width: BW, height: BH, rx: 5, fill: '#fff',
        stroke: mine ? C.amber : CH.GREY, 'stroke-width': mine ? 2 : 1,
        'stroke-dasharray': dropped ? '3 3' : null }));
      var w = (BW - 8) / rs.length;
      rs.forEach(function (r, k) {
        var hh = Math.max(1.5, (r - LO) / (HI - LO) * (BH - 10));
        gg.appendChild(K.n('rect', { x: bx + 4 + k * w + w * 0.12, y: BY + BH - 5 - hh,
          width: Math.max(0.8, w * 0.76), height: hh, fill: mine && !dropped ? C.amber : CH.GREY }));
      });
    });
    var TX = 8 * (BW + 7) + 16;
    K.mono(body, TX, BY + 11, 'outlined: this group, ' + (flat ? 'dropped' : 'adds ' + nn), { size: 12.5, color: C.ink });
    K.mono(body, TX, BY + 28, 'faded: dropped, rewards equal', { size: 12.5, color: C.ink2 });
    K.mono(body, TX, BY + 45, 'batch spread: ' + CH.fmt(sd, 2), { size: 12.5, color: C.ink2 });

    K.callout(body, 0, 426, 640, flat ?
      'Every reward in the group is the same, so every advantage is zero and the group carries ' +
        'no gradient. It is dropped before the trainer and does not count toward the ' +
        F['rl.step_traj_k'] + 'k batch.' :
      'Above the group mean the advantage is positive, so the rollout is made more likely; below ' +
        'it, less likely. The spread is taken across the whole batch, so a group whose rewards ' +
        'barely differ gets small advantages.', { cols: 74 });
  }

  var LAB = [SIZES[0] + ' rollouts, the fewest', SIZES[1] + ', the default', SIZES[2] + ', agentic coding'];
  var sw = K.switcher(svg, 0, 0, LAB, function (i) {
    size = SIZES[i]; rewards = seeded(size, 7 + i * 31); sel = -1;
    if (body.parentNode) draw();
  }, { tint: 'amber' });
  CH.button(svg, 0, 38, 'make them all equal', function () {
    var m = centred(rewards).m;
    rewards = rewards.map(function () { return m; }); sel = -1; draw();
  });
  CH.button(svg, 194, 38, 'reset rewards', function () {
    rewards = seeded(size, 7 + SIZES.indexOf(size) * 31); sel = -1; draw();
  }, { tint: 'ink3' });
  svg.appendChild(body);   // after the controls, so tab order runs controls first
  sw.pick(1);

  K.foot(svg, 'Group sizes and Equation 1 are the paper\'s; rewards and the rest of the batch are ' +
    'schematic. Spread: standard deviation of centered rewards over the kept groups.');
};
