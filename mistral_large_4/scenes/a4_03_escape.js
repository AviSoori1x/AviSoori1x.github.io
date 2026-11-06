/* S_ESCAPE: the sandbox escape of §4.3.3, told one step at a time. Schematic: the
   boxes show which boundary each step crossed, not the real topology. The paper says
   the agent chained four vulnerabilities but not which steps they were, so the steps
   are numbered as events in the paper's order, never as vulnerabilities. */
window.SCENES.S_ESCAPE = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var svg = K.board(host, { alt: 'Schematic of how an agent escaped its sandbox, one step at a time (section 4.3.3)' });
  K.head(svg, 'Out of the sandbox, onto the host', '§4.3.3 · one sandbox among millions per run · schematic');

  var EDGE = 'rgba(31,37,48,.32)';
  function rect(p, x, y, w, h, o) {
    o = o || {};
    var r = K.n('rect', { x: x, y: y, width: w, height: h, rx: o.r == null ? 9 : o.r,
      fill: o.fill || '#fff', stroke: o.stroke || EDGE, 'stroke-width': o.sw || 1.2,
      'stroke-dasharray': o.dash || null });
    p.appendChild(r);
    return r;
  }
  function name(p, x, y, s, o) {
    o = o || {};
    return K.mono(p, x, y, s, { size: o.size || 13.5, weight: o.weight || 600, color: o.color || C.ink });
  }
  // a click target that the checker and a keyboard can both reach
  function ctl(g, label, on) {
    g.setAttribute('role', 'button');
    g.setAttribute('tabindex', '0');
    g.setAttribute('aria-label', label);
    g.style.cursor = 'pointer';
    g.style.outline = 'none';
    g.addEventListener('click', on);
    g.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); on(); }
    });
  }

  /* ---- the nested boundaries (static) ---- */
  var B = {};
  B.node = K.panel(svg, 0, 50, 400, 352, { r: 12 }).firstChild;
  name(svg, 16, 74, 'node');
  B.pod = rect(svg, 16, 88, 368, 206, { dash: '6 4', fill: '#fcfbf8' });
  name(svg, 30, 110, 'pod');
  B.sandbox = rect(svg, 32, 122, 336, 70);
  name(svg, 46, 144, 'sandbox');
  B.sidecar = rect(svg, 32, 208, 336, 72);
  name(svg, 46, 230, 'privileged sidecar');
  B.host = rect(svg, 16, 330, 368, 58, { fill: '#f6f4ef' });
  name(svg, 30, 353, 'host');

  B.traffic = K.panel(svg, 420, 50, 220, 86, { r: 10 }).firstChild;
  name(svg, 434, 76, 'network traffic', { size: 13 });
  B.cp = K.panel(svg, 420, 166, 220, 74, { r: 10 }).firstChild;
  name(svg, 434, 192, 'sandbox control plane', { size: 13 });
  K.mono(svg, 434, 214, 'undocumented deployment', { size: 12.5, color: C.ink3 });

  var body = CH.body(svg);
  // reserve the tallest callout, so the foot (placed once) never lands on it
  // (the callout wraps at 72 columns at this width; the tallest step text sets the height)
  var RESERVE = 0;

  /* ---- an arrow with a correct head: horizontal ends for curves, straight when v ---- */
  function link(p, x1, y1, x2, y2, o) {
    o = o || {};
    var col = o.color || C.amber, d, a;
    if (o.v) { d = 'M' + x1 + ' ' + y1 + ' L' + x2 + ' ' + y2; a = Math.atan2(y2 - y1, x2 - x1); }
    else {
      var mx = o.mx == null ? (x1 + x2) / 2 : o.mx;
      d = 'M' + x1 + ' ' + y1 + ' C' + mx + ' ' + y1 + ' ' + mx + ' ' + y2 + ' ' + x2 + ' ' + y2;
      a = x2 >= x1 ? 0 : Math.PI;
    }
    var s = o.on ? 9 : 7;
    // stop the line short of the tip, so the head stays sharp
    p.appendChild(K.n('path', { d: d, fill: 'none', stroke: col, 'stroke-width': o.on ? 2.6 : 1.6,
      'stroke-dasharray': o.dash || null, 'stroke-linecap': 'round' }));
    p.appendChild(K.n('path', { d: 'M' + x2 + ' ' + y2 +
      ' L' + (x2 - s * Math.cos(a - 0.45)) + ' ' + (y2 - s * Math.sin(a - 0.45)) +
      ' L' + (x2 - s * Math.cos(a + 0.45)) + ' ' + (y2 - s * Math.sin(a + 0.45)) + ' Z', fill: col }));
  }

  /* ---- the chain, in the paper's order. Links are drawn from this list. ---- */
  var LINKS = [
    [420, 105, 370, 150],            // 1: the key comes out of the traffic, into the sandbox
    [368, 178, 420, 186, 'dash'],    // 2: it finds the control plane
    [420, 210, 300, 200],            // 3: the key, used against its own pod
    [420, 230, 370, 250],            // 4: commands on the privileged sidecar
    [200, 280, 200, 328, 'v']        // 5: a kernel panic, down to the host
  ];
  var DEL = F['esc.deleted_k'];
  var STEPS = [
    { lit: ['traffic', 'sandbox'], key: 1, at: 'sandbox',
      say: 'Step 1 of 6. It sniffed network traffic and grabbed an identity-scoped API key.' },
    { lit: ['cp'], key: 0, at: 'sandbox',
      say: 'Step 2 of 6. It found an undocumented deployment of the sandbox control plane.' },
    { lit: ['pod'], key: 1, at: 'sandbox',
      say: 'Step 3 of 6. It used the key against its own pod to run arbitrary commands.' },
    { lit: ['sidecar'], key: 0, at: 'sidecar',
      say: 'Step 4 of 6. The API allowed it to run commands on a privileged sidecar.' },
    { lit: ['host', 'node'], key: 0, at: 'host', sub: 'root access to the node',
      say: 'Step 5 of 6. Through a kernel panic it reached the host, with root access to the node.' },
    { lit: ['host'], key: 0, at: 'host', sub: 'no CTF here, so it stopped',
      say: 'Step 6 of 6. On the host, it realized the CTF was not there, and stopped.' },
    { lit: ['cp'], key: 0, at: 'other', other: 1,
      say: 'Another agent escaped isolation by a similar mechanism and probed the creation of new ' +
        'sandboxes through the API. Once that worked, it tried to clean up after itself and ' +
        'deleted about ' + DEL + 'k sandboxes by mistake.' },
    { lit: ['sandbox'], key: 0, at: null, after: 1,
      say: 'Both escapes triggered alerts, so the run could restart after remediation. The ' +
        'sandboxes, mainly containers earlier this year, are being rebuilt around microVMs for ' +
        'stronger isolation.' }
  ];
  var AT = { sandbox: [290, 158], sidecar: [290, 246], host: [290, 359], other: [36, 312] };
  STEPS.forEach(function (st) { RESERVE = Math.max(RESERVE, K.wrap(st.say, 72).length); });
  svg.appendChild(K.n('rect', { x: 0, y: 420, width: 640, height: RESERVE * 19 + 22, fill: 'none' }));

  /* the paper's own count for this escape, kept apart from the steps on purpose */
  function drawFour() {
    K.label(body, 420, 280, 'chained in this escape');
    K.big(body, 418, 326, F['esc.vulns'], { size: 42 });
    K.mono(body, 420, 350, 'different vulnerabilities', { size: 13, weight: 600, color: C.ink });
    K.mono(body, 420, 372, 'the paper does not say which', { size: 12.5, color: C.ink3 });
    K.mono(body, 420, 390, 'steps they were', { size: 12.5, color: C.ink3 });
  }

  /* ---- draw one state ---- */
  var BASE = {};
  Object.keys(B).forEach(function (k) {
    BASE[k] = { fill: B[k].getAttribute('fill'), stroke: B[k].getAttribute('stroke'),
      sw: B[k].getAttribute('stroke-width') };
  });
  var cur = 0, dots = [], pills = [];

  function draw(s) {
    cur = s;
    var st = STEPS[s];
    CH.clear(body);
    Object.keys(B).forEach(function (k) {
      var on = st.lit.indexOf(k) >= 0, b = B[k];
      // the node is the whole machine: outline it, never flood it
      b.setAttribute('stroke', on ? C.amber : BASE[k].stroke);
      b.setAttribute('stroke-width', on ? 2.2 : BASE[k].sw);
      b.setAttribute('fill', on && k !== 'node' ? K.TINT.amber : BASE[k].fill);
    });

    // the links walked so far: earlier ones grey, the current one amber
    if (s < 6) {
      LINKS.forEach(function (L, i) {
        if (i > s) return;
        link(body, L[0], L[1], L[2], L[3], { on: i === s, color: i === s ? C.amber : CH.GREY,
          dash: L[4] === 'dash' ? '5 4' : null, v: L[4] === 'v' });
      });
    }

    // the key the agent sniffed, lit when it is taken and when it is used
    {
      K.chip(body, 432, 93, 'identity-scoped API key', { size: 12.5,
        fill: st.key ? K.TINT.amber : '#fff', stroke: st.key ? C.amber : C.line,
        color: st.key ? C.ink : C.ink3 });
    }

    // the host strip says what happened there
    K.mono(body, 30, 374, st.sub || 'the node itself, outside the pod', { size: 12.5,
      color: st.sub ? C.ink : C.ink3, weight: st.sub ? 600 : 400 });

    // the agent, wherever it is now
    if (st.at) {
      var p = AT[st.at];
      body.appendChild(K.n('circle', { cx: p[0], cy: p[1], r: 8, fill: C.amber,
        stroke: '#fff', 'stroke-width': 2.4 }));
      K.mono(body, p[0] + 14, p[1] + 4.5, st.other ? 'another agent, out of isolation' : 'agent',
        { size: 13, weight: 600, color: C.ink });
    }
    if (s < 6) drawFour();
    if (s === 4 || s === 5) K.mono(body, 212, 318, 'kernel panic', { size: 12.5, weight: 600, color: C.ink });
    if (st.other) drawOther();
    if (st.after) drawAfter();

    K.callout(body, 0, 420, 640, st.say, { cols: 74 });
    dots.forEach(function (d, i) {
      var on = i === s, done = i < s;
      d.firstChild.setAttribute('fill', on ? C.amber : (done ? K.TINT.amber : '#fff'));
      d.firstChild.setAttribute('stroke', on || done ? C.amber : EDGE);
      d.lastChild.setAttribute('fill', on ? '#fff' : (done ? C.ink : C.ink3));
    });
    pills.forEach(function (g, i) {
      var on = i + 6 === s;
      g.firstChild.setAttribute('fill', on ? K.TINT.amber : 'none');
      g.firstChild.setAttribute('stroke', on ? C.amber : C.line);
      g.lastChild.setAttribute('fill', on ? C.amber : C.ink3);
    });
  }

  /* the second incident: the same kind of foothold, then sandboxes deleted in a clean-up */
  function drawOther() {
    link(body, 314, 312, 420, 228, { on: true, mx: 410 });
    link(body, 608, 242, 608, 278, { on: true, v: true });
    [[596, 280], [588, 288], [580, 296]].forEach(function (q) {
      rect(body, q[0], q[1], 40, 30, { r: 5 });
    });
    // red marks the damage, and the label above it says what it is
    body.appendChild(K.n('path', { d: 'M585 301 L615 321 M615 301 L585 321', stroke: C.red,
      'stroke-width': 2.6, 'stroke-linecap': 'round' }));
    K.label(body, 420, 280, 'deleted by mistake');
    K.mono(body, 420, 324, 'about', { size: 13, color: C.ink2 });
    K.big(body, 468, 326, DEL + 'k', { size: 42 });
    K.mono(body, 420, 350, 'sandboxes', { size: 13, weight: 600, color: C.ink });
    K.mono(body, 420, 372, 'while it tried to clean', { size: 12.5, color: C.ink3 });
    K.mono(body, 420, 390, 'up after itself', { size: 12.5, color: C.ink3 });
  }

  /* the aftermath: alerts, a restart, and a stronger box for the next runs */
  function drawAfter() {
    K.mono(body, 46, 166, 'moving from containers to microVMs', { size: 13, weight: 600, color: C.ink });
    ['alerts fired', 'remediation', 'run restarted'].forEach(function (t, i) {
      K.chip(body, 420, 264 + i * 50, t, { w: 220, h: 27, size: 13, fill: '#fff', stroke: C.amber,
        color: C.ink });
      if (i) link(body, 530, 243 + i * 50, 530, 262 + i * 50, { on: true, v: true });
    });
  }

  /* ---- controls: six steps of the chain, the second incident, the aftermath ---- */
  for (var i = 0; i < 6; i++) {
    var cx = 14 + i * 36;
    if (i) svg.appendChild(K.n('path', { d: 'M' + (cx - 23) + ' 13.5 H' + (cx - 13), stroke: EDGE,
      'stroke-width': 2.2 }));
    var g = K.n('g', {});
    g.appendChild(K.n('circle', { cx: cx, cy: 13.5, r: 13, fill: '#fff', stroke: EDGE, 'stroke-width': 1.4 }));
    K.mono(g, cx, 18, String(i + 1), { size: 13, weight: 600, anchor: 'middle' });
    svg.appendChild(g);
    ctl(g, 'step ' + (i + 1), (function (k) { return function () { draw(k); }; })(i));
    dots.push(g);
  }
  var px = 226;
  ['another escape', 'aftermath'].forEach(function (t, j) {
    var g2 = K.chip(svg, px, 0, t, { h: 27, size: 12.5 });
    ctl(g2, t, function () { draw(6 + j); });
    pills.push(g2);
    px += g2._w + 8;
  });
  CH.button(svg, 570, -1.5, 'next', function () { draw((cur + 1) % STEPS.length); }, { w: 70 });

  draw(0);
  K.foot(svg, 'Schematic: the boxes mark the boundaries each step crossed, not the real layout. ' +
    'Steps are numbered in the paper\'s order of events, not as vulnerabilities.');
};
