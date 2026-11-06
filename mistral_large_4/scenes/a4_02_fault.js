/* S_FAULT: whose fault is a broken episode (§4.3.2). A schematic of the routing
   the paper states: agent failures are counted as incorrect, infrastructure
   failures are discarded in RL and retried in evaluation, ambiguous cases default
   to the agent. The RL group strip uses the paper's default group size (§4.4.1,
   window.ML4.facts['rl.group_default']); one discarded rollout is an illustration. */
window.SCENES.S_FAULT = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var svg = K.board(host, { alt: 'Schematic: where each kind of broken episode goes, ' +
    'counted as incorrect or discarded in RL and retried in evaluation' });
  K.head(svg, 'Whose fault was it?', 'Section 4.3.2 · blame for broken episodes · schematic');

  var G = F['rl.group_default'];   // 16, the default RL group size (§4.4.1)

  // the paper's example failures, and where each goes: 0 = agent, 1 = infrastructure.
  // Every callout wraps to 3 lines at 72 cols, so the reserved band below fits all states.
  var FAIL = [
    { name: 'malformed tool call', to: 0,
      say: 'A malformed tool call is counted as incorrect. The agent is held at fault, ' +
        'so the broken episode counts against it instead of being dropped or retried.' },
    { name: 'context overflow', to: 0,
      say: 'Running out of context is counted as incorrect, the same as a malformed ' +
        'tool call. It is the agent\'s failure, not the infrastructure\'s, so the ' +
        'episode is scored instead of skipped.' },
    { name: 'sandbox crash', to: 1,
      say: 'A sandbox crash is discarded in RL and retried in evaluation. With the line ' +
        'kept strict, it costs throughput without corrupting the reward signal, but can ' +
        'shrink effective group sizes and skew the mixture.' },
    { name: 'service call error', to: 1,
      say: 'A failed call to a tool, verifier, judge or other service is discarded in RL ' +
        'and retried in evaluation. As with a sandbox crash, it costs throughput and can ' +
        'shrink effective groups and skew the mixture.' },
    { name: 'ambiguous case', to: 0,
      say: 'When it is unclear whether the agent or the infrastructure broke the episode, ' +
        'the case defaults to the agent\'s fault, so unreliable behavior is never ' +
        'silently dropped and implicitly rewarded.' }
  ];

  // context: the shared interface this blame rule lives in
  svg.appendChild(K.n('rect', { x: 0, y: 0, width: 640, height: 56, rx: 9,
    fill: K.TINT.ink, stroke: C.line }));
  K.mono(svg, 14, 22, 'RL, evaluation and synthetic data generation share one environment interface.',
    { size: 12.5, color: C.ink2 });
  K.mono(svg, 14, 42, 'Episodes are append-only records; only the policy\'s own outputs are trained on.',
    { size: 12.5, color: C.ink3 });

  K.label(svg, 0, 94, 'pick a failure');
  K.label(svg, 330, 94, 'where the environment sends it');

  // the two destinations
  var BX = 330, BW = 310, BH = 92, BY = [106, 222];
  var box = BY.map(function (y, k) {
    var r = K.n('rect', { x: BX, y: y, width: BW, height: BH, rx: 10, fill: '#fff',
      stroke: C.line, 'stroke-width': 1 });
    svg.appendChild(r);
    // red marks "the agent's fault" only next to its label; infrastructure is grey context
    svg.appendChild(K.n('rect', { x: BX + 16, y: y + 15, width: 10, height: 10, rx: 2,
      fill: k === 0 ? C.red : CH.GREY }));
    K.label(svg, BX + 34, y + 25, k === 0 ? 'agent\'s fault' : 'infrastructure fault',
      { color: C.ink2 });
    return r;
  });
  K.mono(svg, BX + 16, BY[0] + 52, 'counted as incorrect', { size: 15, weight: 600, color: C.ink });
  K.mono(svg, BX + 16, BY[0] + 75, 'also the default when unclear', { size: 13, color: C.ink3 });
  K.mono(svg, BX + 16, BY[1] + 52, 'discarded in RL', { size: 15, weight: 600, color: C.ink });
  K.mono(svg, BX + 16, BY[1] + 75, 'retried in evaluation', { size: 15, weight: 600, color: C.ink });

  // the route from the picked failure to its destination
  var route = K.n('g', {});
  svg.appendChild(route);
  var PY = 106, PH = 34, PG = 9;
  function drawRoute(i) {
    CH.clear(route);
    var y1 = PY + i * (PH + PG) + PH / 2, y2 = BY[FAIL[i].to] + BH / 2;
    K.arrow(route, 204, y1, BX - 4, y2, { curve: true, color: C.amber, w: 2.2, headSize: 8 });
    route.appendChild(K.n('circle', { cx: 204, cy: y1, r: 3.5, fill: C.amber }));
  }

  // one RL group of the paper's default size; the last rollout is the broken one
  K.label(svg, 0, 352, 'RL group: ' + G + ' rollouts of one prompt (default size, Section 4.4.1)');
  var PX0 = 10, PDX = 24, PCY = 380;
  var pips = [];
  for (var j = 0; j < G; j++) {
    var c = K.n('circle', { cx: PX0 + j * PDX, cy: PCY, r: 8, fill: CH.GREY2,
      stroke: '#fff', 'stroke-width': 1.6 });
    svg.appendChild(c);
    pips.push(c);
  }
  var bad = pips[G - 1];
  var badTag = K.mono(svg, PX0 + (G - 1) * PDX + 20, PCY + 4.5, '', { size: 13, weight: 600,
    color: C.ink });
  var out = CH.readout(svg, 0, 418, { size: 13.5, color: C.ink2 });

  // reserve the tallest callout, so the foot (placed once) never lands on it
  svg.appendChild(K.n('rect', { x: 0, y: 438, width: 640, height: 3 * 19 + 22, fill: 'none' }));
  var note = CH.body(svg);

  function draw(i) {
    var f = FAIL[i], agent = f.to === 0;
    drawRoute(i);
    box.forEach(function (r, k) {
      var on = k === f.to;
      r.setAttribute('stroke', on ? C.amber : C.line);
      r.setAttribute('stroke-width', on ? 2 : 1);
      r.setAttribute('fill', on ? K.TINT.amber : '#fff');
    });
    if (agent) {
      bad.setAttribute('fill', C.red);
      bad.setAttribute('stroke', '#fff');
      bad.removeAttribute('stroke-dasharray');
      badTag.textContent = 'counted as incorrect';
      out.set('It stays in its group of ' + G + ' and counts against the agent.');
    } else {
      bad.setAttribute('fill', '#fff');
      bad.setAttribute('stroke', CH.GREY);
      bad.setAttribute('stroke-dasharray', '3 2.5');
      badTag.textContent = 'discarded';
      // derived: one discarded rollout out of the default group size
      out.set('In this example, dropping it leaves ' + (G - 1) + ' of ' + G +
        ': a smaller effective group.');
    }
    CH.clear(note);
    K.callout(note, 0, 438, 640, f.say, { cols: 74 });
  }

  CH.pick(svg, 0, PY, FAIL.map(function (f) { return f.name; }), draw,
    { w: 196, h: PH, gap: PG, tint: 'amber', size: 13.5 }).pick(2);

  K.foot(svg, 'Schematic. The example failures and where each one goes are from Section 4.3.2. ' +
    'Group size is set per data source (' + F['rl.group_code'] + ' for agentic coding, ' +
    F['rl.group_min'] + ' to ' + F['rl.group_max'] + ' overall); the group of ' + G +
    ' is the default, and one broken rollout in it is an illustration. The paper tracks ' +
    'group-size and mixture effects separately (Section 4.4.3).');
};
