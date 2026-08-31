/* How it moves. The two-model stack, the rate ladder, and cross-robot transfer. */
window.SCENES = window.SCENES || {};

(function () {
  var K = window.KIT, C = K.C, T = K.TINT, K3 = window.K3;

  /* ---- 07 two models in series ---- */
  window.SCENES.S_STACK = function (root, api) {
    var f = api.RN.facts;
    var s = K.board(root, { alt: 'A large reasoning model in series with a small geometric one.' });
    K.head(s, 'Split the job, then size each half',
      'reasoning is expensive, geometry is not');

    var ROWS = [
      { c: C.blue, name: 'Robostral Navigate', sz: f.vlm_params_b + 'B', kind: 'vision-language model',
        job: 'read the instruction, understand the room, choose the next waypoint',
        why: 'needs reasoning, so it gets the parameters' },
      { c: C.teal, name: 'diffusion policy', sz: f.policy_params_m + 'M', kind: 'diffusion transformer',
        job: 'turn one waypoint into a dense, obstacle-free trajectory',
        why: 'geometry at this scale does not need a large model' },
      { c: C.ink3, name: 'motion controller', sz: 'varies', kind: 'platform specific',
        job: 'track that trajectory with actuator commands',
        why: 'the only piece that changes when you change robot' }
    ];

    ROWS.forEach(function (r, i) {
      var y = 12 + i * 146;
      K.panel(s, 0, y, 640, 118, { stroke: r.c === C.ink3 ? C.line : r.c,
        fill: r.c === C.ink3 ? '#fff' : (i ? T.teal : T.blue) });
      K.big(s, 20, y + 54, r.sz, { size: r.sz.length > 4 ? 26 : 34, color: r.c });
      K.mono(s, 20, y + 78, r.kind, { size: 11.5, color: C.ink3 });
      K.mono(s, 168, y + 32, r.name, { size: 15, color: r.c, weight: 500 });
      K.para(s, 168, y + 56, r.job, 54, { size: 12.5, color: C.ink2 });
      K.mono(s, 168, y + 100, r.why, { size: 11.5, color: C.ink3 });
      if (i < 2) K.arrow(s, 320, y + 118, 320, y + 158, { color: C.line, w: 1.6, headSize: 7 });
    });

    K.callout(s, 0, 456, 640,
      'Nothing expensive runs at the frequency the motors need. That is the whole reason for ' +
      'the split.', { cols: 66 });
  };

  /* ---- 08 the rate ladder, animated ---- */
  window.SCENES.S_RATES = function (root, api) {
    var f = api.RN.facts;
    var s = K.board(root, { alt: 'Three components running at half a hertz, ten hertz and a hundred hertz.' });
    K.head(s, 'Half a hertz of thinking',
      'each layer down is cheaper, faster and simpler');

    var LANES = [
      { hz: f.hz_vlm, c: C.blue, n: 'waypoint', who: 'vision-language model' },
      { hz: f.hz_policy, c: C.teal, n: 'trajectory', who: 'diffusion policy' },
      { hz: f.hz_motor, c: C.amber, n: 'motor command', who: 'controller' }
    ];
    var X0 = 130, XW = 490, ticks = [];

    LANES.forEach(function (L, i) {
      var y = 46 + i * 118;
      K.mono(s, 0, y - 12, L.who, { size: 12.5, color: L.c, weight: 500 });
      K.mono(s, 0, y + 16, L.hz + ' Hz', { size: 20, color: L.c, weight: 700 });
      K.mono(s, 0, y + 38, L.n, { size: 11, color: C.ink3 });
      s.appendChild(K.n('rect', { x: X0, y: y - 4, width: XW, height: 46, rx: 8,
        fill: i === 0 ? T.blue : (i === 1 ? T.teal : T.amber) }));
      // one second of wall clock, drawn to scale, capped so 100 Hz stays legible
      // 0.5 Hz is one tick every OTHER second, so draw two seconds of wall clock
      var SECS = L.hz < 1 ? 2 : 1;
      var n = Math.min(50, Math.max(1, Math.round(L.hz * SECS)));
      var g = K.n('g', {}); s.appendChild(g);
      for (var k = 0; k < n; k++) {
        var x = X0 + 12 + (XW - 24) * (n === 1 ? 0.5 : k / (n - (n > 1 ? 1 : 0)));
        g.appendChild(K.n('rect', { x: x - 1.4, y: y + 4, width: 2.8, height: 30, rx: 1.4,
          fill: L.c, opacity: .9 }));
      }
      if (L.hz > 50) K.mono(s, X0 + 12, y + 62, 'drawn at 50 of ' + L.hz + ' for legibility',
        { size: 10.5, color: C.ink3 });
      if (SECS === 2) K.mono(s, X0 + 12, y + 62, 'one tick, across two seconds',
        { size: 10.5, color: C.ink3 });
      ticks.push({ g: g, hz: L.hz, y: y, c: L.c });
    });

    K.mono(s, X0, 402, '|', { size: 11, color: C.ink3 });
    s.appendChild(K.n('line', { x1: X0, y1: 396, x2: X0 + XW, y2: 396, stroke: C.line }));
    K.mono(s, X0, 416, 'two seconds of wall clock', { size: 11.5, color: C.ink3 });
    var head = K.n('rect', { x: X0, y: 30, width: 2, height: 356, fill: C.red, opacity: .55 });
    s.appendChild(head);

    K.callout(s, 0, 438, 640,
      'The vision-language model emits one waypoint every two seconds. Everything between that ' +
      'and the actuators exists to fill in the gap.', { cols: 66 });

    return {
      tick: function (t) {
        var p = (t * 0.35) % 1;
        head.setAttribute('x', X0 + (XW) * p);
      }
    };
  };

  /* ---- 09 what goes into the model ---- */
  window.SCENES.S_TOKENS = function (root, api) {
    var s = K.board(root, { alt: 'Instruction tokens and a history of frames become one sequence.' });
    K.head(s, 'Grounding first', 'navigation as an extension of pointing at things');

    K.label(s, 0, 14, 'input sequence');
    K.panel(s, 0, 26, 640, 96);
    K.mono(s, 18, 52, '"go past the lockers, then right at the couch"', { size: 13, color: C.ink });
    K.label(s, 18, 72, 'tokenised instruction', { size: 9.5 });
    var fx = 18;
    for (var i = 0; i < 6; i++) {
      var last = i === 5;
      s.appendChild(K.n('rect', { x: fx, y: 84, width: 46, height: 26, rx: 4,
        fill: last ? T.blue : T.ink, stroke: last ? C.blue : C.line }));
      K.mono(s, fx + 8, 101, 'O' + i, { size: 11, color: last ? C.blue : C.ink3 });
      fx += 54;
    }
    K.mono(s, fx + 8, 101, 'frames so far', { size: 11.5, color: C.ink3 });

    K.arrow(s, 320, 128, 320, 158, { color: C.line, w: 1.6 });

    K.panel(s, 0, 164, 640, 92, { stroke: C.blue, fill: T.blue });
    K.mono(s, 18, 192, 'vision encoder → visual tokens → appended to the instruction',
      { size: 13, color: C.blue });
    K.para(s, 18, 214, 'The model is initialised from a dense vision-language model already ' +
      'trained for spatial grounding: pointing, counting, object localisation.', 62,
      { size: 12, color: C.ink2 });

    K.label(s, 0, 292, 'why that initialisation');
    K.panel(s, 0, 304, 640, 86, { flat: true, fill: T.teal, stroke: C.teal });
    K.para(s, 18, 330, 'Once a model can say where a thing is in an image, moving toward that ' +
      'thing is a new use of an existing skill rather than a new skill.', 62,
      { size: 13, color: C.teal });

    K.callout(s, 0, 408, 640,
      'The frame history is what lets the model know where it has already been, which is how it ' +
      'tracks progress through a multi-step instruction.', { cols: 66 });
    K.foot(s, 'The paper does not name the initialisation checkpoint, so neither does this figure.');
  };

  /* ---- 10 waypoint to motion, and the latency gap ---- */
  window.SCENES.S_CHUNK = function (root, api) {
    var f = api.RN.facts;
    var s = K.board(root, { alt: 'What the diffusion policy consumes and what it emits.' });
    K.head(s, 'Covering for the big model being slow',
      'two frames go in, because they are not the same frame');

    K.label(s, 0, 14, 'inputs to the ' + f.policy_params_m + 'M policy');
    var IN = [
      ['a', 'the waypoint, pointing or displacement', C.blue],
      ['h, r', 'robot height and radius', C.teal],
      ['o_vlm', 'the frame the big model actually looked at', C.purple],
      ['o_t', 'the frame from right now', C.red]
    ];
    IN.forEach(function (row, i) {
      var y = 28 + i * 52;
      K.panel(s, 0, y, 640, 42, { flat: true, fill: 'rgba(255,255,255,.9)', stroke: C.line });
      K.mono(s, 18, y + 27, row[0], { size: 14, color: row[2], weight: 700 });
      K.mono(s, 96, y + 27, row[1], { size: 12.5, color: C.ink2 });
    });

    K.label(s, 0, 254, 'why two frames');
    K.panel(s, 0, 266, 640, 96, { stroke: C.purple, fill: T.purple });
    s.appendChild(K.n('line', { x1: 24, y1: 316, x2: 610, y2: 316, stroke: C.line }));
    [['o_vlm', 60, C.purple], ['o_t', 470, C.red]].forEach(function (m) {
      s.appendChild(K.n('circle', { cx: m[1], cy: 316, r: 5.5, fill: m[2] }));
      K.mono(s, m[1] - 16, 302, m[0], { size: 12, color: m[2] });
    });
    K.arrow(s, 70, 336, 462, 336, { color: C.purple, w: 1.4, dash: '4 4' });
    K.mono(s, 200, 352, 'the robot kept moving during inference', { size: 12, color: C.purple });

    K.arrow(s, 320, 372, 320, 396, { color: C.line, w: 1.6 });
    K.panel(s, 0, 402, 640, 62, { stroke: C.teal, fill: T.teal });
    K.mono(s, 18, 428, f.chunk_steps + ' relative moves (dx, dy, dθ) covering the next second',
      { size: 13.5, color: C.teal });
    K.mono(s, 18, 450, 'expanded to ' + f.hz_motor + ' Hz by the platform controller',
      { size: 11.5, color: C.ink3 });

    K.foot(s, 'Section 2.3. Handing the policy both frames is how the system stays honest about ' +
      'its own inference latency.');
  };

  /* ---- 11 randomise the body, watch the pixel move ---- */
  window.SCENES.S_RANDOM = function (root, api) {
    var f = api.RN.facts;
    K3.head(root, 'Never let it learn one body',
      'the world point is fixed. change the body and the pixel moves.');

    var read = K3.el('div', 'k3read');
    var uvB = K3.el('b', null, 'u, v'), hB = K3.el('b', 'off', 'camera');
    read.appendChild(uvB); read.appendChild(hB);

    var st = { h: 1.1, r: 0.28, camFrac: 0.9, pitch: 10 };

    var handle = K3.mount(root, { aspect: '16 / 9' }, function (ctx) {
      var THREE = ctx.THREE;
      var w = ctx.world(THREE, 21);
      // this figure is about the body, not the building: clear the partitions so
      // the camera always has the same unobstructed view of the same waypoint
      w.openUp();
      ctx.scene.add(w.group);

      var holder = new THREE.Group();
      ctx.scene.add(holder);
      var rob = null;
      function rebuild() {
        if (rob) holder.remove(rob);
        rob = ctx.robot(THREE, { height: st.h, radius: st.r, camFrac: st.camFrac,
          color: ctx.C.teal });
        rob.position.set(-2.0, 0, 6.0);
        rob.rotation.y = Math.atan2(GOAL.x - (-2.0), GOAL.z - 6.0);
        holder.add(rob);
      }
      var GOAL = new THREE.Vector3(11.2, 0, 0.0);
      rebuild();
      var wp = ctx.waypoint(THREE, ctx.C.red);
      wp.position.copy(GOAL);
      ctx.scene.add(wp);

      var camA = ctx.thirdPerson(THREE, 44, [1, 2]);
      var camB = new THREE.PerspectiveCamera(58, 1, 0.05, 200);

      return {
        camera: camA,
        frame: function (t, c) {
          if (Math.abs(rob.userData.h - st.h) > 1e-6 ||
              Math.abs(rob.userData.r - st.r) > 1e-6 ||
              Math.abs(rob.userData.camY - st.h * st.camFrac) > 1e-6) rebuild();
          ctx.camPose(THREE, rob, st.pitch, camB);
          var pr = ctx.project(THREE, new THREE.Vector3(GOAL.x, 0.35, GOAL.z), camB);

          camA.position.set(-11, 7.5, 11);
          camA.lookAt(2.0, 0.6, 0.0);
          camA.updateMatrixWorld();
          wp.userData.pulse.scale.setScalar(1 + Math.sin(t * 3) * 0.1);

          var vp = window.K3.split(c, camA, camB, 10);
          var ov = c.overlay; ov.innerHTML = '';
          window.K3.vlabel(ov, 12, 22, 'the room');
          window.K3.vlabel(ov, vp.rightX + 12, 22, 'what the camera sees');
          if (pr.visible) {
            var px = vp.rightX + pr.u * vp.rightW, py = pr.v * c.h;
            window.K3.crossGuides(ov, px, py, vp.rightX, vp.rightX + vp.rightW, c.h);
            window.K3.reticle(ov, px, py, { r: 14,
              label: 'u ' + pr.u.toFixed(2) + '   v ' + pr.v.toFixed(2),
              maxX: vp.rightX + vp.rightW - 6 });
          } else {
            window.K3.banner(ov, vp.rightX + 14, c.h - 54, vp.rightW - 28,
              'this body cannot see the waypoint at all');
          }

          uvB.textContent = pr.visible
            ? 'u ' + pr.u.toFixed(3) + '   v ' + pr.v.toFixed(3) : 'out of frame';
          uvB.className = pr.visible ? '' : 'hot';
          hB.textContent = 'camera at ' + (st.h * st.camFrac).toFixed(2) + ' m, pitch ' +
            st.pitch.toFixed(0) + '°';
        }
      };
    });

    root.appendChild(read);
    var row = K3.ctl(root);
    K3.slider(row, 'height', f.height_min, f.height_max, 0.05, st.h,
      function (v) { return v.toFixed(2) + ' m'; }, function (v) { st.h = v; });
    K3.slider(row, 'radius', f.radius_min, f.radius_max, 0.01, st.r,
      function (v) { return v.toFixed(2) + ' m'; }, function (v) { st.r = v; });
    var row2 = K3.ctl(root);
    K3.slider(row2, 'camera height', f.cam_h_min, f.cam_h_max, 1, 90,
      function (v) { return v.toFixed(0) + '%'; }, function (v) { st.camFrac = v / 100; });
    K3.slider(row2, 'pitch', f.pitch_min, f.pitch_max, 1, st.pitch,
      function (v) { return v.toFixed(0) + '°'; }, function (v) { st.pitch = v; });

    K3.note(root, 'These are the exact ranges the paper randomises over per trajectory. ' +
      'The policy never sees a consistent viewpoint, which is how the paper reduces its ' +
      'dependence on any particular camera setup, scale or morphology.', 'teal');
    return handle;
  };

  /* ---- 12 two bodies, one set of weights ---- */
  window.SCENES.S_ROBOTS = function (root, api) {
    K3.head(root, 'Two machines that share nothing',
      'same vision-language model, same diffusion policy, different controller');

    var handle = K3.mount(root, { aspect: '16 / 9' }, function (ctx) {
      var THREE = ctx.THREE;
      ctx.scene.add(ctx.world(THREE, 33).group);

      // a tall bimanual-style base and a small wheeled one, at the ends of the
      // randomisation range rather than as portraits of specific products
      var tall = ctx.robot(THREE, { height: 1.7, radius: 0.34, camFrac: 0.93, color: ctx.C.blue, layer: 1 });
      tall.position.set(-7.2, 0, -0.9);
      ctx.scene.add(tall);
      var small = ctx.robot(THREE, { height: 0.5, radius: 0.19, camFrac: 0.86, color: ctx.C.amber, layer: 2 });
      small.position.set(-2.4, 0, 0.7);
      ctx.scene.add(small);

      var GOAL = new THREE.Vector3(11.2, 0, 0.0);
      var wp = ctx.waypoint(THREE, ctx.C.red);
      wp.position.copy(GOAL);
      ctx.scene.add(wp);
      tall.rotation.y = Math.atan2(GOAL.x + 7.2, GOAL.z + 0.9);
      small.rotation.y = Math.atan2(GOAL.x + 2.4, GOAL.z - 0.7);
      ctx.scene.add(ctx.route(THREE, [
        new THREE.Vector3(-7.2, .05, -0.9), new THREE.Vector3(0.4, .05, 0.0),
        new THREE.Vector3(5.0, .05, 0.2), GOAL.clone().setY(.05)], ctx.C.blue));

      var camA = ctx.thirdPerson(THREE, 40, [1, 2]);
      var camB = new THREE.PerspectiveCamera(58, 1, 0.05, 200);
      var which = 0;
      root.__pick = function (i) { which = i; };

      return {
        camera: camA,
        frame: function (t, c) {
          var rob = which ? small : tall;
          // each robot may see the other, never itself
          camB.layers.set(0);
          camB.layers.enable(which ? 1 : 2);
          ctx.camPose(THREE, rob, 10, camB);
          var pr = ctx.project(THREE, new THREE.Vector3(GOAL.x, 0.35, GOAL.z), camB);
          var a = 0.5 + Math.sin(t * 0.07) * 0.2;
          camA.position.set(Math.sin(a) * 13 - 5, 7.5, Math.cos(a) * 13 + 6);
          camA.lookAt(-3.0, 0.7, 0.0);
          camA.updateMatrixWorld();
          wp.userData.pulse.scale.setScalar(1 + Math.sin(t * 3) * 0.1);

          var vp = window.K3.split(c, camA, camB, 10);
          var ov = c.overlay; ov.innerHTML = '';
          window.K3.vlabel(ov, 12, 22, 'the room');
          window.K3.vlabel(ov, vp.rightX + 12, 22,
            (which ? 'the short robot' : 'the tall robot') + ' sees');
          if (pr.visible) {
            window.K3.reticle(ov, vp.rightX + pr.u * vp.rightW, pr.v * c.h, {
              label: '(' + pr.u.toFixed(3) + ', ' + pr.v.toFixed(3) + ')',
              maxX: vp.rightX + vp.rightW - 6 });
          }
        }
      };
    });

    var row = K3.ctl(root);
    K3.seg(row, ['look through the tall one', 'look through the short one'],
      function (i) { if (root.__pick) root.__pick(i); }, 0);
    K3.note(root, 'The paper deploys on a Galaxea R1 and a Hiwonder JetAuto with identical ' +
      'weights, changing only the low-level controller. Both are wheeled, so the legged and ' +
      'aerial part of the claim is an argument rather than a demonstration.');
    return handle;
  };
})();
