/* The problem. Why sensors are the bottleneck, and what pointing replaces. */
window.SCENES = window.SCENES || {};

(function () {
  var K = window.KIT, C = K.C, T = K.TINT, K3 = window.K3;

  /* ---- 01 the sensor tax ---- */
  window.SCENES.S_SENSORS = function (root, api) {
    var RN = api.RN, s = K.board(root, { alt: 'Sensing requirements and what each one costs.' });
    K.head(s, 'Every sensor is a robot you lose', 'add a requirement, read what it costs you');

    var TIERS = [
      { k: 'rgb', name: 'one RGB camera', chips: ['rgb'],
        cost: ['nothing to calibrate', 'runs on anything with a webcam'] },
      { k: 'depth', name: 'plus depth', chips: ['rgb', 'depth'],
        cost: ['needs a depth sensor on every unit', 'fails on glass, dark and sunlight'] },
      { k: 'rig', name: 'plus a camera rig', chips: ['rgb', 'depth', 'cam x4'],
        cost: ['fixed extrinsics per platform', 'recalibrate whenever the body changes'] },
      { k: 'map', name: 'plus a prebuilt map', chips: ['rgb', 'depth', 'cam x4', 'map'],
        cost: ['survey every building first', 'stale the moment furniture moves'] }
    ];

    K.label(s, 0, 12, 'what the system is allowed to assume');
    var body = K.n('g', {}); s.appendChild(body);
    var n_single = (RN.table1 || []).filter(function (r) { return r.group === 'single'; }).length;
    var n_depth = (RN.table1 || []).filter(function (r) { return r.group === 'depth'; }).length;

    K.switcher(s, 0, 24, ['rgb', '+depth', '+rig', '+map'], function (i) { draw(i); },
      { tint: 'amber' });

    function draw(i) {
      body.innerHTML = '';
      var t = TIERS[i];
      K.panel(body, 0, 74, 640, 138);
      var cx = 20;
      t.chips.forEach(function (c, j) {
        var on = j === t.chips.length - 1;
        var g = K.chip(body, cx, 100, c, {
          size: 12, h: 27,
          fill: on ? T.amber : T.ink,
          stroke: on ? C.amber : C.line,
          color: on ? C.amber : C.ink3
        });
        cx += g._w + 8;
        if (j < t.chips.length - 1) {
          K.mono(body, cx - 3, 118, '+', { size: 13, color: C.ink3 });
          cx += 12;
        }
      });
      K.mono(body, 20, 168, t.name, { size: 14, color: C.ink, weight: 500 });
      K.mono(body, 20, 190, 'sensing budget: ' + t.chips.length +
        (t.chips.length === 1 ? ' modality' : ' modalities'), { size: 12, color: C.ink3 });

      K.label(body, 0, 244, 'what it costs you');
      t.cost.forEach(function (line, j) {
        K.panel(body, 0, 258 + j * 56, 640, 46, { flat: true, fill: i === 0 ? T.teal : T.red,
          stroke: i === 0 ? C.teal : C.red });
        K.mono(body, 18, 286 + j * 56, line, { size: 13, color: i === 0 ? C.teal : C.red });
      });

      K.callout(body, 0, 384, 640,
        'Of the ' + (n_single + n_depth) + ' published systems the paper compares against, ' +
        n_single + ' work from a single RGB camera and ' + n_depth +
        ' require depth, LiDAR or several cameras.',
        { cols: 66 });
    }
    draw(0);
    K.foot(s, 'The three costs are the ones named in the introduction: fewer compatible robots, ' +
      'higher cost per unit, and environment-specific calibration.');
  };

  /* ---- 02 the three constraints ---- */
  window.SCENES.S_RECIPE = function (root, api) {
    var s = K.board(root, { alt: 'The three constraints the paper optimises for.' });
    K.head(s, 'Optimise the recipe', 'three constraints, and every later decision traces to one');

    var ITEMS = [
      { c: C.blue, k: 'minimise sensing', t: 'one monocular RGB stream, the sensor every platform already has',
        to: 'points at a pixel instead of measuring a distance' },
      { c: C.teal, k: 'generalise across bodies', t: 'no dependence on a particular height, lens or base',
        to: 'image-space output plus randomised bodies in training' },
      { c: C.amber, k: 'train efficiently', t: 'iteration speed is a first-class design constraint',
        to: 'prefix-tree packing, then RL with the compute it saved' }
    ];

    ITEMS.forEach(function (it, i) {
      var y = 20 + i * 132;
      K.panel(s, 0, y, 640, 112, { stroke: it.c, fill: 'rgba(255,255,255,.86)' });
      K.n('rect', {});
      s.appendChild(K.n('rect', { x: 0, y: y, width: 5, height: 112, rx: 2.5, fill: it.c }));
      K.big(s, 22, y + 52, String(i + 1), { size: 40, color: it.c });
      K.mono(s, 64, y + 34, it.k, { size: 14, color: it.c, weight: 500 });
      K.para(s, 64, y + 58, it.t, 62, { size: 12.5, color: C.ink2 });
      K.mono(s, 64, y + 94, '→ ' + it.to, { size: 12, color: C.ink3 });
    });

    K.callout(s, 0, 424, 640,
      'The paper is explicit that it is chasing a scalable recipe rather than a leaderboard ' +
      'position. The odd-looking choices later are all downstream of that.', { cols: 66 });
  };

  /* ---- 03 a metric waypoint belongs to one body ---- */
  window.SCENES.S_METRIC = function (root, api) {
    var RN = api.RN, f = RN.facts, s = K.board(root, { alt: 'The same command on two different bodies.' });
    K.head(s, 'A distance in metres belongs to one robot',
      'same command, two bodies, two different meanings');

    K.label(s, 0, 14, 'the command');
    K.code(s, 0, 26, 300, [
      { t: 'forward  Δx = 1.4 m', c: '#9fd0e6' },
      { t: 'lateral  Δy = 0.0 m', c: '#9fd0e6' },
      { t: 'turn     Δθ = 22°', c: '#9fd0e6' }
    ]);

    var BODIES = [
      { n: 'tall, narrow lens', h: f.height_max, r: f.radius_min, pitch: f.pitch_min, c: C.blue },
      { n: 'short, wide lens', h: f.height_min, r: f.radius_max, pitch: f.pitch_max, c: C.purple }
    ];

    BODIES.forEach(function (b, i) {
      var x = i * 330, y = 150;
      K.panel(s, x, y, 310, 216, { stroke: b.c, fill: i ? T.purple : T.blue });
      K.label(s, x + 18, y + 28, b.n, { color: b.c });
      // a little elevation drawing of the body, to scale against 1.8 m
      var gx = x + 42, gy = y + 190, scale = 74 / f.height_max;
      s.appendChild(K.n('line', { x1: x + 18, y1: gy, x2: x + 290, y2: gy,
        stroke: C.line, 'stroke-width': 1 }));
      var hh = b.h * scale;
      s.appendChild(K.n('rect', { x: gx - b.r * scale * 26, y: gy - hh,
        width: b.r * scale * 52, height: hh, rx: 3, fill: b.c, opacity: .82 }));
      s.appendChild(K.n('circle', { cx: gx, cy: gy - hh - 5, r: 4.5, fill: C.ink }));
      K.mono(s, x + 18, y + 76, b.h.toFixed(1) + ' m tall', { size: 12, color: b.c });
      K.mono(s, x + 18, y + 94, 'radius ' + b.r.toFixed(2) + ' m', { size: 12, color: b.c });
      K.mono(s, x + 18, y + 112, 'pitch ' + b.pitch + '°', { size: 12, color: b.c });
      K.mono(s, x + 108, y + 208, i ? 'clips the doorway frame' : 'clears it', { size: 12.5, color: b.c });
    });

    K.callout(s, 0, 392, 640,
      'The number did not change. The geometry it was measured against did. That is the ' +
      'coupling image-space output is there to break.', { cols: 66 });
    K.foot(s, 'Body ranges are the ones the paper randomises over in section 2.4. The two ' +
      'silhouettes are the ends of those ranges, not measured platforms.');
  };

  /* ---- 04 THE HERO: pointing, derived by projection ---- */
  window.SCENES.S_POINT = function (root, api) {
    K3.head(root, 'Point at the picture',
      'left, the room. right, the camera. the crosshair is projected, not drawn on.');

    var read = K3.el('div', 'k3read');
    var uvB = K3.el('b', null, 'u, v'), thB = K3.el('b', null, 'Δθ'),
        dB = K3.el('b', 'off', 'range');
    [uvB, thB, dB].forEach(function (b) { read.appendChild(b); });

    var state = { p: 0.12 };

    var h = K3.mount(root, { aspect: '16 / 9' }, function (ctx) {
      var THREE = ctx.THREE;
      var w = ctx.world(THREE, 11);
      ctx.scene.add(w.group);

      // straight down the corridor, so the onboard view has depth and a
      // landmark at the end rather than a wall 40 cm from the lens
      var PATH = [
        new THREE.Vector3(-11.2, 0.05, 1.4), new THREE.Vector3(-6.4, 0.05, 0.6),
        new THREE.Vector3(-1.4, 0.05, -0.4), new THREE.Vector3(3.6, 0.05, 0.2),
        new THREE.Vector3(7.8, 0.05, 0.9), new THREE.Vector3(10.0, 0.05, 1.1)
      ];
      var line = ctx.route(THREE, PATH, ctx.C.blue);
      ctx.scene.add(line);
      var curve = line.userData.curve;

      var rob = ctx.robot(THREE, { height: 1.15, radius: 0.3, camFrac: 0.9 });
      ctx.scene.add(rob);
      var wp = ctx.waypoint(THREE, ctx.C.red);
      ctx.scene.add(wp);

      var camA = ctx.thirdPerson(THREE, 42, [1, 2]);
      var camB = new THREE.PerspectiveCamera(58, 1, 0.05, 200);

      var target = new THREE.Vector3();
      var pos = new THREE.Vector3(), ahead = new THREE.Vector3();

      function place(p) {
        curve.getPointAt(Math.min(0.995, p), pos);
        curve.getPointAt(Math.min(0.999, p + 0.02), ahead);
        rob.position.set(pos.x, 0, pos.z);
        rob.rotation.y = Math.atan2(ahead.x - pos.x, ahead.z - pos.z);
        // the paper's rule: the furthest point along the route still in view
        var best = null;
        for (var q = 0.99; q > p; q -= 0.012) {
          curve.getPointAt(q, target);
          ctx.camPose(THREE, rob, 8, camB);
          var pr = ctx.project(THREE, new THREE.Vector3(target.x, 0.35, target.z), camB);
          if (pr.visible) { best = { q: q, x: target.x, z: target.z, pr: pr }; break; }
        }
        if (!best) {
          curve.getPointAt(Math.min(0.999, p + 0.05), target);
          ctx.camPose(THREE, rob, 8, camB);
          best = { q: p + 0.05, x: target.x, z: target.z,
                   pr: ctx.project(THREE, new THREE.Vector3(target.x, 0.35, target.z), camB) };
        }
        wp.position.set(best.x, 0, best.z);
        return best;
      }

      return {
        camera: camA,
        frame: function (t, c) {
          var best = place(state.p);
          // slow orbit so the room reads as a room
          var a = 0.55 + Math.sin(t * 0.06) * 0.22;
          camA.position.set(Math.sin(a) * 19, 15.5, Math.cos(a) * 19);
          camA.lookAt(0, 0, 0);
          camA.updateMatrixWorld();
          ctx.camPose(THREE, rob, 8, camB);
          wp.userData.pulse.scale.setScalar(1 + Math.sin(t * 3) * 0.11);

          var vp = window.K3.split(c, camA, camB, 10);

          // the reticle, positioned from the projection rather than by hand
          var ov = c.overlay; ov.innerHTML = '';
          var pr = best.pr;
          window.K3.vlabel(ov, 12, 22, 'the room');
          window.K3.vlabel(ov, vp.rightX + 12, 22, 'what the camera sees');
          if (pr.visible) {
            window.K3.reticle(ov, vp.rightX + pr.u * vp.rightW, pr.v * c.h, {
              label: '(' + pr.u.toFixed(3) + ', ' + pr.v.toFixed(3) + ')',
              maxX: vp.rightX + vp.rightW - 6 });
          }

          uvB.textContent = 'u ' + pr.u.toFixed(3) + '   v ' + pr.v.toFixed(3);
          uvB.className = pr.visible ? '' : 'off';
          var yaw = Math.atan2(best.x - rob.position.x, best.z - rob.position.z) - rob.rotation.y;
          while (yaw > Math.PI) yaw -= 2 * Math.PI;
          while (yaw < -Math.PI) yaw += 2 * Math.PI;
          thB.textContent = 'Δθ ' + (yaw * 180 / Math.PI).toFixed(0) + '°';
          dB.textContent = pr.depth.toFixed(1) + ' m away';
        }
      };
    });

    root.appendChild(read);
    var row = K3.ctl(root);
    K3.slider(row, 'along the route', 0, 0.94, 0.005, state.p,
      function (v) { return (v * 100).toFixed(0) + '%'; },
      function (v) { state.p = v; });
    K3.note(root, 'The image coordinate is computed, not placed. It is the world waypoint ' +
      'projected through the camera on the right.', 'blue');
    return h;
  };

  /* ---- 05 the five quantities ---- */
  window.SCENES.S_UV = function (root, api) {
    var RN = api.RN, f = RN.facts;
    var s = K.board(root, { alt: 'The five quantities predicted when the goal is visible.' });
    K.head(s, 'Five numbers, most of the time', 'a_vis when the destination is in frame');

    K.label(s, 0, 14, 'predicted jointly');
    var FIELDS = [
      { k: 'u', d: 'image column of the waypoint', c: C.red, kind: 'point' },
      { k: 'v', d: 'image row of the waypoint', c: C.red, kind: 'point' },
      { k: 'Δx', d: 'forward displacement, metres', c: C.ink3, kind: 'metric' },
      { k: 'Δy', d: 'lateral displacement, metres', c: C.ink3, kind: 'metric' },
      { k: 'Δθ', d: 'heading change on arrival', c: C.blue, kind: 'both' }
    ];
    FIELDS.forEach(function (fl, i) {
      var y = 30 + i * 62;
      K.panel(s, 0, y, 640, 50, { flat: i > 1 && i < 4,
        fill: fl.kind === 'point' ? T.red : (fl.kind === 'both' ? T.blue : '#fff'),
        stroke: fl.kind === 'metric' ? C.line : fl.c });
      K.mono(s, 20, y + 32, fl.k, { size: 20, color: fl.c, weight: 700 });
      K.mono(s, 74, y + 31, fl.d, { size: 13, color: C.ink2 });
      K.chip(s, 470, y + 13, fl.kind === 'point' ? 'pointing' :
        (fl.kind === 'both' ? 'both modes' : 'fallback'), {
        size: 10.5, h: 24, color: fl.c, stroke: fl.c === C.ink3 ? C.line : fl.c });
    });

    K.callout(s, 0, 356, 640,
      'Pointing is the preferred mode. The metric pair is trained on every step as a co-training ' +
      'task, which is affordable precisely because only about ' + f.invisible_pct +
      '% of the data actually needs it.', { cols: 66, color: C.blue, tint: T.blue });
    K.foot(s, 'Equation 1 in section 2.2. The model also emits STOP when the instruction is done.');
  };

  /* ---- 06 the fallback, shown by turning away ---- */
  window.SCENES.S_FALLBACK = function (root, api) {
    var f = api.RN.facts;
    K3.head(root, 'You cannot point at what is behind you',
      'turn past the edge of the frame and the output changes shape');

    var read = K3.el('div', 'k3read');
    var mode = K3.el('b', null, 'pointing'), out = K3.el('b', 'off', 'a_vis');
    read.appendChild(mode); read.appendChild(out);

    var state = { yaw: 0 };

    var h = K3.mount(root, { aspect: '16 / 9' }, function (ctx) {
      var THREE = ctx.THREE;
      var w = ctx.world(THREE, 4);
      ctx.scene.add(w.group);
      var rob = ctx.robot(THREE, { height: 1.15, radius: 0.3, color: ctx.C.purple });
      rob.position.set(-5.2, 0, 0.6);
      ctx.scene.add(rob);
      var GOAL = new THREE.Vector3(10.2, 0, 1.1);
      var wp = ctx.waypoint(THREE, ctx.C.red);
      wp.position.copy(GOAL);
      ctx.scene.add(wp);

      var camA = ctx.thirdPerson(THREE, 44, [1, 2]);
      var camB = new THREE.PerspectiveCamera(58, 1, 0.05, 200);
      var base = Math.atan2(GOAL.x - rob.position.x, GOAL.z - rob.position.z);

      return {
        camera: camA,
        frame: function (t, c) {
          rob.rotation.y = base + state.yaw * Math.PI / 180;
          ctx.camPose(THREE, rob, 6, camB);
          var pr = ctx.project(THREE, new THREE.Vector3(GOAL.x, 0.35, GOAL.z), camB);

          camA.position.set(-4, 15, 17);
          camA.lookAt(2, 0, 2);
          camA.updateMatrixWorld();
          wp.userData.pulse.scale.setScalar(1 + Math.sin(t * 3) * 0.11);

          var vp = window.K3.split(c, camA, camB, 10);
          var ov = c.overlay; ov.innerHTML = '';
          window.K3.vlabel(ov, 12, 22, 'the room');
          window.K3.vlabel(ov, vp.rightX + 12, 22, 'what the camera sees');
          if (pr.visible) {
            window.K3.reticle(ov, vp.rightX + pr.u * vp.rightW, pr.v * c.h,
              { label: 'a_vis  (u, v, Δx, Δy, Δθ)', maxX: vp.rightX + vp.rightW - 6 });
          } else {
            window.K3.banner(ov, vp.rightX + 14, c.h - 54, vp.rightW - 28,
              'no pixel in this frame refers to the goal');
          }

          mode.textContent = pr.visible ? 'pointing mode' : 'displacement fallback';
          mode.className = pr.visible ? 'ok' : 'hot';
          out.textContent = pr.visible
            ? 'a_vis = (u, v, Δx, Δy, Δθ)'
            : 'a_invis = (Δx, Δy, Δθ)';
          out.className = pr.visible ? 'off' : 'hot';
        }
      };
    });

    root.appendChild(read);
    var row = K3.ctl(root);
    K3.slider(row, 'turn the robot', -180, 180, 1, 0,
      function (v) { return v.toFixed(0) + '°'; },
      function (v) { state.yaw = v; });
    K3.note(root, 'About ' + f.invisible_pct + '% of the training data has no visible ' +
      'destination. The fallback is what guarantees the robot can always make progress.', 'red');
    return h;
  };
})();
