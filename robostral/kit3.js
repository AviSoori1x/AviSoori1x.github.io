/* The 3D half of the drawing kit.

   Robostral Navigate points at a pixel. Any figure that merely draws a dot on a
   photograph is asserting that idea rather than showing it. So the guide builds
   a small indoor world, puts a camera on a robot inside it, and derives the
   image coordinate by actually projecting the world waypoint through that
   camera. Move the camera and the pixel moves, because it is the same maths the
   model is trained against.

   One WebGL context is shared by every 3D figure. The scroll engine only ticks
   the scene you are looking at, so the canvas is simply re-parented on activate. */
(function () {
  var C = {
    bg: 0xf4f1ea, paper: 0xece7dc, ink: 0x141922, ink3: 0x5d6474,
    blue: 0x1c6e8c, teal: 0x0f6e56, amber: 0xb4551a,
    purple: 0x6b3fa8, red: 0xa32d2d, wall: 0xe4ded1, floor: 0xdcd5c6
  };

  /* ---------- module loading ---------- */
  var waiting = [];
  function ready(cb) {
    if (window.THREE) cb(window.THREE);
    else waiting.push(cb);
  }
  document.addEventListener('three:ready', function () {
    var q = waiting.splice(0);
    q.forEach(function (f) { try { f(window.THREE); } catch (e) { console.error(e); } });
  });

  /* ---------- one renderer for the whole page ---------- */
  var R = null;
  function renderer(THREE) {
    if (R) return R;
    R = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' });
    R.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    R.setClearColor(C.bg, 1);
    R.outputColorSpace = THREE.SRGBColorSpace;
    R.domElement.style.display = 'block';
    R.domElement.style.width = '100%';
    R.domElement.style.borderRadius = '12px';
    return R;
  }

  /* deterministic noise, so a "randomly generated" scene is the same every visit */
  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  /* ---------- materials ---------- */
  function mats(THREE) {
    function lam(c, o) {
      o = o || {};
      return new THREE.MeshLambertMaterial({
        color: c, transparent: o.op != null, opacity: o.op == null ? 1 : o.op,
        side: o.side || THREE.FrontSide
      });
    }
    return {
      floor: lam(C.floor), wall: lam(C.wall), ink: lam(C.ink),
      blue: lam(C.blue), teal: lam(C.teal), amber: lam(C.amber),
      purple: lam(C.purple), red: lam(C.red), paper: lam(C.paper),
      lam: lam
    };
  }

  /* ---------- the indoor world ----------
     A plan of rooms off a central corridor, with doorway gaps, furniture in the
     page palette, and a route through it. Generated from a seed so the figure is
     reproducible but nothing is hand-placed. */
  function world(THREE, seed) {
    var M = mats(THREE), rand = rng(seed || 7);
    var g = new THREE.Group();
    var W = 26, D = 18, WALL_H = 2.6, T = 0.16;

    var floor = new THREE.Mesh(new THREE.BoxGeometry(W, 0.12, D), M.floor);
    floor.position.y = -0.06;
    g.add(floor);

    // faint tile lines, which is what stops a big flat plane reading as a void
    var grid = new THREE.GridHelper(W, W, 0xb9b0a0, 0xc9c1b2);
    grid.position.y = 0.008;
    grid.material.transparent = true;
    grid.material.opacity = 0.5;
    g.add(grid);

    var walls = [];
    function wall(x, z, w, d) {
      var m = new THREE.Mesh(new THREE.BoxGeometry(w, WALL_H, d), M.wall);
      m.position.set(x, WALL_H / 2, z);
      g.add(m);
      walls.push(m);
      // a darker cap line so walls read as built rather than as fog
      var cap = new THREE.Mesh(new THREE.BoxGeometry(w + 0.02, 0.05, d + 0.02), M.lam(0xc2b9a8));
      cap.position.set(0, WALL_H / 2, 0);
      m.add(cap);
      return m;
    }

    // perimeter, left open at the far end so the corridor reads as continuing
    wall(0, -D / 2, W, T);
    wall(-W / 2, 0, T, D);
    wall(W / 2, 0, T, D);
    wall(-W / 4 - 2, D / 2, W / 2 - 4, T);
    wall(W / 4 + 2, D / 2, W / 2 - 4, T);

    // two interior partitions forming a corridor down the middle, with doorways.
    // Collected separately so a figure that needs clean sightlines can hide them.
    var partitions = [];
    [-3.2, 3.2].forEach(function (z, i) {
      var sgn = i ? 1 : -1;
      partitions.push(wall(-8.5, z, 9, T));
      partitions.push(wall(4.0, z, 8, T));
      // a stub that makes the doorway read as a doorway
      partitions.push(wall(-3.4, z + sgn * 1.1, T, 2.2));
    });

    // furniture, in the accent palette so the 3D matches the 2D figures
    function box(x, z, w, h, d, m, rot) {
      var e = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
      e.position.set(x, h / 2, z);
      if (rot) e.rotation.y = rot;
      g.add(e);
      return e;
    }
    var props = [];
    // desks along the top rooms
    for (var i = 0; i < 5; i++) {
      props.push(box(-10 + i * 3.4, -6.2 - rand() * 0.6, 2.0, 0.74, 1.1, M.lam(0xb9ac93)));
    }
    // lockers
    props.push(box(-11.4, -1.0, 0.9, 2.0, 4.0, M.lam(0x6f7787)));
    // the orange couch, the landmark in every one of these episodes
    props.push(box(11.2, 0.0, 1.1, 0.82, 2.6, M.amber));
    // planters
    [[-6.4, 6.1], [1.2, 6.3], [11.0, -5.4]].forEach(function (p) {
      props.push(box(p[0], p[1], 0.7, 0.9, 0.7, M.teal));
    });
    [[-8.2, -1.5], [-3.0, 1.6], [2.6, -1.6], [7.4, 1.5]].forEach(function (p) {
      props.push(box(p[0], p[1], 0.6, 0.62, 0.6, M.teal));
    });
    // A colonnade down BOTH sides of the corridor. These used to sit on the
    // centreline, which put a grey cylinder in the middle of every onboard view.
    for (var k = -2; k <= 2; k++) {
      [-2.3, 2.3].forEach(function (cz) {
        var cm = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.26, WALL_H, 14),
          M.lam(0xd2caba));
        cm.position.set(k * 5.2, WALL_H / 2, cz);
        g.add(cm);
      });
    }

    return { group: g, W: W, D: D, mats: M, props: props, walls: walls,
             partitions: partitions,
             openUp: function () { partitions.forEach(function (m) { m.visible = false; }); } };
  }

  /* ---------- a robot, parameterised the way the paper randomises it ---------- */
  function robot(THREE, opts) {
    opts = opts || {};
    var M = mats(THREE);
    var h = opts.height == null ? 1.1 : opts.height;
    var r = opts.radius == null ? 0.28 : opts.radius;
    var camFrac = opts.camFrac == null ? 0.9 : opts.camFrac;
    var col = opts.color == null ? C.blue : opts.color;

    var g = new THREE.Group();
    var base = new THREE.Mesh(new THREE.CylinderGeometry(r, r * 1.04, h * 0.16, 24), M.lam(0x3b4250));
    base.position.y = h * 0.08;
    g.add(base);
    var mast = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.42, r * 0.5, h * 0.78, 16), M.lam(col));
    mast.position.y = h * 0.16 + h * 0.39;
    g.add(mast);
    var head = new THREE.Mesh(new THREE.BoxGeometry(r * 1.5, h * 0.13, r * 1.0), M.lam(0x2a303c));
    head.position.y = h * camFrac;
    g.add(head);
    // the lens, so it is obvious which way the thing is looking
    var lens = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.035, h * 0.035, 0.05, 14), M.lam(0x11151c));
    lens.rotation.x = Math.PI / 2;
    lens.position.set(0, h * camFrac, r * 0.52);
    g.add(lens);
    // fake contact shadow, far cheaper than a shadow map and reads better on paper
    var sh = new THREE.Mesh(new THREE.CircleGeometry(r * 1.7, 24),
      new THREE.MeshBasicMaterial({ color: 0x8d8676, transparent: true, opacity: 0.26 }));
    sh.rotation.x = -Math.PI / 2;
    sh.position.y = 0.012;
    g.add(sh);

    // A real onboard camera cannot see its own chassis, and ours was rendering
    // the inside of the robot's head. Park the body on its own layer; the
    // third-person camera opts in, the onboard camera never does.
    var layer = opts.layer == null ? 1 : opts.layer;
    g.traverse(function (o) { o.layers.set(layer); });
    g.userData = { h: h, r: r, camY: h * camFrac, layer: layer };
    return g;
  }

  /** a third-person camera, which unlike the onboard one may see robot bodies */
  function thirdPerson(THREE, fov, layers) {
    var c = new THREE.PerspectiveCamera(fov == null ? 42 : fov, 1, 0.1, 250);
    (layers || [1]).forEach(function (l) { c.layers.enable(l); });
    return c;
  }

  /** where the onboard camera sits, given the robot group and a pitch in degrees */
  function camPose(THREE, rob, pitchDeg, cam) {
    var p = new THREE.Vector3();
    rob.getWorldPosition(p);
    cam.position.set(p.x, rob.userData.camY, p.z);
    var yaw = rob.rotation.y;
    var pitch = -(pitchDeg || 0) * Math.PI / 180;
    var look = new THREE.Vector3(
      p.x + Math.sin(yaw) * Math.cos(pitch),
      rob.userData.camY + Math.sin(pitch),
      p.z + Math.cos(yaw) * Math.cos(pitch)
    );
    cam.lookAt(look);
    cam.updateMatrixWorld();
  }

  /** project a world point into a camera, returning normalised image coords */
  function project(THREE, pt, cam) {
    var v = pt.clone().project(cam);
    var fwd = new THREE.Vector3();
    cam.getWorldDirection(fwd);
    var rel = pt.clone().sub(cam.position);
    return {
      u: (v.x + 1) / 2,
      v: (1 - v.y) / 2,
      visible: rel.dot(fwd) > 0 && v.x >= -1 && v.x <= 1 && v.y >= -1 && v.y <= 1,
      depth: rel.length()
    };
  }

  /* ---------- a marker for the waypoint, in world space ---------- */
  function waypoint(THREE, color) {
    var g = new THREE.Group();
    var col = color == null ? C.red : color;
    var ring = new THREE.Mesh(new THREE.RingGeometry(0.26, 0.32, 28),
      new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.02;
    g.add(ring);
    var post = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.7, 8),
      new THREE.MeshBasicMaterial({ color: col }));
    post.position.y = 0.35;
    g.add(post);
    g.userData.pulse = ring;
    return g;
  }

  /** a route drawn as a tube on the floor */
  function route(THREE, pts, color) {
    var curve = new THREE.CatmullRomCurve3(pts);
    var geo = new THREE.TubeGeometry(curve, Math.max(24, pts.length * 8), 0.045, 8, false);
    var m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: color == null ? C.blue : color }));
    m.userData.curve = curve;
    return m;
  }

  /* ---------- mount ----------
     `build(ctx)` runs once, when THREE has loaded and the holder has a size.
     It returns an object with an optional `frame(t, ctx)`. Everything else,
     resizing, viewport splitting and re-parenting the shared canvas, is here. */
  function mount(host, opts, build) {
    opts = opts || {};
    var holder = document.createElement('div');
    holder.className = 'k3';
    holder.style.position = 'relative';
    holder.style.width = '100%';
    holder.style.aspectRatio = opts.aspect || '16 / 10';
    host.appendChild(holder);

    // inside the holder, which is the positioned ancestor, so inset:0 lines the
    // overlay up with the canvas rather than with the whole scene column
    var overlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    overlay.setAttribute('class', 'k3ov');
    overlay.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none';
    holder.appendChild(overlay);

    var ctx = null, live = false, scratch = null;

    ready(function (THREE) {
      var rd = renderer(THREE);
      var scene = new THREE.Scene();
      scene.background = new THREE.Color(C.bg);

      var hemi = new THREE.HemisphereLight(0xffffff, 0xcfc6b4, 2.1);
      scene.add(hemi);
      var dir = new THREE.DirectionalLight(0xfff6e8, 2.0);
      dir.position.set(6, 12, 8);
      scene.add(dir);
      var fill = new THREE.DirectionalLight(0xdce7f0, 0.7);
      fill.position.set(-8, 6, -6);
      scene.add(fill);

      ctx = {
        THREE: THREE, scene: scene, renderer: rd, holder: holder,
        overlay: overlay, C: C, mats: mats(THREE),
        world: world, robot: robot, camPose: camPose, project: project,
        waypoint: waypoint, route: route, rng: rng, thirdPerson: thirdPerson,
        w: 1, h: 1
      };
      try { scratch = build(ctx) || {}; }
      catch (e) { console.error('k3 build', e); scratch = {}; }
      if (live) attach();
    });

    function size() {
      if (!ctx) return;
      var r = holder.getBoundingClientRect();
      ctx.w = Math.max(1, Math.round(r.width));
      ctx.h = Math.max(1, Math.round(r.height));
      ctx.renderer.setSize(ctx.w, ctx.h, false);
      overlay.setAttribute('viewBox', '0 0 ' + ctx.w + ' ' + ctx.h);
      if (scratch && scratch.resize) { try { scratch.resize(ctx); } catch (e) {} }
    }

    function attach() {
      if (!ctx) return;
      // the canvas must sit under the overlay, so insert it first rather than append
      if (ctx.renderer.domElement.parentNode !== holder) {
        holder.insertBefore(ctx.renderer.domElement, holder.firstChild);
      }
      size();
    }

    var ro = window.ResizeObserver ? new ResizeObserver(function () { if (live) size(); }) : null;
    if (ro) ro.observe(holder);

    return {
      start: function () { live = true; attach(); },
      stop: function () { live = false; },
      tick: function (t) {
        if (!live || !ctx || !scratch) return;
        if (scratch.frame) scratch.frame(t, ctx);
        else ctx.renderer.render(ctx.scene, scratch.camera);
      },
      ctx: function () { return ctx; }
    };
  }

  /** render two cameras side by side into the one canvas: the world, and what
      the robot sees. WebGL viewport coordinates start at the bottom left. */
  function split(ctx, camA, camB, gap) {
    var g = gap == null ? 8 : gap;
    var w = ctx.w, h = ctx.h;
    var half = Math.floor((w - g) / 2);
    var r = ctx.renderer;
    r.setScissorTest(true);
    r.setViewport(0, 0, half, h);
    r.setScissor(0, 0, half, h);
    camA.aspect = half / h; camA.updateProjectionMatrix();
    r.render(ctx.scene, camA);
    r.setViewport(half + g, 0, w - half - g, h);
    r.setScissor(half + g, 0, w - half - g, h);
    camB.aspect = (w - half - g) / h; camB.updateProjectionMatrix();
    r.render(ctx.scene, camB);
    r.setScissorTest(false);
    return { half: half, gap: g, rightX: half + g, rightW: w - half - g };
  }

  /* ---------- overlay drawing, shared so every 3D figure marks a waypoint the
     same way. A thin dark ring vanishes against furniture, so every stroke gets
     a paper-coloured halo underneath it. ---------- */
  function sv(tag, attrs) {
    var e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    return e;
  }
  function haloed(ov, tag, attrs) {
    var under = sv(tag, attrs);
    under.setAttribute('stroke', '#f4f1ea');
    under.setAttribute('stroke-width', (parseFloat(attrs['stroke-width'] || 2) + 2.8));
    under.setAttribute('stroke-opacity', '.92');
    ov.appendChild(under);
    ov.appendChild(sv(tag, attrs));
  }
  /** an open crosshair, open in the middle so it frames the target rather than
      covering it */
  function reticle(ov, x, y, opts) {
    opts = opts || {};
    var col = opts.color || '#a32d2d', r = opts.r || 16, w = opts.w || 2.4;
    haloed(ov, 'circle', { cx: x, cy: y, r: r, fill: 'none', stroke: col, 'stroke-width': w });
    [[-1, 0], [1, 0], [0, -1], [0, 1]].forEach(function (d) {
      haloed(ov, 'line', {
        x1: x + d[0] * (r + 6), y1: y + d[1] * (r + 6),
        x2: x + d[0] * (r + 15), y2: y + d[1] * (r + 15),
        stroke: col, 'stroke-width': w });
    });
    if (opts.label) {
      var tw = opts.label.length * 7.4 + 16;
      var lx = Math.max(4, Math.min(x + r + 12, (opts.maxX || 1e5) - tw));
      var ly = y - r - 26;
      if (ly < 4) ly = y + r + 8;
      ov.appendChild(sv('rect', { x: lx, y: ly, width: tw, height: 22, rx: 5,
        fill: col, opacity: .95 }));
      var t = sv('text', { x: lx + 8, y: ly + 15.5, fill: '#fff', 'font-size': 12.5,
        'font-weight': 600 });
      t.textContent = opts.label;
      ov.appendChild(t);
    }
  }
  /** the small uppercase caption in the corner of a viewport */
  function vlabel(ov, x, y, txt) {
    var t = sv('text', { x: x, y: y, fill: '#39404f', 'font-size': 11,
      'letter-spacing': '.14em', 'font-weight': 500,
      stroke: '#f4f1ea', 'stroke-width': 3.4, 'paint-order': 'stroke' });
    t.textContent = String(txt).toUpperCase();
    ov.appendChild(t);
  }
  /** a tinted banner across the foot of a viewport, for a state message */
  function banner(ov, x, y, w, txt, col) {
    col = col || '#a32d2d';
    ov.appendChild(sv('rect', { x: x, y: y, width: w, height: 38, rx: 8,
      fill: '#f4f1ea', opacity: .93 }));
    ov.appendChild(sv('rect', { x: x, y: y, width: w, height: 38, rx: 8,
      fill: 'none', stroke: col, 'stroke-width': 1.5 }));
    var t = sv('text', { x: x + 13, y: y + 24, fill: col, 'font-size': 13, 'font-weight': 600 });
    t.textContent = txt;
    ov.appendChild(t);
  }
  /** a faint guide line pair through the reticle, for the randomisation figure */
  function crossGuides(ov, x, y, x0, x1, h, col) {
    [[x0, y, x1, y], [x, 0, x, h]].forEach(function (l) {
      ov.appendChild(sv('line', { x1: l[0], y1: l[1], x2: l[2], y2: l[3],
        stroke: col || '#a32d2d', 'stroke-width': 1, 'stroke-dasharray': '3 6', opacity: .55 }));
    });
  }

  /* ---------- HTML furniture, so a 3D figure carries the same headline,
     subtitle and payoff box as an SVG one ---------- */
  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }
  function head(host, title, sub) {
    var w = el('div', 'k3head');
    w.appendChild(el('i'));
    w.appendChild(el('h4', null, title));
    if (sub) w.appendChild(el('p', null, sub));
    host.appendChild(w);
    return w;
  }
  function note(host, txt, tone) {
    var e = el('div', 'k3note' + (tone ? ' ' + tone : ''), txt);
    host.appendChild(e);
    return e;
  }
  /** a labelled slider row; returns a function to read the value */
  function slider(row, label, min, max, step, val, fmt, onInput) {
    var l = el('label', null, label);
    var i = document.createElement('input');
    i.type = 'range'; i.min = min; i.max = max; i.step = step; i.value = val;
    var o = document.createElement('output');
    function show() { o.textContent = fmt(parseFloat(i.value)); }
    i.addEventListener('input', function () { show(); onInput(parseFloat(i.value)); });
    show();
    l.appendChild(i); l.appendChild(o);
    row.appendChild(l);
    return function () { return parseFloat(i.value); };
  }
  /** a segmented control; calls onPick(index) and starts on `start` */
  function seg(row, items, onPick, start) {
    var g = el('div', 'seg');
    var btns = items.map(function (it, i) {
      var b = el('button', null, it);
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', function () {
        btns.forEach(function (o, j) { o.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
        onPick(i);
      });
      g.appendChild(b);
      return b;
    });
    row.appendChild(g);
    btns[start || 0].click();
    return btns;
  }
  function ctl(host) {
    var r = el('div', 'ctl');
    host.appendChild(r);
    return r;
  }

  window.K3 = {
    C: C, ready: ready, mount: mount, split: split,
    head: head, note: note, slider: slider, seg: seg, ctl: ctl, el: el,
    world: world, robot: robot, camPose: camPose, project: project,
    waypoint: waypoint, route: route, rng: rng, mats: mats,
    thirdPerson: thirdPerson,
    sv: sv, reticle: reticle, vlabel: vlabel, banner: banner,
    crossGuides: crossGuides
  };
})();
