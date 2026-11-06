/* Shared chart pieces for the ML4 figures: axes, series, a slider and a hover
   readout. Three data colours only, checked for colour-blind separation:
   amber is Mistral Large 4 (or the choice it made), blue is the one thing it is
   compared against, grey is context. Every value comes from window.ML4. */
(function () {
  var K = window.KIT, C = K.C, n = K.n;
  var GREY = '#8a909c', GREY2 = '#c3c7ce';

  function scale(d0, d1, r0, r1, log) {
    if (log) {
      var a = Math.log10(d0), b = Math.log10(d1);
      return function (v) { return r0 + (Math.log10(v) - a) / (b - a) * (r1 - r0); };
    }
    return function (v) { return r0 + (v - d0) / (d1 - d0) * (r1 - r0); };
  }

  /* axes. o: x0 x1 y0 y1, xt yt (tick values), xf yf (tick formatters),
     xl yl (axis labels), logx logy, noY (no left spine, for dot rows). The y label sits above the axis, never rotated. */
  function axes(p, x, y, w, h, o) {
    var g = n('g', {});
    p.appendChild(g);
    var sx = scale(o.x0, o.x1, x, x + w, o.logx), sy = scale(o.y0, o.y1, y + h, y, o.logy);
    var xf = o.xf || String, yf = o.yf || String;
    (o.yt || []).forEach(function (t) {
      g.appendChild(n('line', { x1: x, x2: x + w, y1: sy(t), y2: sy(t), stroke: C.line2 }));
      K.mono(g, x - 9, sy(t) + 4.5, yf(t), { size: 12.5, anchor: 'end', color: C.ink3 });
    });
    (o.xt || []).forEach(function (t) {
      g.appendChild(n('line', { x1: sx(t), x2: sx(t), y1: y, y2: y + h, stroke: C.line2 }));
      K.mono(g, sx(t), y + h + 20, xf(t), { size: 12.5, anchor: 'middle', color: C.ink3 });
    });
    g.appendChild(n('path', { d: (o.noY ? 'M' + x + ' ' + (y + h) : 'M' + x + ' ' + y + ' V' + (y + h)) + ' H' + (x + w),
      stroke: C.ink3, fill: 'none', 'stroke-width': 1 }));
    if (o.xl) K.label(g, x + w, y + h + 44, o.xl, { anchor: 'end' });
    if (o.yl) K.label(g, x - (o.ylIn ? 0 : 0), y - 14, o.yl);
    return { g: g, sx: sx, sy: sy, x: x, y: y, w: w, h: h, o: o };
  }

  /** polyline through [[x,y],...] in data units */
  function line(p, ax, pts, o) {
    o = o || {};
    var d = pts.filter(function (q) { return q[1] != null; }).map(function (q, i) {
      return (i ? 'L' : 'M') + ax.sx(q[0]).toFixed(1) + ' ' + ax.sy(q[1]).toFixed(1);
    }).join(' ');
    var e = n('path', { d: d, fill: 'none', stroke: o.color || C.amber, 'stroke-width': o.w || 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': o.dash || null,
      opacity: o.opacity == null ? 1 : o.opacity });
    p.appendChild(e);
    return e;
  }

  /** markers with a 2px surface ring, so overlapping ones stay separable */
  function dots(p, ax, pts, o) {
    o = o || {};
    var g = n('g', {});
    pts.forEach(function (q) {
      if (q[1] == null) return;
      g.appendChild(n('circle', { cx: ax.sx(q[0]).toFixed(1), cy: ax.sy(q[1]).toFixed(1),
        r: o.r || 4.2, fill: o.hollow ? '#fff' : (o.color || C.amber),
        stroke: o.hollow ? (o.color || C.amber) : '#fff', 'stroke-width': o.hollow ? 2 : 1.6,
        opacity: o.opacity == null ? 1 : o.opacity }));
    });
    p.appendChild(g);
    return g;
  }

  /** one reusable readout line, so hovering never adds text nodes */
  function readout(p, x, y, o) {
    o = o || {};
    var t = K.mono(p, x, y, o.text || '', { size: o.size || 13, color: o.color || C.ink,
      anchor: o.anchor || 'start', weight: 500 });
    return { el: t, set: function (s, col) { t.textContent = s; if (col) t.setAttribute('fill', col); } };
  }

  /* An invisible, generous hit target that reports hover, focus and click (the
     last one is for touch). Marked data-hover so the page checker exercises it. */
  function hot(p, cx, cy, r, on, o) {
    o = o || {};
    var c = n('circle', { cx: cx, cy: cy, r: r, fill: 'transparent', tabindex: 0,
      'data-hover': '1', 'aria-label': o.label || 'point', style: 'cursor:pointer;outline:none' });
    if (o.rect) {
      c = n('rect', { x: o.rect[0], y: o.rect[1], width: o.rect[2], height: o.rect[3], fill: 'transparent',
        tabindex: 0, 'data-hover': '1', 'aria-label': o.label || 'item', style: 'cursor:pointer;outline:none' });
    }
    ['mouseenter', 'focus', 'click'].forEach(function (ev) { c.addEventListener(ev, on); });
    p.appendChild(c);
    return c;
  }

  function svgOf(el) { return el.ownerSVGElement || el; }
  function toLocal(el, evt) {
    var s = svgOf(el), pt = s.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    return pt.matrixTransform(el.getScreenCTM().inverse());
  }

  /* A slider drawn in SVG. Drag, click the track, or use the arrow keys.
     o: min max step value label fmt onChange color */
  function slider(p, x, y, w, o) {
    var g = n('g', { tabindex: 0, role: 'slider', 'aria-label': o.label || 'slider',
      'aria-valuemin': o.min, 'aria-valuemax': o.max, style: 'cursor:pointer;outline:none' });
    var col = o.color || C.amber;
    g.appendChild(n('rect', { x: x - 12, y: y - 16, width: w + 24, height: 32, fill: 'transparent' }));
    g.appendChild(n('rect', { x: x, y: y - 3, width: w, height: 6, rx: 3, fill: K.TINT.ink }));
    var fill = n('rect', { x: x, y: y - 3, width: 0, height: 6, rx: 3, fill: col, opacity: .55 });
    g.appendChild(fill);
    var knob = n('circle', { cx: x, cy: y, r: 9.5, fill: '#fff', stroke: col, 'stroke-width': 2.5 });
    g.appendChild(knob);
    p.appendChild(g);
    var v = o.value == null ? o.min : o.value, step = o.step || 0;
    function set(nv, fire) {
      nv = Math.max(o.min, Math.min(o.max, nv));
      if (step) nv = Math.round((nv - o.min) / step) * step + o.min;
      v = +nv.toFixed(6);
      var px = x + (v - o.min) / (o.max - o.min) * w;
      knob.setAttribute('cx', px);
      fill.setAttribute('width', Math.max(0, px - x));
      g.setAttribute('aria-valuenow', v);
      if (o.fmt) g.setAttribute('aria-valuetext', o.fmt(v));
      if (fire !== false && o.onChange) o.onChange(v);
    }
    function fromEvt(e) { var q = toLocal(g, e); set(o.min + (q.x - x) / w * (o.max - o.min)); }
    var drag = false;
    g.addEventListener('pointerdown', function (e) {
      drag = true; try { g.setPointerCapture(e.pointerId); } catch (er) {}
      fromEvt(e); e.preventDefault();
    });
    g.addEventListener('pointermove', function (e) { if (drag) fromEvt(e); });
    g.addEventListener('pointerup', function () { drag = false; });
    g.addEventListener('pointercancel', function () { drag = false; });
    g.addEventListener('keydown', function (e) {
      var s = step || (o.max - o.min) / 50;
      if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { set(v + s); e.preventDefault(); }
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') { set(v - s); e.preventDefault(); }
      if (e.key === 'Home') { set(o.min); e.preventDefault(); }
      if (e.key === 'End') { set(o.max); e.preventDefault(); }
    });
    set(v);
    return { g: g, set: set, get: function () { return v; } };
  }

  /** short line swatch plus a name, for legends */
  function key(p, x, y, name, o) {
    o = o || {};
    p.appendChild(n('path', { d: 'M' + x + ' ' + (y - 4.5) + ' h18', stroke: o.color || C.amber,
      'stroke-width': o.w || 2.5, 'stroke-dasharray': o.dash || null, 'stroke-linecap': 'round' }));
    if (o.dot !== false) p.appendChild(n('circle', { cx: x + 9, cy: y - 4.5, r: 3.6,
      fill: o.color || C.amber, stroke: '#fff', 'stroke-width': 1.2 }));
    return K.mono(p, x + 25, y, name, { size: o.size || 12.5, color: o.text || C.ink2 });
  }

  /** a body group that a scene clears and redraws on every interaction */
  function body(p) { var g = n('g', {}); p.appendChild(g); return g; }
  function clear(g) { while (g.firstChild) g.removeChild(g.firstChild); }

  /* A column (or row) of fixed-width choice buttons, for when a horizontal
     switcher does not fit. o: w h gap tint horizontal. Returns {pick, btns}. */
  function pick(p, x, y, items, onPick, o) {
    o = o || {};
    var w = o.w || 150, h = o.h || 30, gap = o.gap || 9, tint = o.tint || 'amber', btns = [];
    items.forEach(function (it, i) {
      var bx = o.horizontal ? x + i * (w + gap) : x, by = o.horizontal ? y : y + i * (h + gap);
      var g = n('g', { tabindex: 0, role: 'button', 'aria-label': it, style: 'cursor:pointer;outline:none' });
      g.appendChild(n('rect', { x: bx, y: by, width: w, height: h, rx: 8, fill: '#fff', stroke: C.line }));
      K.mono(g, bx + 12, by + h / 2 + 4.5, it, { size: o.size || 13, color: C.ink2 });
      function go() {
        btns.forEach(function (b, j) {
          var on = j === i;
          b.firstChild.setAttribute('fill', on ? K.TINT[tint] : '#fff');
          b.firstChild.setAttribute('stroke', on ? C[tint] : C.line);
          b.lastChild.setAttribute('fill', on ? C[tint] : C.ink2);
          b.lastChild.setAttribute('font-weight', on ? 600 : 400);
        });
        onPick(i);
      }
      g.addEventListener('click', go);
      g.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); }
      });
      g._go = go;
      p.appendChild(g);
      btns.push(g);
    });
    return { btns: btns, pick: function (i) { btns[i]._go(); } };
  }

  /** a single action button, for "step", "send", "reset" */
  function button(p, x, y, label, on, o) {
    o = o || {};
    var col = C[o.tint || 'amber'];
    var g = n('g', { tabindex: 0, role: 'button', 'aria-label': label, style: 'cursor:pointer;outline:none' });
    var w = o.w || (label.length * 8 + 30);
    g.appendChild(n('rect', { x: x, y: y, width: w, height: 30, rx: 15, fill: col }));
    K.mono(g, x + w / 2, y + 19.5, label, { size: 13, color: '#fff', anchor: 'middle', weight: 600 });
    g.addEventListener('click', on);
    g.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); on(); }
    });
    p.appendChild(g);
    return g;
  }

  function fmt(v, d) { return (+v).toFixed(d == null ? 1 : d); }
  function pct(v, d) { return fmt(v, d) + '%'; }
  function comma(v) { return Math.round(v).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); }

  /** a seeded generator, so a figure draws the same way on every visit */
  function rng(seed) {
    var st = seed >>> 0;
    return function () { st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296; };
  }

  window.CH = { GREY: GREY, GREY2: GREY2, scale: scale, axes: axes, line: line, dots: dots,
    readout: readout, hot: hot, slider: slider, key: key, body: body, clear: clear,
    pick: pick, button: button, fmt: fmt, pct: pct, comma: comma, rng: rng, toLocal: toLocal };
})();
