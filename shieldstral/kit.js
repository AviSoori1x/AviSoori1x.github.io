/* One drawing vocabulary for all 34 figures.

   The previous attempt let every figure invent its own layout and the page read
   as thirty-four unrelated dashboards. Everything here draws into a fixed
   640 x 520 SVG board using the same handful of primitives, so the figures share
   a visual language by construction. */
(function () {
  var W = 640, H = 520;
  var NS = 'http://www.w3.org/2000/svg';

  var C = {
    ink: '#141922', ink2: '#39404f', ink3: '#5d6474',
    line: 'rgba(31,37,48,.14)', line2: 'rgba(31,37,48,.08)',
    panel: '#ffffff', bg: '#f4f1ea',
    blue: '#1c6e8c', teal: '#0f6e56', amber: '#b4551a',
    purple: '#6b3fa8', red: '#a32d2d'
  };
  var TINT = {
    blue: 'rgba(28,110,140,.08)', teal: 'rgba(15,110,86,.08)',
    amber: 'rgba(180,85,26,.09)', purple: 'rgba(107,63,168,.08)',
    red: 'rgba(163,45,45,.07)', ink: 'rgba(31,37,48,.05)'
  };

  function n(tag, a) {
    var e = document.createElementNS(NS, tag);
    for (var k in a) if (a.hasOwnProperty(k) && a[k] != null) e.setAttribute(k, a[k]);
    return e;
  }

  /* ---------- board ---------- */
  function board(root, opts) {
    opts = opts || {};
    var svg = n('svg', {
      viewBox: '0 0 ' + W + ' ' + H,
      width: '100%', role: 'img',
      'aria-label': opts.alt || 'figure'
    });
    svg.style.display = 'block';
    var ids = defs(svg);
    // engineering grid behind everything, then a grain wash over the top
    var bg = n('rect', { x: -40, y: -120, width: W + 200, height: H + 320,
      fill: 'url(#' + ids.gd + ')', 'pointer-events': 'none' });
    bg.setAttribute('mask', '');
    svg.appendChild(bg);
    svg._grainAfter = true;
    root.appendChild(svg);
    // Trim the board to whatever the scene actually drew. A fixed height left a
    // dead half-screen under the shorter figures.
    requestAnimationFrame(function () {
      try {
        // place the honesty line under whatever was drawn, then fit the board
        if (svg._foot) {
          var fb = svg.getBBox();
          wrap(svg._foot, 92).forEach(function (ln, i) {
            var tn = n('text', {
              x: 0, y: fb.y + fb.height + 26 + i * 14,
              'font-family': "'IBM Plex Sans',system-ui,sans-serif",
              'font-size': 11.5, fill: C.ink3
            });
            tn.textContent = ln;
            svg.appendChild(tn);
          });
          svg._foot = null;
        }
        var bb = svg.getBBox(), pad = 10;
        var x0 = Math.min(0, bb.x - pad), y0 = Math.min(0, bb.y - pad);
        var x1 = Math.max(W, bb.x + bb.width + pad);
        var y1 = bb.y + bb.height + pad;
        svg.setAttribute('viewBox', x0 + ' ' + y0 + ' ' + (x1 - x0) + ' ' + (y1 - y0));
        if (svg._grainAfter) {
          svg._grainAfter = false;
          var gn = n('rect', { x: x0, y: y0, width: x1 - x0, height: y1 - y0,
            filter: 'url(#' + svg._ids.gr + ')', opacity: .05,
            'mix-blend-mode': 'multiply', 'pointer-events': 'none' });
          svg.appendChild(gn);
        }
      } catch (e) {}
    });
    return svg;
  }


  /* ---------- one defs block per board: depth, texture, sheen ----------
     The palette and type are fixed by the reference, so all the craft has to
     come from surface: a real cast shadow instead of a hairline, a paper grain
     over the whole board, a faint engineering grid behind it, and a slight
     sheen down every filled bar. */
  var UID = 0;
  function defs(svg) {
    var id = 'k' + (++UID);
    var d = n('defs', {});

    var f = n('filter', { id: id + 'sh', x: '-20%', y: '-20%', width: '150%', height: '160%' });
    f.appendChild(n('feDropShadow', {
      dx: 0, dy: 6, stdDeviation: 9, 'flood-color': '#141922', 'flood-opacity': .10 }));
    f.appendChild(n('feDropShadow', {
      dx: 0, dy: 1, stdDeviation: 1.5, 'flood-color': '#141922', 'flood-opacity': .07 }));
    d.appendChild(f);

    var grain = n('filter', { id: id + 'gr', x: '0%', y: '0%', width: '100%', height: '100%' });
    grain.appendChild(n('feTurbulence', {
      type: 'fractalNoise', baseFrequency: '.9', numOctaves: 3, stitchTiles: 'stitch' }));
    grain.appendChild(n('feColorMatrix', { type: 'saturate', values: '0' }));
    d.appendChild(grain);

    var grid = n('pattern', { id: id + 'gd', width: 28, height: 28,
      patternUnits: 'userSpaceOnUse' });
    grid.appendChild(n('circle', { cx: 1, cy: 1, r: 1, fill: 'rgba(31,37,48,.13)' }));
    d.appendChild(grid);

    var sheen = n('linearGradient', { id: id + 'sn', x1: 0, y1: 0, x2: 0, y2: 1 });
    sheen.appendChild(n('stop', { offset: 0, 'stop-color': '#fff', 'stop-opacity': .26 }));
    sheen.appendChild(n('stop', { offset: 1, 'stop-color': '#fff', 'stop-opacity': 0 }));
    d.appendChild(sheen);

    svg.appendChild(d);
    svg._ids = { sh: id + 'sh', gr: id + 'gr', gd: id + 'gd', sn: id + 'sn' };
    return svg._ids;
  }

  /* ---------- primitives ---------- */

  /** white card with a hairline border */
  function panel(p, x, y, w, h, o) {
    o = o || {};
    var g = n('g', {});
    var r = o.r == null ? 10 : o.r;
    var svg = p.ownerSVGElement || p;
    var ids = (svg && svg._ids) || {};
    var base = n('rect', {
      x: x, y: y, width: w, height: h, rx: r,
      fill: o.fill || C.panel, stroke: o.stroke || C.line, 'stroke-width': 1
    });
    if (o.flat !== true && ids.sh) base.setAttribute('filter', 'url(#' + ids.sh + ')');
    g.appendChild(base);
    // a one pixel lit edge along the top, which is what stops a flat rect
    // reading as a flat rect
    if (o.flat !== true) {
      g.appendChild(n('path', {
        d: 'M' + (x + r) + ' ' + (y + .75) + ' H' + (x + w - r),
        stroke: 'rgba(255,255,255,.85)', 'stroke-width': 1.2, fill: 'none'
      }));
    }
    p.appendChild(g);
    return g;
  }

  /** small uppercase mono label, the caption voice of the whole guide */
  function label(p, x, y, s, o) {
    o = o || {};
    var t = n('text', {
      x: x, y: y, 'font-family': "'JetBrains Mono',monospace",
      'font-size': o.size || 10.5, 'letter-spacing': '.14em',
      fill: o.color || C.ink3, 'text-anchor': o.anchor || 'start'
    });
    t.textContent = String(s).toUpperCase();
    p.appendChild(t);
    return t;
  }

  /** mono body line, for prompts, tokens and data */
  function mono(p, x, y, s, o) {
    o = o || {};
    var t = n('text', {
      x: x, y: y, 'font-family': "'JetBrains Mono',monospace",
      'font-size': o.size || 13, fill: o.color || C.ink2,
      'text-anchor': o.anchor || 'start', 'font-weight': o.weight || 400
    });
    t.textContent = s;
    p.appendChild(t);
    return t;
  }

  /** sans line, used sparingly, the cards carry the prose */
  function text(p, x, y, s, o) {
    o = o || {};
    var t = n('text', {
      x: x, y: y, 'font-family': "'IBM Plex Sans',system-ui,sans-serif",
      'font-size': o.size || 14, fill: o.color || C.ink2,
      'text-anchor': o.anchor || 'start', 'font-weight': o.weight || 400
    });
    t.textContent = s;
    p.appendChild(t);
    return t;
  }

  /** the big display numeral, one per figure at most */
  function big(p, x, y, s, o) {
    o = o || {};
    var t = n('text', {
      x: x, y: y, 'font-family': "'Sora',system-ui,sans-serif",
      'font-size': o.size || 62, 'font-weight': 700, 'letter-spacing': '-.03em',
      fill: o.color || C.ink, 'text-anchor': o.anchor || 'start'
    });
    t.textContent = s;
    p.appendChild(t);
    return t;
  }

  /** pill, for a category or a state */
  function chip(p, x, y, s, o) {
    o = o || {};
    var pad = 9, size = o.size || 11.5;
    var w = o.w || (String(s).length * size * 0.62 + pad * 2);
    var g = n('g', {});
    g.appendChild(n('rect', {
      x: x, y: y, width: w, height: o.h || 23, rx: 11.5,
      fill: o.fill || 'none', stroke: o.stroke || C.line, 'stroke-width': 1
    }));
    var t = n('text', {
      x: x + w / 2, y: y + (o.h || 23) / 2 + size * 0.36,
      'font-family': "'JetBrains Mono',monospace", 'font-size': size,
      fill: o.color || C.ink3, 'text-anchor': 'middle'
    });
    t.textContent = s;
    g.appendChild(t);
    p.appendChild(g);
    g._w = w;
    // width was estimated from character count and under-measured, so long
    // labels ran outside the pill. Measure once laid out and resize.
    requestAnimationFrame(function () {
      try {
        var real = t.getComputedTextLength();
        if (real + pad * 2 > w - 1) {
          var nw = real + pad * 2;
          g.firstChild.setAttribute('width', nw);
          t.setAttribute('x', x + nw / 2);
          g._w = nw;
        }
      } catch (e) {}
    });
    return g;
  }

  /** horizontal proportional bar on a track */
  function bar(p, x, y, w, h, frac, o) {
    o = o || {};
    var g = n('g', {});
    var svg2 = p.ownerSVGElement || p;
    var ids2 = (svg2 && svg2._ids) || {};
    g.appendChild(n('rect', { x: x, y: y, width: w, height: h, rx: h / 2,
      fill: o.track || TINT.ink }));
    g.appendChild(n('path', {
      d: 'M' + (x + h / 2) + ' ' + (y + .6) + ' H' + (x + w - h / 2),
      stroke: 'rgba(31,37,48,.10)', 'stroke-width': 1.1, fill: 'none' }));
    var fw = Math.max(0, Math.min(1, frac)) * w;
    var f = n('rect', { x: x, y: y, width: fw, height: h, rx: h / 2,
      fill: o.color || C.blue });
    g.appendChild(f);
    if (h >= 9 && ids2.sn) {
      g.appendChild(n('rect', { x: x, y: y, width: fw, height: h / 2, rx: h / 2,
        fill: 'url(#' + ids2.sn + ')', 'pointer-events': 'none' }));
    }
    p.appendChild(g);
    g._fill = f;
    return g;
  }

  /** connector with an arrow head */
  function arrow(p, x1, y1, x2, y2, o) {
    o = o || {};
    var col = o.color || C.line;
    var g = n('g', {});
    var d;
    if (o.curve) {
      var mx = (x1 + x2) / 2;
      d = 'M' + x1 + ' ' + y1 + ' C' + mx + ' ' + y1 + ' ' + mx + ' ' + y2 + ' ' + x2 + ' ' + y2;
    } else {
      d = 'M' + x1 + ' ' + y1 + ' L' + x2 + ' ' + y2;
    }
    g.appendChild(n('path', { d: d, fill: 'none', stroke: col,
      'stroke-width': o.w || 1.2, 'stroke-dasharray': o.dash || null }));
    if (o.head !== false) {
      var a = Math.atan2(y2 - y1, x2 - (o.curve ? (x1 + x2) / 2 : x1));
      var s = o.headSize || 6;
      g.appendChild(n('path', {
        d: 'M' + x2 + ' ' + y2 +
           ' L' + (x2 - s * Math.cos(a - 0.42)) + ' ' + (y2 - s * Math.sin(a - 0.42)) +
           ' L' + (x2 - s * Math.cos(a + 0.42)) + ' ' + (y2 - s * Math.sin(a + 0.42)) + ' Z',
        fill: col
      }));
    }
    p.appendChild(g);
    return g;
  }

  /** dark code slab, matching the reference's .codecard */
  function code(p, x, y, w, lines, o) {
    o = o || {};
    var lh = o.lh || 17, padY = 14, padX = 14;
    var h = lines.length * lh + padY * 2;
    var g = n('g', {});
    g.appendChild(n('rect', { x: x, y: y, width: w, height: h, rx: 10, fill: '#151a23' }));
    lines.forEach(function (ln, i) {
      var t = n('text', {
        x: x + padX, y: y + padY + lh * (i + 0.72),
        'font-family': "'JetBrains Mono',monospace", 'font-size': o.size || 12.5,
        fill: ln.c || '#c8cfdb'
      });
      t.textContent = ln.t == null ? ln : ln.t;
      g.appendChild(t);
    });
    p.appendChild(g);
    g._h = h;
    return g;
  }

  /** the yes / no verdict, the recurring motif of the whole guide */
  function verdict(p, x, y, yes, o) {
    o = o || {};
    var g = n('g', {});
    var col = yes ? C.red : C.teal;
    big(g, x, y, yes ? 'yes' : 'no', { size: o.size || 58, color: col });
    label(g, x, y + 28, yes ? 'flagged' : 'not flagged', { color: col, size: 10.5 });
    p.appendChild(g);
    return g;
  }

  /** wrap a string to a width in characters, returning lines */
  function wrap(s, cols) {
    var words = String(s).split(/\s+/), out = [], cur = '';
    words.forEach(function (w) {
      if ((cur + ' ' + w).trim().length > cols) { out.push(cur.trim()); cur = w; }
      else cur += ' ' + w;
    });
    if (cur.trim()) out.push(cur.trim());
    return out;
  }

  /** a block of mono text, wrapped */
  function para(p, x, y, s, cols, o) {
    o = o || {};
    var lines = wrap(s, cols), lh = o.lh || 18;
    lines.forEach(function (ln, i) {
      mono(p, x, y + i * lh, ln, { size: o.size || 13, color: o.color });
    });
    return { h: lines.length * lh, lines: lines.length };
  }

  /** the honesty line every schematic figure carries, placed after layout */
  function foot(svg, s) { svg._foot = s; }

  /** interactive switch row, returns the group and wires selection */
  function switcher(p, x, y, items, onPick, o) {
    o = o || {};
    var g = n('g', {}), cx = x, btns = [];
    items.forEach(function (it, i) {
      var b = chip(g, cx, y, it, { size: o.size || 11.5, h: 25 });
      b.style.cursor = 'pointer';
      b.setAttribute('tabindex', '0');
      b.setAttribute('role', 'button');
      b.setAttribute('aria-label', it);
      var pick = function () {
        btns.forEach(function (o2, j) {
          var on = j === i;
          o2.firstChild.setAttribute('fill', on ? TINT[o.tint || 'blue'] : 'none');
          o2.firstChild.setAttribute('stroke', on ? C[o.tint || 'blue'] : C.line);
          o2.lastChild.setAttribute('fill', on ? C[o.tint || 'blue'] : C.ink3);
        });
        onPick(i);
      };
      b.addEventListener('click', pick);
      b.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); }
      });
      btns.push(b);
      cx += b._w + 7;
    });
    p.appendChild(g);
    // pills settle at their measured width, so re-flow the row afterwards
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        var rx = x;
        btns.forEach(function (b2) {
          var cur = parseFloat(b2.firstChild.getAttribute('x'));
          b2.setAttribute('transform', 'translate(' + (rx - cur) + ',0)');
          rx += b2._w + 7;
        });
      });
    });
    if (btns.length) btns[0].dispatchEvent(new Event('click'));
    return { g: g, btns: btns, pick: function (i) { btns[i].dispatchEvent(new Event('click')); } };
  }


  /* ---------- figure furniture, the thing that was missing ----------
     The reference gives every art panel its own headline, a mono subtitle and a
     tinted payoff box. Ours were bare diagrams with a small label, which is why
     they read as sparse. These draw ABOVE y=0 so existing layouts do not shift. */

  function head(p, title, sub) {
    p.appendChild(n('rect', { x: 0, y: -78, width: 34, height: 3, rx: 1.5, fill: C.amber }));
    var t1 = n('text', {
      x: 0, y: -46, 'font-family': "'Sora',system-ui,sans-serif",
      'font-size': 27, 'font-weight': 700, 'letter-spacing': '-.022em', fill: C.ink
    });
    t1.textContent = title;
    p.appendChild(t1);
    if (sub) {
      var t2 = n('text', {
        x: 0, y: -22, 'font-family': "'JetBrains Mono',monospace",
        'font-size': 13, fill: C.ink3
      });
      t2.textContent = sub;
      p.appendChild(t2);
    }
  }

  /** the tinted box that carries a figure's punchline */
  function callout(p, x, y, w, s, o) {
    o = o || {};
    var col = o.color || C.amber, tint = o.tint || TINT.amber;
    var lines = wrap(s, o.cols || 62), lh = 19;
    var h = lines.length * lh + 22;
    p.appendChild(n('rect', { x: x, y: y, width: w, height: h, rx: 9,
      fill: tint, stroke: col, 'stroke-opacity': .45 }));
    lines.forEach(function (ln, i) {
      var tn = n('text', {
        x: x + 16, y: y + 22 + i * lh, 'font-family': "'JetBrains Mono',monospace",
        'font-size': o.size || 13.5, fill: col, 'font-weight': 500
      });
      tn.textContent = ln;
      p.appendChild(tn);
    });
    return h;
  }

  /** one sans sentence of narration under the headline */
  function lede(p, y, s, cols) {
    var lines = wrap(s, cols || 74), lh = 21;
    lines.forEach(function (ln, i) {
      var tn = n('text', {
        x: 0, y: y + i * lh, 'font-family': "'IBM Plex Sans',system-ui,sans-serif",
        'font-size': 14.5, fill: C.ink2
      });
      tn.textContent = ln;
      p.appendChild(tn);
    });
    return lines.length * lh;
  }

  window.KIT = {
    W: W, H: H, C: C, TINT: TINT, n: n,
    board: board, panel: panel, label: label, mono: mono, text: text, big: big,
    chip: chip, bar: bar, arrow: arrow, code: code, verdict: verdict,
    wrap: wrap, para: para, foot: foot, switcher: switcher,
    head: head, callout: callout, lede: lede
  };
})();
