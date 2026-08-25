/* Scroll engine for the Shieldstral visual guide.
   Each .beat carries data-s="SCENE_ID". The matching entry in window.SCENES
   builds the live art once, lazily, and is then just shown or hidden. */
(function () {
  var SS = window.SS || {};
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- tiny DOM helpers handed to every scene ---------- */
  function el(tag, cls, txt) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt != null) n.textContent = txt;
    return n;
  }
  function svg(tag, attrs) {
    var n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) if (attrs.hasOwnProperty(k)) n.setAttribute(k, attrs[k]);
    return n;
  }
  function frag(html) {
    var d = document.createElement('div');
    d.innerHTML = html;
    return d;
  }
  function num(n, d) {
    return (n == null) ? 'n/a' : Number(n).toFixed(d == null ? 1 : d);
  }
  var api = { SS: SS, el: el, svg: svg, frag: frag, num: num, reduce: reduce };
  window.SS_API = api;

  var stage = document.getElementById('stageInner');
  var beats = Array.prototype.slice.call(document.querySelectorAll('.beat'));
  var rail = document.getElementById('railwrap');
  var status = document.getElementById('status');
  var built = {};
  var current = null;
  var ticking = null;

  /* ---------- rail ---------- */
  beats.forEach(function (b, i) {
    var t = el('b');
    t.className = 'a' + (b.getAttribute('data-act') || '1');
    t.title = (b.getAttribute('data-title') || ('Beat ' + (i + 1)));
    rail.appendChild(t);
    b._pip = t;
  });

  function sceneFor(id) {
    if (built[id]) return built[id];
    var host = el('div', 'scene');
    host.setAttribute('data-scene', id);
    var fn = (window.SCENES || {})[id];
    var handle = null;
    if (typeof fn === 'function') {
      try {
        handle = fn(host, api) || null;
      } catch (err) {
        // a broken scene must never take the page down with it
        host.innerHTML = '';
        host.appendChild(frag('<div style="font-family:var(--fm);font-size:12px;color:var(--ink4)">'
          + 'scene ' + id + ' failed to build</div>'));
        if (window.console) console.error('scene ' + id, err);
      }
    } else {
      host.appendChild(frag('<div style="font-family:var(--fm);font-size:12px;color:var(--ink4)">'
        + id + '</div>'));
    }
    // Some scenes assign root.className rather than adding to it, which strips the
    // class the stage positions and fades with. Re-assert it rather than trusting them.
    host.classList.add('scene');
    host.setAttribute('data-scene', id);
    stage.appendChild(host);
    built[id] = { host: host, handle: handle };
    return built[id];
  }

  function show(beat) {
    if (!beat || beat === current) return;
    var id = beat.getAttribute('data-s');
    var next = sceneFor(id);

    Object.keys(built).forEach(function (k) {
      built[k].host.classList.toggle('on', built[k] === next);
    });

    if (ticking && ticking.stop) { try { ticking.stop(); } catch (e) {} }
    ticking = (next.handle && next.handle.tick) ? next.handle : null;
    if (ticking && ticking.start && !reduce) { try { ticking.start(); } catch (e) {} }

    beats.forEach(function (b) { b._pip.classList.toggle('on', b === beat); });

    var n = beats.indexOf(beat) + 1;
    var act = beat.getAttribute('data-act') || '1';
    var sec = beat.getAttribute('data-sec') || '';
    status.innerHTML = 'Act ' + ['', 'I', 'II', 'III', 'IV'][+act] +
      (sec ? ' &nbsp;·&nbsp; ' + sec : '') +
      ' &nbsp;·&nbsp; <b>' + n + ' / ' + beats.length + '</b>';

    current = beat;
  }

  /* ---------- which beat is in the reading band ---------- */
  var wide = matchMedia('(min-width: 70.01rem)');

  function pick() {
    var band = window.innerHeight * (wide.matches ? 0.5 : 0.62);
    var best = null, bestD = Infinity;
    for (var i = 0; i < beats.length; i++) {
      var r = beats[i].getBoundingClientRect();
      if (r.bottom < 0 || r.top > window.innerHeight) continue;
      var d = Math.abs((r.top + r.height / 2) - band);
      if (d < bestD) { bestD = d; best = beats[i]; }
    }
    if (best) show(best);
  }

  var raf = 0;
  function onScroll() {
    if (raf) return;
    raf = requestAnimationFrame(function () { raf = 0; pick(); });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);

  /* on narrow screens the art rides above the card it belongs to */
  function reflow() {
    var wrap = document.getElementById('stageWrap');
    if (!wide.matches) {
      document.body.insertBefore(wrap, document.getElementById('beats'));
    }
  }
  if (wide.addEventListener) wide.addEventListener('change', reflow);
  reflow();

  /* prebuild the first few so the opening is instant */
  beats.slice(0, 3).forEach(function (b) { sceneFor(b.getAttribute('data-s')); });
  pick();

  /* ---------- animation pump ---------- */
  if (!reduce) {
    var t0 = performance.now();
    (function pump(t) {
      if (ticking && ticking.tick) { try { ticking.tick((t - t0) / 1000); } catch (e) {} }
      requestAnimationFrame(pump);
    })(t0);
  }
})();
