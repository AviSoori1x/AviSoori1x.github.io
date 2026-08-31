/* Scroll engine. Each .beat names a scene in data-s; window.SCENES builds it
   once into the sticky art panel, lazily, then it is only shown or hidden. */
(function () {
  var SS = window.RN || {};
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  var api = { RN: SS, SS: SS, KIT: window.KIT, K3: window.K3, reduce: reduce };
  window.RN_API = api;

  var art = document.getElementById('art');
  var beats = [].slice.call(document.querySelectorAll('.beat'));
  var rail = document.getElementById('rail');
  var actlab = document.getElementById('actlab');
  var SECTION = ['', 'The problem', 'How it moves', 'How it learns', 'Does it hold up'];
  var built = {}, current = null, ticking = null;

  function make(id) {
    if (built[id]) return built[id];
    var host = document.createElement('div');
    var fn = (window.SCENES || {})[id];
    var handle = null;
    if (typeof fn === 'function') {
      try { handle = fn(host, api) || null; }
      catch (err) {
        host.innerHTML = '';
        if (window.console) console.error('scene ' + id, err);
      }
    }
    // scenes must never be able to drop the class the stage positions with
    host.classList.add('scene');
    host.setAttribute('data-scene', id);
    art.appendChild(host);
    // stagger the entrance so the figure assembles rather than snapping in
    requestAnimationFrame(function () {
      var sv = host.querySelector('svg:not(.k3ov)');
      if (!sv) return;
      var kids = sv.children, step = Math.min(26, 620 / Math.max(1, kids.length));
      for (var i = 0; i < kids.length; i++) {
        kids[i].style.animationDelay = (i * step).toFixed(0) + 'ms';
      }
    });
    built[id] = { host: host, handle: handle };
    return built[id];
  }

  function show(beat) {
    if (!beat || beat === current) return;
    var next = make(beat.getAttribute('data-s'));
    Object.keys(built).forEach(function (k) {
      built[k].host.classList.toggle('on', built[k] === next);
    });
    if (ticking && ticking.stop) { try { ticking.stop(); } catch (e) {} }
    ticking = (next.handle && next.handle.tick) ? next.handle : null;
    if (ticking && ticking.start && !reduce) { try { ticking.start(); } catch (e) {} }

    var i = beats.indexOf(beat) + 1;
    actlab.textContent = SECTION[+(beat.getAttribute('data-act') || 1)] +
      ' · ' + (beat.getAttribute('data-sec') || '') + ' · ' + i + '/' + beats.length;
    current = beat;
  }

  var wide = matchMedia('(min-width: 1080.01px)');

  // Track the CARD, not the 118svh beat container. The container is taller than
  // the viewport, so its midpoint drifts away from the card you are reading and
  // the figure changed late, which read as the two columns being out of step.
  var cards = beats.map(function (b) { return b.querySelector('.stack') || b; });

  function pick() {
    var band = innerHeight * (wide.matches ? 0.5 : 0.3);
    var best = null, bd = Infinity;
    for (var i = 0; i < beats.length; i++) {
      var r = cards[i].getBoundingClientRect();
      var d = Math.abs((r.top + r.height / 2) - band);
      if (d < bd) { bd = d; best = beats[i]; }
    }
    if (best) show(best);
    var doc = document.documentElement;
    var max = doc.scrollHeight - doc.clientHeight;
    rail.style.height = (max > 0 ? (doc.scrollTop / max) * 100 : 0).toFixed(2) + '%';
  }

  var raf = 0;
  addEventListener('scroll', function () {
    if (raf) return;
    raf = requestAnimationFrame(function () { raf = 0; pick(); });
  }, { passive: true });
  addEventListener('resize', pick);

  beats.slice(0, 2).forEach(function (b) { make(b.getAttribute('data-s')); });
  pick();

  if (!reduce) {
    var t0 = performance.now();
    (function pump(t) {
      if (ticking && ticking.tick) { try { ticking.tick((t - t0) / 1000); } catch (e) {} }
      requestAnimationFrame(pump);
    })(t0);
  }
})();
