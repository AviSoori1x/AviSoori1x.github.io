/* Host page behaviour: masthead, contents, progress, reveal. */
(function () {
  var SS = window.SS;
  if (!SS) return;

  /* ---------- masthead links ---------- */
  var links = [
    { href: SS.links.hf, label: 'Weights on Hugging Face' },
    { href: SS.links.paper, label: 'Technical report (arXiv)' },
    { href: SS.links.blog, label: 'Mistral announcement' }
  ];
  var lr = document.getElementById('masthead-links');
  if (lr) {
    lr.innerHTML = links.map(function (l) {
      return '<a href="' + l.href + '" rel="noopener">' + l.label +
             ' <span class="arw" aria-hidden="true">&#8599;</span></a>';
    }).join('');
  }

  /* ---------- headline stats, all read from the data file ---------- */
  var h = SS.headline;
  var stats = [
    { n: h.params, k: 'parameters' },
    { n: h.totalSamples + 'M', k: 'training samples' },
    { n: h.textF1 + '<em>%</em>', k: 'text safety F1, avg' },
    { n: h.multimodalF1 + '<em>%</em>', k: 'multimodal F1, avg' },
    { n: h.adaptabilityF1 + '<em>%</em>', k: 'policy adaptability F1' },
    { n: SS.languages.length, k: 'supported languages' }
  ];
  var sc = document.getElementById('masthead-stats');
  if (sc) {
    sc.innerHTML = stats.map(function (s) {
      return '<div><div class="n">' + s.n + '</div><div class="k">' + s.k + '</div></div>';
    }).join('');
  }

  /* ---------- contents, built from the sections actually present ---------- */
  var toc = document.getElementById('toc-list');
  if (toc) {
    var secs = document.querySelectorAll('.section[id]');
    var items = [];
    for (var i = 0; i < secs.length; i++) {
      var head = secs[i].querySelector('h2');
      if (!head) continue;
      var num = ('0' + (i + 1)).slice(-2);
      items.push('<li><a href="#' + secs[i].id + '"><span class="num">' + num +
                 '</span><span>' + head.textContent + '</span></a></li>');
    }
    toc.innerHTML = items.join('');
  }

  /* ---------- citation block ---------- */
  var cite = document.getElementById('cite-block');
  if (cite) {
    cite.textContent =
      'Shieldstral. Core contributors: ' + SS.coreContributors.join(', ') + '.\n' +
      'Technical report: ' + SS.links.paper + '\n' +
      'Model card: ' + SS.links.hf;
  }

  /* ---------- reading progress ---------- */
  var bar = document.getElementById('progress');
  if (bar) {
    var tick = function () {
      var d = document.documentElement;
      var max = d.scrollHeight - d.clientHeight;
      var pct = max > 0 ? (d.scrollTop / max) * 100 : 0;
      bar.style.width = pct.toFixed(2) + '%';
    };
    window.addEventListener('scroll', tick, { passive: true });
    window.addEventListener('resize', tick);
    tick();
  }

  /* ---------- reveal on scroll ---------- */
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var targets = document.querySelectorAll('.section, .fig, .toc');
  var i2;
  if (reduce || !('IntersectionObserver' in window)) {
    for (i2 = 0; i2 < targets.length; i2++) targets[i2].classList.add('in');
    return;
  }
  for (i2 = 0; i2 < targets.length; i2++) targets[i2].classList.add('rv');
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        e.target.classList.add('in');
        io.unobserve(e.target);
      }
    });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.02 });
  for (i2 = 0; i2 < targets.length; i2++) io.observe(targets[i2]);
})();
