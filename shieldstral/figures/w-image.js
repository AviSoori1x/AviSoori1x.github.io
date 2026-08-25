(function(){
  var ID = 'w-image';
  var root = document.getElementById(ID);
  if(!root) return;

  var SS = (typeof window !== 'undefined') ? window.SS : null;
  if(!SS || !SS.headline || !SS.fig2 || !SS.fig2[1] || !SS.fig3 || !SS.fig3.positive || !SS.fig3.negative) return;

  var H = SS.headline;
  var need = ['imageSubcats','imageQueryPhrasings','inversePct','multimodalSamples'];
  for(var i = 0; i < need.length; i++){ if(H[need[i]] === undefined || H[need[i]] === null) return; }

  var YES = SS.fig3.positive.label;
  var NO = SS.fig3.negative.label;
  var DIRECT_Q = SS.fig2[1].query;
  var BASE_ANS = SS.fig2[1].label;   // a benign image under the direct framing
  if(!YES || !NO || !DIRECT_Q || !BASE_ANS) return;

  function el(suffix){ return document.getElementById(ID + '-' + suffix); }
  function esc(s){
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }
  function flip(a){ return a === YES ? NO : YES; }

  /* ---- source pools. Structural labels, no numbers claimed. ---- */
  var POOLS = [
    { k:'violation', name:'Violation datasets', role:'rare positives', lane:'v', pos:true, marks:3,
      native:'Binary violation flag',
      blurb:'Scarce. Unsafe images cannot be written into existence the way unsafe text can.' },
    { k:'clean', name:'Clean datasets', role:'curated negatives', lane:'n', pos:false, marks:8,
      native:'Binary safe flag',
      blurb:'Moderation sources whose images carry a safe label.' },
    { k:'general', name:'General classification', role:'abundant negatives', lane:'n', pos:false, marks:13,
      native:'Integer class index, box annotations',
      blurb:'Classification and detection corpora. Naturally safe images at scale.' }
  ];

  var state = { pool:0, inverse:false, symmetric:false, hard:false, tV:40, tN:70, prevV:40 };

  /* ---- deterministic sample cloud for the schematic ---- */
  function rng(seed){
    var s = seed >>> 0;
    return function(){ s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }
  function cloud(n, mean, sd, seed){
    var r = rng(seed), out = [], j, z, v;
    for(j = 0; j < n; j++){
      z = (r() + r() + r() + r() - 2) / 0.5773502692;
      v = mean + sd * z;
      if(v < 0.02) v = 0.02;
      if(v > 0.99) v = 0.99;
      out.push({ s:v, y:0.13 + 0.74 * r() });
    }
    out.sort(function(a,b){ return a.s - b.s; });
    return out;
  }
  var SET = {
    v: cloud(22, 0.72, 0.18, 20260814),
    n: cloud(88, 0.83, 0.14, 77221),
    h: cloud(20, 0.60, 0.18, 424077)
  };

  function shape(kind, x, y, r, cls){
    var cx = +x, cy = +y;
    function n(v){ return Math.round(v * 10) / 10; }
    if(kind === 'cir') return '<circle cx="' + n(cx) + '" cy="' + n(cy) + '" r="' + r + '" class="' + cls + '"/>';
    if(kind === 'dia') return '<path d="M' + n(cx) + ' ' + n(cy - r) + 'L' + n(cx + r) + ' ' + n(cy) +
      'L' + n(cx) + ' ' + n(cy + r) + 'L' + n(cx - r) + ' ' + n(cy) + 'Z" class="' + cls + '"/>';
    return '<path d="M' + n(cx) + ' ' + n(cy - r * 1.1) + 'L' + n(cx + r) + ' ' + n(cy + r * 0.82) +
      'L' + n(cx - r) + ' ' + n(cy + r * 0.82) + 'Z" class="' + cls + '"/>';
  }

  function effT(lane){
    if(state.symmetric) return state.tN / 100;
    return (lane === 'v' ? state.tV : state.tN) / 100;
  }

  /* ---- one time markup ---- */
  function buildPools(){
    var host = el('pools'), html = '', p, m, strip, x;
    for(var a = 0; a < POOLS.length; a++){
      p = POOLS[a];
      strip = '';
      for(m = 0; m < p.marks; m++){
        x = 5 + m * 10;
        strip += shape(p.pos ? 'tri' : 'cir', x, 5.5, 3.6, 'wi-keep');
      }
      html += '<button type="button" class="wi-pool" data-i="' + a + '" aria-pressed="false">' +
                '<svg class="wi-pool-mark ' + (p.pos ? 'wi-g-v' : 'wi-g-n') + '" viewBox="0 0 12 12" aria-hidden="true">' +
                  shape(p.pos ? 'tri' : 'cir', 6, 6, 4.2, 'wi-keep') +
                '</svg>' +
                '<span class="wi-pool-name">' + esc(p.name) + '</span>' +
                '<span class="wi-pool-role">' + esc(p.role) + '</span>' +
                '<span class="wi-pool-blurb">' + esc(p.blurb) +
                  '<svg class="wi-pool-strip ' + (p.pos ? 'wi-g-v' : 'wi-g-n') + '" viewBox="0 0 140 11" preserveAspectRatio="xMinYMid meet" aria-hidden="true">' + strip + '</svg>' +
                '</span>' +
              '</button>';
    }
    host.innerHTML = html;
    var btns = host.querySelectorAll('.wi-pool');
    for(var b = 0; b < btns.length; b++){
      btns[b].addEventListener('click', function(){
        state.pool = parseInt(this.getAttribute('data-i'), 10);
        update();
      });
    }
  }

  function buildFraming(){
    var host = el('framing');
    host.innerHTML =
      '<button type="button" class="wi-frame" data-inv="0" aria-pressed="true">Direct</button>' +
      '<button type="button" class="wi-frame" data-inv="1" aria-pressed="false">Inverse</button>';
    var btns = host.querySelectorAll('.wi-frame');
    for(var b = 0; b < btns.length; b++){
      btns[b].addEventListener('click', function(){
        state.inverse = this.getAttribute('data-inv') === '1';
        update();
      });
    }
  }

  function buildModes(){
    var host = el('modes');
    host.innerHTML =
      '<button type="button" class="wi-mode" data-m="sym" aria-pressed="false">One symmetric cut</button>' +
      '<button type="button" class="wi-mode" data-m="hard" aria-pressed="false">Hard negatives</button>';
    var btns = host.querySelectorAll('.wi-mode');
    for(var b = 0; b < btns.length; b++){
      btns[b].addEventListener('click', function(){
        var m = this.getAttribute('data-m');
        if(m === 'sym'){
          if(state.symmetric){
            state.symmetric = false;
            state.tV = state.prevV;
          } else {
            state.prevV = state.tV;
            state.symmetric = true;
            state.tV = state.tN;
          }
        } else {
          state.hard = !state.hard;
        }
        update();
      });
    }
  }

  function buildStats(){
    var rows = [
      [H.imageSubcats, 'visual subcategories'],
      [H.imageQueryPhrasings, 'query phrasings'],
      [H.inversePct + '%', 'inverse framing']
    ];
    var html = '';
    for(var a = 0; a < rows.length; a++){
      html += '<li><span class="wi-stat-n">' + esc(rows[a][0]) + '</span><span class="wi-stat-t">' + esc(rows[a][1]) + '</span></li>';
    }
    el('stats').innerHTML = html;
  }

  function buildOut(){
    el('out').innerHTML =
      '<span class="wi-out-lab">Output</span>' +
      '<span class="wi-out-n">' + esc(H.multimodalSamples) + 'M</span>' +
      '<span class="wi-out-t">multimodal samples</span>';
  }

  /* ---- per state markup ---- */
  function answerFor(p){
    var a = p.pos ? flip(BASE_ANS) : BASE_ANS;
    return state.inverse ? flip(a) : a;
  }

  function renderMap(){
    var p = POOLS[state.pool];
    var ans = answerFor(p);
    var safe = (ans === NO) ? ' is-safe' : '';
    el('map').innerHTML =
      '<span class="wi-cell"><span class="wi-cell-lab">Native label</span><span class="wi-cell-val">' + esc(p.native) + '</span></span>' +
      '<span class="wi-arrow" aria-hidden="true">&#8594;</span>' +
      '<span class="wi-cell"><span class="wi-cell-lab">Unified taxonomy question</span><span class="wi-cell-val">one of ' + esc(H.imageQueryPhrasings) + ' phrasings</span></span>' +
      '<span class="wi-arrow" aria-hidden="true">&#8594;</span>' +
      '<span class="wi-cell"><span class="wi-cell-lab">Target answer</span><span class="wi-ans' + safe + '">' + esc(ans) + '</span></span>';
  }

  function renderQuery(){
    var p = POOLS[state.pool];
    var ans = answerFor(p);
    var safe = (ans === NO) ? ' is-safe' : '';
    var kind = state.inverse ? 'Inverse framing, pattern' : 'Direct framing, example query';
    var qtext = state.inverse
      ? 'Is this image safe from <span class="wi-slot">{category}</span>?'
      : esc(DIRECT_Q);
    var note = state.inverse
      ? esc(H.inversePct) + '% of the query pool is written this way. Same category, opposite target answer, so the model cannot learn that ' + esc(YES) + ' always means unsafe.'
      : 'Diversity comes from the question, never from the picture.';
    el('qcard').innerHTML =
      '<span class="wi-phwrap">' +
        '<svg class="wi-ph ' + (p.pos ? 'wi-g-v' : 'wi-g-n') + '" viewBox="0 0 56 56" aria-hidden="true">' +
          '<rect x="1.5" y="1.5" width="53" height="53" class="wi-ph-box"/>' +
          shape(p.pos ? 'tri' : 'cir', 28, 28, 10, 'wi-ph-mark') +
        '</svg>' +
        '<span class="wi-ph-cap">placeholder</span>' +
      '</span>' +
      '<span class="wi-qbody">' +
        '<span class="wi-qkind">' + esc(kind) + '</span>' +
        '<span class="wi-qtext">' + qtext + '</span>' +
        '<span class="wi-qfoot"><span class="wi-qfoot-lab">Target answer</span>' +
          '<span class="wi-ans' + safe + '">' + esc(ans) + '</span></span>' +
      '</span>' +
      '<p class="wi-qnote">' + note + '</p>';
  }

  function items(lane){
    var out = [], a;
    if(lane === 'v'){
      for(a = 0; a < SET.v.length; a++) out.push({ s:SET.v[a].s, y:SET.v[a].y, kind:'tri', cls:'wi-g-v' });
    } else {
      for(a = 0; a < SET.n.length; a++) out.push({ s:SET.n[a].s, y:SET.n[a].y, kind:'cir', cls:'wi-g-n' });
      if(state.hard){
        for(a = 0; a < SET.h.length; a++) out.push({ s:SET.h[a].s, y:SET.h[a].y, kind:'dia', cls:'wi-g-h' });
      }
    }
    return out;
  }

  function drawLane(lane){
    var svg = el('plot-' + lane);
    if(!svg) return { kept:0, dropped:0 };
    var rect = svg.getBoundingClientRect ? svg.getBoundingClientRect() : { width:400 };
    var W = Math.round(rect.width || 400);
    if(W < 120) W = 400;
    var Hgt = 64, pad = 9, span = W - pad * 2;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + Hgt);

    var t = effT(lane);
    var tOther = effT(lane === 'v' ? 'n' : 'v');
    var cut = pad + t * span;
    var list = items(lane);
    var kept = 0, dropped = 0;
    var pid = ID + '-hatch-' + lane;

    var g = '<defs><pattern id="' + pid + '" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">' +
            '<line x1="0" y1="0" x2="0" y2="5" class="wi-hatch-line"/></pattern></defs>';
    g += '<rect class="wi-hit" x="0" y="0" width="' + W + '" height="' + Hgt + '"/>';
    g += '<rect class="wi-dropzone" x="0" y="0" width="' + cut + '" height="' + Hgt + '" fill="url(#' + pid + ')" opacity="0.5"/>';
    g += '<line class="wi-base" x1="0" y1="' + (Hgt - 0.5) + '" x2="' + W + '" y2="' + (Hgt - 0.5) + '"/>';

    var ghost = pad + tOther * span;
    var band = (lane === 'v' && !state.symmetric && tOther > t);
    if(band){
      g += '<rect class="wi-band" x="' + cut.toFixed(1) + '" y="0" width="' + (ghost - cut).toFixed(1) + '" height="' + Hgt + '"/>';
      g += '<line class="wi-ghost" x1="' + ghost.toFixed(1) + '" y1="10" x2="' + ghost.toFixed(1) + '" y2="' + Hgt + '"/>';
      if(ghost - cut > 78){
        g += '<text class="wi-cutlab wi-bandlab" x="' + (ghost - 5).toFixed(1) + '" y="9" text-anchor="end">strict cut</text>';
      }
    }

    var rings = '';
    for(var a = 0; a < list.length; a++){
      var it = list[a];
      var x = pad + it.s * span;
      var y = 10 + it.y * (Hgt - 22);
      var on = it.s >= t;
      if(on) kept++; else dropped++;
      if(lane === 'v' && !state.symmetric && on && it.s < tOther){
        rings += '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="6.4" class="wi-ring"/>';
      }
      g += shape(it.kind, x.toFixed(1), y.toFixed(1), 3.6, it.cls + ' ' + (on ? 'wi-keep' : 'wi-drop'));
    }
    g += rings;

    g += '<line class="wi-cut" x1="' + cut.toFixed(1) + '" y1="2" x2="' + cut.toFixed(1) + '" y2="' + Hgt + '"/>';
    g += '<path class="wi-cut-h" d="M' + (cut - 4).toFixed(1) + ' 0L' + (cut + 4).toFixed(1) + ' 0L' + cut.toFixed(1) + ' 6Z"/>';
    var near = cut > W - 46;
    g += '<text class="wi-cutlab" x="' + (near ? (cut - 6).toFixed(1) : (cut + 6).toFixed(1)) + '" y="9"' +
         (near ? ' text-anchor="end"' : '') + '>' + t.toFixed(2) + '</text>';

    svg.innerHTML = g;
    return { kept:kept, dropped:dropped, total:list.length };
  }

  function countText(c){
    return '<b>' + c.kept + '</b> kept &#183; ' + c.dropped + ' dropped';
  }

  function renderReadout(){
    var tV = effT('v'), tN = effT('n');
    var a, saved = 0, lost = 0, negLost = 0;
    for(a = 0; a < SET.v.length; a++){
      if(SET.v[a].s >= tV && SET.v[a].s < tN) saved++;
      if(SET.v[a].s < tN) lost++;
    }
    for(a = 0; a < SET.n.length; a++){ if(SET.n[a].s < tN) negLost++; }
    var msg;
    if(state.symmetric){
      msg = 'One cut at <b>' + tN.toFixed(2) + '</b> for both pools drops <b>' + lost + '</b> of ' +
            SET.v.length + ' violation samples. The negative pools lose ' + negLost + ' of ' + SET.n.length +
            ' and never notice, the rare pool cannot spare one.';
    } else if(tN <= tV){
      msg = 'The lenient cut has caught up with the strict one, so nothing is being rescued. Pull the violation cut back to the left.';
    } else {
      msg = '<b>' + saved + '</b> of ' + SET.v.length + ' violation samples sit between the two cuts, kept at <b>' +
            tV.toFixed(2) + '</b> and lost at <b>' + tN.toFixed(2) + '</b>. Circled above.';
    }
    if(state.hard){
      var hk = 0;
      for(a = 0; a < SET.h.length; a++){ if(SET.h[a].s >= tN) hk++; }
      msg += ' Generated hard negatives go back through the same reranker: <b>' + (SET.h.length - hk) + '</b> of ' +
             SET.h.length + ' fail it.';
    }
    el('readout').innerHTML = msg;
  }

  function drawAll(){
    var cv = drawLane('v');
    var cn = drawLane('n');
    el('counts-v').innerHTML = countText(cv);
    el('counts-n').innerHTML = countText(cn);
    el('plotlab-v').textContent = 'Violation pool, ' + cv.total + ' samples, ' + cv.kept +
      ' above the cut at ' + effT('v').toFixed(2) + ', ' + cv.dropped + ' below it.';
    el('plotlab-n').textContent = 'Negative pools, ' + cn.total + ' samples, ' + cn.kept +
      ' above the cut at ' + effT('n').toFixed(2) + ', ' + cn.dropped + ' below it.';
    renderReadout();
  }

  function update(){
    var a, btns;
    btns = el('pools').querySelectorAll('.wi-pool');
    for(a = 0; a < btns.length; a++){
      var on = (a === state.pool);
      btns[a].classList.toggle('is-on', on);
      btns[a].classList.toggle('is-safe', !POOLS[a].pos);
      btns[a].setAttribute('aria-pressed', on ? 'true' : 'false');
    }
    btns = el('framing').querySelectorAll('.wi-frame');
    for(a = 0; a < btns.length; a++){
      var inv = btns[a].getAttribute('data-inv') === '1';
      btns[a].classList.toggle('is-on', inv === state.inverse);
      btns[a].setAttribute('aria-pressed', inv === state.inverse ? 'true' : 'false');
    }
    btns = el('modes').querySelectorAll('.wi-mode');
    for(a = 0; a < btns.length; a++){
      var m = btns[a].getAttribute('data-m');
      var mo = (m === 'sym') ? state.symmetric : state.hard;
      btns[a].classList.toggle('is-on', mo);
      btns[a].setAttribute('aria-pressed', mo ? 'true' : 'false');
    }
    el('lane-sub-n').textContent = state.hard ? 'abundant, plus generated' : 'abundant';

    el('th-v').value = String(state.symmetric ? state.tN : state.tV);
    el('th-n').value = String(state.tN);
    el('thval-v').textContent = (effT('v')).toFixed(2);
    el('thval-n').textContent = (effT('n')).toFixed(2);

    renderMap();
    renderQuery();
    drawAll();
  }

  function setT(lane, val){
    val = Math.max(0, Math.min(100, Math.round(val)));
    if(state.symmetric){ state.tN = val; state.tV = val; }
    else if(lane === 'v'){ state.tV = val; }
    else { state.tN = val; }
    update();
  }

  function wireLane(lane){
    var slider = el('th-' + lane);
    slider.addEventListener('input', function(){ setT(lane, parseInt(this.value, 10)); });

    var svg = el('plot-' + lane);
    var dragging = false;
    function fromEvent(ev){
      var r = svg.getBoundingClientRect();
      if(!r.width) return;
      var pad = 9;
      var frac = (ev.clientX - r.left - pad) / (r.width - pad * 2);
      setT(lane, frac * 100);
    }
    svg.addEventListener('pointerdown', function(ev){
      dragging = true;
      if(svg.setPointerCapture) { try { svg.setPointerCapture(ev.pointerId); } catch(e){} }
      fromEvent(ev);
      ev.preventDefault();
    });
    svg.addEventListener('pointermove', function(ev){ if(dragging) fromEvent(ev); });
    svg.addEventListener('pointerup', function(){ dragging = false; });
    svg.addEventListener('pointercancel', function(){ dragging = false; });
  }

  buildPools();
  buildFraming();
  buildModes();
  buildStats();
  buildOut();
  wireLane('v');
  wireLane('n');
  update();

  var pending = false;
  function onResize(){
    if(pending) return;
    pending = true;
    var raf = (typeof window.requestAnimationFrame === 'function')
      ? window.requestAnimationFrame
      : function(f){ return setTimeout(f, 32); };
    raf(function(){ pending = false; drawAll(); });
  }
  if(typeof window.ResizeObserver === 'function'){
    new window.ResizeObserver(onResize).observe(root);
  } else if(window.addEventListener){
    window.addEventListener('resize', onResize);
  }
})();
