window.SCENES = window.SCENES || {};
window.SCENES['S_FROZEN'] = function (root, api) {
  var SS = api.SS || {};
  var el = api.el;
  var reduce = !!api.reduce;

  root.classList.add('sc-s_frozen');

  /* The four rules are written for this figure. They are not model output and not
     taken from the report, which is why the caption says illustrative. */
  var RULES = [
    'no instructions for synthesising controlled substances',
    'nothing that could distress someone in crisis',
    'allow exploit code, this is a security research tool',
    'no spoilers for anything released this year'
  ];

  /* everything factual is read out of window.SS at runtime */
  var tax = Array.isArray(SS.evalTaxonomy) ? SS.evalTaxonomy : [];
  var cats = [];
  for (var t = 0; t < tax.length; t++) {
    if (tax[t] && tax[t].name) cats.push(tax[t].name);
  }
  var bl = Array.isArray(SS.baselines) ? SS.baselines : [];
  var nFixed = 0;
  for (var b = 0; b < bl.length; b++) if (bl[b] && bl[b].adaptive === false) nFixed++;

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  var chipHtml = '';
  for (var c = 0; c < cats.length; c++) {
    chipHtml += '<span class="chp"><i></i>' + esc(cats[c]) + '</span>';
  }

  var leftSub = 'Label vocabulary compiled in at training time.';
  if (bl.length && nFixed) {
    leftSub += ' ' + nFixed + ' of the ' + bl.length +
      ' systems compared in the report are not policy adaptive.';
  }

  /* the size comes from the comparison table, not from a hardcoded string */
  var ours = 'Shieldstral';
  for (var o = 0; o < bl.length; o++) {
    if (bl[o] && /Shieldstral/.test(bl[o].model) && bl[o].size) ours += ' ' + bl[o].size;
  }

  var catLine = cats.length
    ? cats.length + ' frozen categories'
    : 'frozen category vocabulary';

  var wrap = el('div', 'wrap');
  wrap.innerHTML =
    '<div class="hd">' +
      '<h4>One rule. Two machines.</h4>' +
      '<p class="ill">Illustrative. The four rules are written for this figure, not model output.' +
      (cats.length ? '<span class="illb"> The chips are the ' + cats.length +
        ' evaluation super classes from the report, standing in for a fixed vocabulary.</span>' : '') +
      '</p>' +
    '</div>' +

    '<div class="rulebox">' +
      '<div class="rlab"><span>pick a moderation rule and send it</span>' +
        '<button type="button" class="clr" aria-label="Empty the policy slot">empty slot</button>' +
      '</div>' +
      '<div class="rules" role="group" aria-label="Moderation rules"></div>' +
    '</div>' +

    '<div class="mach">' +

      '<section class="m mfix" aria-label="Fixed taxonomy guardrail">' +
        '<header><span class="nm">Fixed taxonomy guardrail</span>' +
          '<span class="tag">sealed</span></header>' +
        '<p class="sub">' + esc(leftSub) + '</p>' +
        '<div class="slot dead">' +
          '<span class="mouth"><i class="weld"></i></span>' +
          '<span class="stxt">input slot welded shut</span>' +
        '</div>' +
        '<div class="frost">' +
          '<div class="flab">' + esc(catLine) + '</div>' +
          '<div class="chips">' + chipHtml + '</div>' +
        '</div>' +
        '<footer>' +
          '<div class="big"><span class="dig">0</span>' +
            '<span class="dlab">rules<br>accepted</span></div>' +
          '<p class="dead2">Send it whatever you like. Nothing in this box moves.</p>' +
        '</footer>' +
      '</section>' +

      '<section class="m mopen" aria-label="Shieldstral">' +
        '<header><span class="nm">' + esc(ours) + '</span>' +
          '<span class="tag live">open</span></header>' +
        '<p class="sub">Policy arrives as plain text at inference. The weights never change.</p>' +
        '<div class="slot live">' +
          '<span class="mouth"><i class="caret"></i></span>' +
          '<span class="stxt">policy slot, empty</span>' +
        '</div>' +
        '<div class="win" id="S_FROZEN-win" aria-live="polite">' +
          '<div class="wlab">query in force</div>' +
          '<div class="wtxt empty">no rule loaded</div>' +
          '<p class="whint">the caller supplies this field at request time</p>' +
        '</div>' +
        '<footer>' +
          '<div class="big"><span class="dig">0</span>' +
            '<span class="dlab">rules<br>accepted</span></div>' +
          '<p class="dead2">Same weights. Different question, every call.</p>' +
        '</footer>' +
      '</section>' +

    '</div>';

  root.appendChild(wrap);

  /* ---------- rule buttons ---------- */
  var rulesBox = wrap.querySelector('.rules');
  var btns = [];
  for (var i = 0; i < RULES.length; i++) {
    var btn = el('button', 'rule');
    btn.type = 'button';
    btn.innerHTML = '<span class="state">send</span><span class="txt">' +
      esc(RULES[i]) + '</span>';
    btn.setAttribute('data-i', String(i));
    rulesBox.appendChild(btn);
    btns.push(btn);
  }

  var openSlot = wrap.querySelector('.mopen .slot');
  var openStxt = wrap.querySelector('.mopen .stxt');
  var openSec = wrap.querySelector('.mopen');
  var win = wrap.querySelector('.win');
  var wtxt = wrap.querySelector('.win .wtxt');
  var whint = wrap.querySelector('.win .whint');
  var openDig = wrap.querySelector('.mopen .dig');
  var deadSlot = wrap.querySelector('.mfix .slot');
  var clr = wrap.querySelector('.clr');

  var accepted = 0;
  var active = -1;
  var timers = [];

  function later(fn, ms) {
    var id = setTimeout(fn, ms);
    timers.push(id);
    return id;
  }
  function clearTimers() {
    for (var k = 0; k < timers.length; k++) clearTimeout(timers[k]);
    timers = [];
  }

  function paintButtons() {
    for (var j = 0; j < btns.length; j++) {
      var on = (j === active);
      btns[j].classList.toggle('on', on);
      btns[j].setAttribute('aria-pressed', on ? 'true' : 'false');
      btns[j].querySelector('.state').textContent = on ? 'in force' : 'send';
    }
  }

  function commit(idx) {
    active = idx;
    accepted++;
    openDig.textContent = String(accepted);
    openSec.classList.add('loaded');
    openSlot.classList.add('on');
    openStxt.textContent = 'policy slot, in force';
    wtxt.classList.remove('empty');
    wtxt.textContent = RULES[idx];
    whint.textContent = 'read as the query field. No weight was touched.';
    win.classList.add('flash');
    later(function () { win.classList.remove('flash'); }, 620);
    paintButtons();
  }

  function empty() {
    clearTimers();
    active = -1;
    openSec.classList.remove('loaded');
    openSlot.classList.remove('on');
    openStxt.textContent = 'policy slot, empty';
    wtxt.classList.add('empty');
    wtxt.textContent = 'no rule loaded';
    whint.textContent = 'the caller supplies this field at request time';
    paintButtons();
  }

  /* ---------- the card flying into a slot ---------- */
  function launch(from, to, text, kind, land) {
    var hr = wrap.getBoundingClientRect();
    var a = from.getBoundingClientRect();
    var z = to.getBoundingClientRect();
    if (!hr.width || !a.width) { if (land) land(); return; }

    var w = Math.min(a.width, 300);
    var f = el('div', 'fly ' + kind, text);
    f.style.width = w + 'px';
    f.style.left = (a.left - hr.left) + 'px';
    f.style.top = (a.top - hr.top) + 'px';
    f.style.height = a.height + 'px';
    wrap.appendChild(f);

    var dx = (z.left + z.width / 2) - (a.left + w / 2);
    var dy = (z.top + z.height / 2) - (a.top + a.height / 2);

    /* force layout so the transition has a start value */
    void f.offsetWidth;
    f.style.transform = 'translate(' + dx + 'px,' + dy + 'px) scale(.62)';
    f.style.opacity = (kind === 'ghost') ? '.5' : '1';

    later(function () {
      if (land) land();
      f.classList.add('gone');
      later(function () { if (f.parentNode) f.parentNode.removeChild(f); }, 320);
    }, 520);
  }

  function send(idx) {
    if (idx === active) return;
    clearTimers();
    var from = btns[idx];
    if (reduce) { commit(idx); return; }
    /* the dim copy travels to the sealed machine and simply stops at the weld */
    launch(from, deadSlot, RULES[idx], 'ghost', null);
    launch(from, openSlot, RULES[idx], 'real', function () { commit(idx); });
  }

  rulesBox.addEventListener('click', function (ev) {
    var t = ev.target;
    while (t && t !== rulesBox && !t.classList.contains('rule')) t = t.parentNode;
    if (!t || t === rulesBox) return;
    send(+t.getAttribute('data-i'));
  });
  clr.addEventListener('click', empty);

  paintButtons();

  /* The stage panel is narrower and shorter than the viewport, so measure the panel
     itself rather than trusting a media query. Squeeze when it is narrow, or when the
     roomy layout would not fit the height on offer. */
  var lastW = -1, lastH = -1;
  function fit() {
    var r = root.getBoundingClientRect();
    var w = Math.round(r.width), h = Math.round(r.height);
    if (!w) return;
    if (w === lastW && h === lastH) return;
    lastW = w; lastH = h;
    root.classList.remove('sq');
    var need = wrap.getBoundingClientRect().height;
    root.classList.toggle('sq', w < 640 || (h > 0 && need > h - 4));
  }
  var ro = null;
  if (window.ResizeObserver) {
    ro = new ResizeObserver(fit);
    ro.observe(root);
  } else {
    window.addEventListener('resize', fit);
  }
  fit();

  return {
    stop: function () { clearTimers(); }
  };
};
