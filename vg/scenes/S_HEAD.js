/* S_HEAD: the output head. Two logits in, one score out. Real arithmetic. */
window.SCENES = window.SCENES || {};

window.SCENES.S_HEAD = function (root, api) {
  var el = api.el;

  var PRESETS = [
    { name: 'confident safe', yes: -3.2, no: 2.6 },
    { name: 'borderline', yes: 0.35, no: 0.15 },
    { name: 'confident unsafe', yes: 3.4, no: -2.1 }
  ];
  var TAU = 0.5;
  var st = { yes: 3.4, no: -2.1, tau: TAU };

  var wrap = el('div', 'sc-s_head');
  root.appendChild(wrap);

  wrap.appendChild(el('div', 'sc-s_head-lab', 'unembed to two token ids only'));

  /* ---- score readout ---- */
  var score = el('div', 'sc-s_head-score');
  var big = el('div', 'sc-s_head-big', '0.000');
  var verdict = el('div', 'sc-s_head-verdict', 'not flagged');
  score.appendChild(big);
  score.appendChild(verdict);
  wrap.appendChild(score);

  /* ---- 0..1 track ---- */
  var track = el('div', 'sc-s_head-track');
  var fillSafe = el('i', 'sc-s_head-fill');
  var tauTick = el('i', 'sc-s_head-tau');
  var marker = el('i', 'sc-s_head-mark');
  track.appendChild(fillSafe);
  track.appendChild(tauTick);
  track.appendChild(marker);
  wrap.appendChild(track);

  var ends = el('div', 'sc-s_head-ends');
  ends.appendChild(el('span', null, '0.0 no'));
  ends.appendChild(el('span', 'sc-s_head-tauread', 'tau 0.50'));
  ends.appendChild(el('span', null, 'yes 1.0'));
  wrap.appendChild(ends);

  /* ---- sliders ---- */
  function slider(key, label, min, max, step) {
    var row = el('div', 'sc-s_head-row');
    var name = el('label', 'sc-s_head-name', label);
    var id = 'S_HEAD-' + key;
    name.setAttribute('for', id);
    var input = el('input');
    input.type = 'range';
    input.id = id;
    input.min = min; input.max = max; input.step = step;
    input.value = st[key];
    input.className = 'sc-s_head-range';
    input.setAttribute('aria-label', label);
    var read = el('span', 'sc-s_head-read');
    input.addEventListener('input', function () {
      st[key] = parseFloat(input.value);
      clearPreset();
      draw();
    });
    row.appendChild(name); row.appendChild(input); row.appendChild(read);
    wrap.appendChild(row);
    return { input: input, read: read };
  }

  var sYes = slider('yes', 'z_yes', -8, 8, 0.1);
  var sNo = slider('no', 'z_no', -8, 8, 0.1);
  var sTau = slider('tau', 'threshold', 0.05, 0.95, 0.01);

  /* ---- presets ---- */
  var pbar = el('div', 'sc-s_head-presets');
  var pbtns = PRESETS.map(function (p, i) {
    var b = el('button', 'sc-s_head-pre', p.name);
    b.type = 'button';
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', function () {
      st.yes = p.yes; st.no = p.no;
      sYes.input.value = p.yes; sNo.input.value = p.no;
      pbtns.forEach(function (o, oi) { o.setAttribute('aria-pressed', oi === i ? 'true' : 'false'); });
      draw();
    });
    pbar.appendChild(b);
    return b;
  });
  wrap.appendChild(pbar);
  function clearPreset() {
    pbtns.forEach(function (o) { o.setAttribute('aria-pressed', 'false'); });
  }

  /* ---- the arithmetic, written out ---- */
  var maths = el('div', 'sc-s_head-maths');
  wrap.appendChild(maths);

  var gapnote = el('div', 'sc-s_head-note');
  wrap.appendChild(gapnote);

  function draw() {
    var ey = Math.exp(st.yes), en = Math.exp(st.no);
    var s = ey / (ey + en);
    var flagged = s > st.tau;

    big.textContent = s.toFixed(3);
    big.classList.toggle('is-on', flagged);
    verdict.textContent = flagged ? 'flagged' : 'not flagged';
    verdict.classList.toggle('is-on', flagged);

    marker.style.left = (s * 100) + '%';
    marker.classList.toggle('is-on', flagged);
    tauTick.style.left = (st.tau * 100) + '%';
    fillSafe.style.width = (st.tau * 100) + '%';

    sYes.read.textContent = (st.yes >= 0 ? '+' : '') + st.yes.toFixed(1);
    sNo.read.textContent = (st.no >= 0 ? '+' : '') + st.no.toFixed(1);
    sTau.read.textContent = st.tau.toFixed(2);
    wrap.querySelector('.sc-s_head-tauread').textContent = 'tau ' + st.tau.toFixed(2);

    maths.innerHTML =
      '<div class="sc-s_head-mline">s = exp(z_yes) / ( exp(z_yes) + exp(z_no) )</div>' +
      '<div class="sc-s_head-mline sc-s_head-dim">&nbsp;&nbsp;= exp(' + st.yes.toFixed(1) +
        ') / ( exp(' + st.yes.toFixed(1) + ') + exp(' + st.no.toFixed(1) + ') )</div>' +
      '<div class="sc-s_head-mline sc-s_head-dim">&nbsp;&nbsp;= ' + ey.toFixed(2) + ' / ' +
        (ey + en).toFixed(2) + '</div>' +
      '<div class="sc-s_head-mline">&nbsp;&nbsp;= <b>' + s.toFixed(3) + '</b></div>';

    var gap = st.yes - st.no;
    gapnote.textContent = 'Only the gap decides it. z_yes minus z_no is ' +
      (gap >= 0 ? '+' : '') + gap.toFixed(1) + '. Slide both sliders up together and the score ' +
      'does not move.';
  }

  draw();
};
