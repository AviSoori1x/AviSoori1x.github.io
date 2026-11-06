/* S_SCORES: the five capability tables of section 5 (Tables 1 to 4 and 6), one
   row per benchmark. Every score is the table's own value from window.ML4.tables;
   the ranks are computed from those values at parse time, higher is better, and
   count only the models a row reports. Each row is a 0 to 100% strip: ML4 is the
   amber dot, the best other model the blue one, the rest grey. A model the paper
   gives no score for is simply absent from that row. Table 5, safety, has its own
   figure next. */
window.SCENES.S_SCORES = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, n = K.n, D = api.D;
  var M = D.ml4, NAMES = D.short || D.cols;
  var svg = K.board(host, { alt: 'Tables 1 to 4 and 6: every benchmark score for Mistral Large 4 and the six models it is compared with' });
  K.head(svg, 'Where ML4 leads, and where it trails', 'Tables 1 to 4 and 6 · scores in % · safety, Table 5, is next');

  var TAB = [], TN = [], ORD = ['', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th'];
  var WORD = ['', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh'];
  // short switch names, from each table's own title
  var SHORT = { 1: 'agents', 2: 'reasoning', 3: 'code', 4: 'cyber', 6: 'finance, law' };
  Object.keys(D.tables).forEach(function (k) {
    var t = D.tables[k];
    if (SHORT[t.n]) { TAB.push(t); TN.push(t.n + ' ' + SHORT[t.n]); }
  });
  if (TAB.length !== 5) {
    K.mono(svg, 0, 100, 'Table values are not available.', { size: 14, color: C.ink2 });
    return;
  }

  // a row's rank phrase: "4th of 7", "tied 1st of 7"
  function rankOf(r) {
    var k = r.rank[M], tie = r.vals.filter(function (v) { return v === r.vals[M]; }).length > 1;
    return { k: k, tie: tie, txt: (tie ? 'tied ' : '') + ORD[k] + ' of ' + r.n };
  }
  // the best score among the other models, the blue dot
  function rival(r) {
    var bi = -1;
    r.vals.forEach(function (v, i) {
      if (i !== M && v != null && (bi < 0 || v > r.vals[bi])) bi = i;
    });
    return bi;
  }
  function rowName(r) { return r.label + (r.src === 'internal' ? ', internal' : ''); }

  // the callout: group a table's rows by ML4's rank, in the paper's row order
  function summary(t) {
    var by = {}, order = [];
    t.rows.forEach(function (r) {
      var q = rankOf(r), key = q.k === r.n && !q.tie ? 'last' : (q.tie ? 'tied ' : '') + WORD[q.k];
      if (!by[key]) { by[key] = []; order.push(key); }
      by[key].push(r.label);
    });
    order.sort(function (a, b) {
      var ra = a === 'last' ? 9 : WORD.indexOf(a.replace('tied ', ''));
      var rb = b === 'last' ? 9 : WORD.indexOf(b.replace('tied ', ''));
      return ra - rb;
    });
    var parts = order.map(function (key) {
      var L = by[key], s = L.length > 1 ? L.slice(0, -1).join(', ') + ' and ' + L[L.length - 1] : L[0];
      return key + ' on ' + s;
    });
    var out = parts.join('; ');
    return 'Table ' + t.n + ', ' + t.title.toLowerCase().replace(/ evaluations$/, '') + '. ML4 is ' + out + '.';
  }

  var swG = CH.body(svg), body = CH.body(svg), pads = CH.body(svg);
  var X0 = 0, XW = 640, sx = function (v) { return X0 + v / 100 * XW; };
  var Y0 = 62, RH = 46;
  var out = null, ring = null, T = 0;

  // when ML4 and the blue dot nearly coincide, the amber would hide the blue, so the
  // pair is nudged apart vertically; x, the score, stays exact
  function lift(r, i, bi) {
    if (bi < 0 || Math.abs(sx(r.vals[M]) - sx(r.vals[bi])) >= 13) return 0;
    return i === M ? 6 : i === bi ? -6 : 0;
  }

  // read one model of one row into the readout and ring its dot
  function read(r, i, cy) {
    ring.setAttribute('cx', sx(r.vals[i]));
    ring.setAttribute('cy', cy + lift(r, i, rival(r)));
    ring.setAttribute('opacity', 1);
    out.set(NAMES[i] + ' · ' + r.label + ' · ' + r.raw[i] + ' · ' + ORD[r.rank[i]] + ' of ' + r.n);
  }

  function draw(ti) {
    T = ti;
    CH.clear(body);
    CH.clear(pads);
    var t = TAB[ti];
    // a light 0 / 50 / 100 scale over the strips
    [0, 50, 100].forEach(function (v) {
      body.appendChild(n('line', { x1: sx(v), x2: sx(v), y1: Y0 - 8, y2: Y0 + t.rows.length * RH - 14,
        stroke: v === 50 ? C.line2 : C.line, 'stroke-dasharray': v === 50 ? '2 4' : null }));
      K.mono(body, sx(v), Y0 - 16, v === 100 ? '100%' : String(v),
        { size: 12.5, anchor: v === 0 ? 'start' : v === 100 ? 'end' : 'middle', color: C.ink3 });
    });
    t.rows.forEach(function (r, j) {
      var y = Y0 + j * RH, cy = y + 18, q = rankOf(r), bi = rival(r);
      K.mono(body, 0, y + 2, rowName(r), { size: 12.5, color: C.ink2 });
      // a shared first place names how many share it; otherwise the rival's score
      var others = r.vals.filter(function (v, i) { return i !== M && v === r.vals[M]; }).length;
      var rt = q.k === 1 && q.tie ? '1st of ' + r.n + ', tied with ' + others + (others > 1 ? ' others' : ' other')
        : q.txt + (q.k === 1 ? ' · next ' : ' · best ') + r.raw[bi];
      K.mono(body, 640, y + 2, r.raw[M] + ' · ' + rt, { size: 12.5, anchor: 'end',
        color: C.ink, weight: 600 });
      body.appendChild(n('rect', { x: 0, y: cy - 1, width: 640, height: 2, rx: 1, fill: K.TINT.ink }));
      // grey context first, then the rival, then ML4 on top
      var ix = r.vals.map(function (v, i) { return i; }).filter(function (i) { return r.vals[i] != null; });
      ix.sort(function (a, b) { return (a === M ? 2 : a === bi ? 1 : 0) - (b === M ? 2 : b === bi ? 1 : 0); });
      ix.forEach(function (i) {
        var col = i === M ? C.amber : i === bi ? C.blue : CH.GREY;
        body.appendChild(n('circle', { cx: sx(r.vals[i]), cy: cy + lift(r, i, bi), r: i === M ? 6.5 : i === bi ? 5.5 : 4.2,
          fill: col, stroke: '#fff', 'stroke-width': 1.8 }));
      });
      // one hover pad per row reads the dot nearest the pointer (local helper:
      // the kit's hot target is one fixed shape, and close dots would cover each other)
      var pad = n('rect', { x: 0, y: y + 6, width: 640, height: 26, fill: 'transparent', tabindex: 0,
        'data-hover': '1', 'aria-label': rowName(r) + ': read each model', style: 'cursor:crosshair;outline:none' });
      var near = function (evt) {
        var m = pad.getScreenCTM();
        if (!m) return;
        var pt = svg.createSVGPoint();
        pt.x = evt.clientX; pt.y = evt.clientY;
        var lx = pt.matrixTransform(m.inverse()).x, best = M, bd = 1e9;
        ix.forEach(function (i) {
          var d = Math.abs(sx(r.vals[i]) - lx);
          if (d < bd) { bd = d; best = i; }
        });
        read(r, best, cy);
      };
      ['mouseenter', 'mousemove', 'click'].forEach(function (ev) { pad.addEventListener(ev, near); });
      pad.addEventListener('focus', function () { read(r, M, cy); });
      pads.appendChild(pad);
    });
    var yb = Y0 + t.rows.length * RH + 6;
    // parked mid-strip until a hover moves it: at the default cx of 0 its radius ran
    // past the left edge, and the board's fit shifted the whole figure right
    ring = n('circle', { cx: 320, cy: Y0 + 18, r: 10, fill: 'none', stroke: C.ink, 'stroke-width': 1.6, opacity: 0,
      'pointer-events': 'none' });
    body.appendChild(ring);
    out = CH.readout(body, 0, yb, { size: 13, text: 'Hover a row to read any model\'s score.' });
    K.callout(body, 0, yb + 14, 640, summary(t), { cols: 74 });
  }

  // the foot note goes under whatever is drawn first, so reserve the height of the
  // tallest table: otherwise switching to a longer one runs into the note
  var low = 0;
  TAB.forEach(function (t) {
    var yb = Y0 + t.rows.length * RH + 6;
    low = Math.max(low, yb + 14 + K.wrap(summary(t), 72).length * 19 + 22);
  });
  svg.appendChild(n('rect', { x: 0, y: 0, width: 1, height: low, fill: 'none', 'pointer-events': 'none' }));

  K.switcher(swG, 0, 0, TN, draw, { tint: 'amber' });
  K.foot(svg, 'Scores as printed in the tables. Ranks count only the models a row reports, higher is better, ' +
    'and the blue dot is the best of the other six. A model with no score is left out of that row. Where ' +
    'ML4 and the blue dot nearly coincide, the two are nudged apart vertically.');
};
