/* S_DSA: section 3.2.3. A lightweight indexer picks the top 2,048 keys for
   each query; attention runs over those only. The slider's range is
   illustrative: the paper never states ML4's maximum context, so it runs from
   2,048 (the top-k itself) to 1M, the length of the YOCO prefill benchmark.
   The picked positions are illustrative; their total width is exact. The strip
   has 160 columns, round(160 x 2,048 / L) of them (at least one) hold a pick,
   and each pick is drawn W x 2,048 / L / n wide, so the amber adds up to exactly
   2,048 / L of the strip. The window sliver and the keys bar are to scale too.
   Marks thinner than THIN px get a small triangle below them (beside the strip
   for the window sliver) so they can be found; the triangles mark position,
   they add no width. */
window.SCENES.S_DSA = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var TOP = F['dsa.topk'], WIN = F['kv.swa_window'], S1 = F['dsa.stage1_b'], S2 = F['dsa.stage2_b'];
  var svg = K.board(host, { alt: 'DeepSeek Sparse Attention: keys read per query against context length, and the two adaptation stages' });
  K.head(svg, 'Sparse again, chosen per query', 'DeepSeek Sparse Attention · top ' + CH.comma(TOP) +
    ' keys per query');
  var X0 = 12, W = 616, XR = X0 + W, NCOL = 160;
  var LO = Math.round(Math.log2(TOP)), HI = Math.round(Math.log2(F['kv.yoco_ctx_m'] * 1048576));
  var short = function (L) { return L >= 1048576 ? (L / 1048576) + 'M' : (L / 1024) + 'K'; };

  K.label(svg, 0, 14, 'context length, illustrative');
  var out = CH.readout(svg, 640, 14, { anchor: 'end', size: 14 });

  // ---- what a query can reach: a fixed window, then DSA's picks ----
  K.label(svg, 0, 90, 'sliding window, before: last ' + WIN);
  K.mono(svg, 640, 90, 'the query is at the right end', { size: 12.5, anchor: 'end', color: C.ink3 });
  svg.appendChild(K.n('rect', { x: X0, y: 98, width: W, height: 20, rx: 3, fill: '#fff', stroke: CH.GREY2 }));
  var winG = CH.body(svg);
  K.label(svg, 0, 154, 'DSA: the indexer picks ' + CH.comma(TOP));
  svg.appendChild(K.n('rect', { x: X0, y: 170, width: W, height: 20, rx: 3, fill: '#fff', stroke: CH.GREY2 }));
  var pickG = CH.body(svg);

  // ---- keys read per query ----
  K.label(svg, 0, 222, 'keys read per query');
  K.mono(svg, 0, 248, 'full attention', { size: 13, color: C.ink2 });
  K.mono(svg, 0, 292, 'DSA, top ' + CH.comma(TOP), { size: 13, color: C.ink2 });
  var keysG = CH.body(svg);

  // ---- the two adaptation stages, to scale in tokens ----
  var sw1 = W * S1 / (S1 + S2);            // stage 1's share of the 15.1B + 198B tokens
  K.label(svg, 0, 346, 'adapting to DSA, tokens to scale');
  K.mono(svg, 0, 368, 'stage 1 · about ' + S1 + 'B · indexers distill the frozen model, full attention',
    { size: 12.5, color: C.ink2 });
  svg.appendChild(K.n('rect', { x: X0, y: 376, width: sw1, height: 26, fill: C.blue }));
  svg.appendChild(K.n('rect', { x: X0 + sw1 + 2, y: 376, width: W - sw1 - 2, height: 26, fill: C.amber }));
  K.mono(svg, X0 + sw1 + 12, 394, 'stage 2 · ' + S2 + 'B · trained together, sparse', { size: 13,
    weight: 600, color: '#fff' });
  var note = CH.body(svg);

  var L = 0, seed = 1, THIN = 2.5;
  // a small upward triangle under a mark, apex at y, to find marks too thin to see
  function pointer(p, x, y, col) {
    p.appendChild(K.n('path', { d: 'M' + x + ' ' + y + ' l-4.5 7 h9 Z', fill: col }));
  }
  function paintPicks() {
    CH.clear(pickG);
    // how many of the 160 columns hold a pick (at least one), and how wide each pick
    // is so that the amber adds up to exactly 2,048 / L of the strip
    var lit = Math.max(1, Math.min(NCOL, Math.round(NCOL * TOP / L))), cw = W / NCOL;
    var pw = W * TOP / L / lit;
    var idx = [], r = CH.rng(seed * 7919);
    for (var i = 0; i < NCOL; i++) idx.push(i);
    for (var k = NCOL - 1; k > 0; k--) { var j = Math.floor(r() * (k + 1)), tmp = idx[k]; idx[k] = idx[j]; idx[j] = tmp; }
    idx.slice(0, lit).forEach(function (c) {
      var px = X0 + c * cw + (cw - pw) / 2;
      pickG.appendChild(K.n('rect', { x: px, y: 171, width: pw, height: 18, fill: C.amber }));
      if (pw < THIN) pointer(pickG, px + pw / 2, 192, C.amber);
    });
  }
  function show(v) {
    L = Math.pow(2, v);
    var ratio = L / TOP;                    // both powers of two, so an exact whole number
    out.set(short(L) + ' tokens · ' + CH.comma(L));
    CH.clear(winG);
    var ww = W * WIN / L;                  // to scale; a triangle finds it when it is a hairline
    winG.appendChild(K.n('rect', { x: XR - ww, y: 99, width: ww, height: 18, fill: CH.GREY }));
    // the sliver sits at the strip's right end, above the button, so its triangle
    // points left at it from just past the end instead of up from below
    if (ww < THIN) winG.appendChild(K.n('path', { d: 'M' + (XR + 3) + ' 108 l7 -4.5 v9 Z', fill: CH.GREY }));
    paintPicks();
    CH.clear(keysG);
    keysG.appendChild(K.n('rect', { x: X0, y: 256, width: W, height: 14, rx: 2, fill: C.blue }));
    var kw = W / ratio;                    // to scale, down to 1.2 px at 1M
    keysG.appendChild(K.n('rect', { x: X0, y: 300, width: kw, height: 14, rx: Math.min(2, kw / 2), fill: C.amber }));
    if (kw < THIN) pointer(keysG, X0 + kw / 2, 317, C.amber);
    K.mono(keysG, 640, 248, CH.comma(L) + ' keys', { size: 13.5, anchor: 'end', weight: 600, color: C.ink });
    K.mono(keysG, 640, 292, ratio === 1 ? 'the same ' + CH.comma(TOP) : CH.comma(ratio) + '× fewer keys',
      { size: 13.5, anchor: 'end', weight: 600, color: C.ink });
    CH.clear(note);
    K.callout(note, 0, 420, 640, ratio === 1 ?
      'At ' + short(L) + ' tokens the top ' + CH.comma(TOP) + ' is the whole context, so DSA reads every key, ' +
        'just as full attention does. The saving grows with the context.' :
      'At ' + short(L) + ' tokens full attention reads ' + CH.comma(L) + ' keys per query; DSA reads ' +
        CH.comma(TOP) + ', ' + CH.comma(ratio) + '× fewer, picked by a lightweight indexer. From layer ' +
        F['dsa.share_from'] + ' on, each group of former window layers shares one indexer.', { cols: 74 });
  }

  CH.slider(svg, X0, 40, W, { min: LO, max: HI, step: 1, value: 17, label: 'context length',
    fmt: function (v) { return short(Math.pow(2, v)) + ' tokens'; }, onChange: show });
  CH.button(svg, 500, 134, 'another query', function () { seed++; paintPicks(); }, { w: 128 });
  K.foot(svg, 'The slider\'s range is illustrative: the paper does not give ML4\'s maximum context, and 1M is ' +
    'the length in its YOCO prefill test. Which keys get picked is illustrative, but every width is to ' +
    'scale; a small triangle points at any mark too thin to see.');
};
